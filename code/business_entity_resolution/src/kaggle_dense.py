#!/usr/bin/env python3
"""Dense candidate retrieval on a cloud GPU (Kaggle "GPU T4 x2", SageMaker, ...): fine-tuned multilingual bi-encoder.

Self-contained (torch, transformers, pandas, pyarrow); input = the package written by export_dense_package.py:
  records.parquet (rid, text), partitions.parquet, train_pairs.parquet (q, pos, neg), existing.parquet (q, p),
  eval_pairs.parquet (q, p)
Steps
  1. fine-tune intfloat/multilingual-e5-small (MIT) as a bi-encoder: InfoNCE over in-batch negatives + one hard negative
     per pair (a candidate of the same entity that is not a match), temperature 0.05, both directions;
  2. per (split, country) partition -- one worker process per GPU -- embed every pool record and every queried S1 record,
     search each query's TOPK nearest pool records (cosine, exact search on the GPU);
  3. drop the queries' existing candidates, keep the KEEP_NEW best new ones -> dense_new.parquet (q, p, score, rank),
     and measure on validation / hold-out true pairs how much recall the new candidates add -> dense_recall.json.
Outputs in /kaggle/working: dense_new.parquet, dense_recall.json, dense_log.txt, bienc_model/ (fp16 weights).
Usage in a Kaggle notebook cell:   !python $(find /kaggle/input -name kaggle_dense.py | head -n 1)
"""
from __future__ import annotations

import glob
import json
import math
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

MODEL = os.environ.get("DE_MODEL", "intfloat/multilingual-e5-small")
MAX_LEN = 64
BATCH = int(os.environ.get("DE_BATCH", "512"))            # training pairs per step (split across GPUs)
EPOCHS = float(os.environ.get("DE_EPOCHS", "1.0"))
LR = 5e-5
TEMP = 0.05
TOPK = 30
KEEP_NEW = 10
ENC_TOKENS = 65536                                         # tokens per encoding batch
EMBED_CHUNK = int(os.environ.get("DE_EMBED_CHUNK", "200000"))   # texts tokenized at a time (bounded RAM)
Q_CHUNK, P_BLOCK = 1024, 1_000_000                         # exact search tiling
TIME_BUDGET_H = float(os.environ.get("DE_TIME_BUDGET_H", "11.0"))
OUT = os.environ.get("DE_OUT", "/kaggle/working")
SMOKE = os.environ.get("DE_SMOKE") == "1"
PREFIX = "query: "                                         # e5 convention; symmetric task -> same prefix both sides
T_START = float(os.environ.get("DE_T_START", time.time()))


def log(msg):
    line = f"[{(time.time() - T_START) / 60:7.1f} min] {msg}"
    print(line, flush=True)
    with open(os.path.join(OUT, "dense_log.txt"), "a", encoding="utf-8") as f:
        f.write(line + "\n")


def find_input(name):
    extra = os.environ.get("DE_DATA")
    hits = sorted(glob.glob(f"/kaggle/input/**/{name}", recursive=True)) + \
        (sorted(glob.glob(os.path.join(extra, "**", name), recursive=True)) if extra else [])
    return hits[0] if hits else None


os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")      # keep the notebook log readable
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from transformers import AutoModel, AutoTokenizer  # noqa: E402

try:
    from transformers.utils import logging as _hf_logging
    _hf_logging.disable_progress_bar()
except Exception:  # noqa: BLE001 - cosmetic only
    pass

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16
EMB = torch.float16 if torch.cuda.is_available() else torch.float32     # half-precision matmul is emulated (slow) on CPU


class BiEncoder(nn.Module):
    def __init__(self, name):
        super().__init__()
        try:
            self.enc = AutoModel.from_pretrained(name, dtype=torch.float32)
        except TypeError:
            self.enc = AutoModel.from_pretrained(name, torch_dtype=torch.float32)
        self.enc.float()

    def forward(self, ids, mask):
        with torch.autocast("cuda", dtype=AMP, enabled=ids.is_cuda):
            h = self.enc(input_ids=ids, attention_mask=mask).last_hidden_state
        m = mask.unsqueeze(-1).float()
        e = (h.float() * m).sum(1) / m.sum(1).clamp(min=1.0)      # mean pooling (e5)
        return F.normalize(e, dim=-1)


def read_texts(path, lo, hi):
    t = pq.read_table(path, filters=[("rid", ">=", int(lo)), ("rid", "<", int(hi))])
    r = t["rid"].to_numpy()
    txt = t["text"].to_pylist()
    out = [""] * (hi - lo)
    for i, x in zip(r - lo, txt):
        out[i] = x
    return out


def read_text_array(path, lo, hi):
    """Texts of rids [lo, hi) as ONE Arrow string array in rid order (compact: no Python objects until sliced)."""
    import pyarrow as pa
    t = pq.read_table(path, filters=[("rid", ">=", int(lo)), ("rid", "<", int(hi))]).sort_by("rid")
    r = t["rid"].to_numpy()
    if len(r) == hi - lo and (len(r) == 0 or (r[0] == lo and r[-1] == hi - 1)):
        return t["text"].combine_chunks()
    return pa.array(read_texts(path, lo, hi), pa.large_string())     # gaps: fall back to the slow path


def embed_range(model, tok, path, lo, hi, label):
    """Embed rids [lo, hi) chunk by chunk: tokenizing millions of texts at once needs tens of GB of RAM (the Rust
    encodings and Python id lists), so only EMBED_CHUNK texts are ever tokenized at a time."""
    texts = read_text_array(path, lo, hi)
    out = torch.empty((hi - lo, model.enc.config.hidden_size), dtype=EMB, device=DEV)
    t0 = last = time.time()
    for a in range(0, hi - lo, EMBED_CHUNK):
        n = min(EMBED_CHUNK, hi - lo - a)
        seqs = tokenize(tok, texts.slice(a, n).to_pylist())
        out[a:a + n] = encode(model, seqs, tok.pad_token_id)
        del seqs
        if time.time() - last > 120:
            last = time.time()
            log(f"    {label}: {a + n:,}/{hi - lo:,} embedded ({(a + n) / (last - t0):,.0f}/s)")
    return out


def tokenize(tok, texts, max_len=MAX_LEN):
    enc = tok([PREFIX + x for x in texts], truncation=True, max_length=max_len, padding=False,
              return_attention_mask=False, return_token_type_ids=False)
    return [np.asarray(x, dtype=np.int32) for x in enc["input_ids"]]


def pad(seqs, pad_id):
    n = max(len(s) for s in seqs)
    ids = np.full((len(seqs), n), pad_id, dtype=np.int64)
    mask = np.zeros((len(seqs), n), dtype=np.int64)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = s
        mask[i, :len(s)] = 1
    return torch.from_numpy(ids), torch.from_numpy(mask)


@torch.inference_mode()
def encode(model, seqs, pad_id):
    lengths = np.fromiter((len(s) for s in seqs), dtype=np.int32, count=len(seqs))
    order = np.argsort(lengths, kind="stable")
    out = torch.empty((len(seqs), model.enc.config.hidden_size), dtype=EMB, device=DEV)
    i, n_all = 0, len(order)
    while i < n_all:
        L = int(lengths[order[min(n_all - 1, i + 4095)]])       # lengths ascend: an upper bound for the next batch
        n = max(1, min(4096, ENC_TOKENS // max(L, 1)))
        b = order[i:i + n]
        ids, mask = pad([seqs[j] for j in b], pad_id)
        out[torch.from_numpy(b).to(DEV)] = model(ids.to(DEV), mask.to(DEV)).to(EMB)
        i += len(b)
    return out


def train(tok, rec_path, n_gpu):
    tr = pd.read_parquet(find_input("train_pairs.parquet"))
    if SMOKE:
        tr = tr.head(2000)
    tr = tr.sample(frac=1.0, random_state=42).reset_index(drop=True)
    need = np.unique(np.concatenate([tr.q.to_numpy(), tr.pos.to_numpy(), tr.neg.to_numpy()[tr.neg.to_numpy() >= 0]]))
    t = pq.read_table(rec_path, filters=[("rid", "in", need.tolist())])
    text = dict(zip(t["rid"].to_numpy().tolist(), t["text"].to_pylist()))
    seq = dict(zip(text.keys(), tokenize(tok, list(text.values()))))
    log(f"fine-tuning pairs {len(tr):,} (hard negative for {float((tr.neg >= 0).mean()):.1%}); {len(seq):,} distinct records tokenized")
    model = BiEncoder(MODEL).to(DEV)
    dp = nn.DataParallel(model) if n_gpu > 1 else model
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=(AMP == torch.float16 and DEV.type == "cuda"))
    steps = max(1, int(math.ceil(len(tr) / BATCH * EPOCHS)))
    warm = max(1, int(0.05 * steps))
    pad_id = tok.pad_token_id
    dp.train()
    t0 = last = time.time()
    run = 0.0
    for step in range(steps):
        if (time.time() - T_START) / 3600 > TIME_BUDGET_H - 6:
            log("time budget: stopping fine-tuning early")
            break
        a = (step * BATCH) % len(tr)
        b = tr.iloc[a:a + BATCH]
        for g in opt.param_groups:
            g["lr"] = LR * (step / warm if step < warm else max(0.0, (steps - step) / max(1, steps - warm)))
        q = dp(*[x.to(DEV) for x in pad([seq[r] for r in b.q], pad_id)])
        p = dp(*[x.to(DEV) for x in pad([seq[r] for r in b.pos], pad_id)])
        negs = [seq[r] for r in b.neg if r >= 0]
        cand = torch.cat([p, dp(*[x.to(DEV) for x in pad(negs, pad_id)])]) if negs else p
        tgt = torch.arange(len(b), device=DEV)
        loss = F.cross_entropy(q @ cand.T / TEMP, tgt) + F.cross_entropy(p @ q.T / TEMP, tgt)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        run = 0.98 * run + 0.02 * float(loss) if step else float(loss)
        if time.time() - last > 300 or step == steps - 1:
            last = time.time()
            log(f"  step {step + 1:,}/{steps:,} loss {run:.4f} ({(step + 1) * BATCH / (last - t0):,.0f} pairs/s)")
    os.makedirs(os.path.join(OUT, "bienc_model"), exist_ok=True)
    torch.save({k: v.half() for k, v in model.state_dict().items()}, os.path.join(OUT, "bienc_model", "bienc_fp16.pt"))
    tok.save_pretrained(os.path.join(OUT, "bienc_model"))
    json.dump({"model": MODEL, "max_len": MAX_LEN, "batch": BATCH, "epochs": EPOCHS, "lr": LR, "temperature": TEMP,
               "pairs": len(tr)}, open(os.path.join(OUT, "bienc_model", "bienc_config.json"), "w"), indent=2)
    log("fine-tuning done; weights saved")


def search_worker(parts_json, rec_path, weights):
    """Embed + exact kNN for the given partitions on the single visible GPU; one parquet of neighbours per partition."""
    parts = json.loads(parts_json)
    tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
    model = BiEncoder(MODEL)
    model.load_state_dict({k: v.float() for k, v in torch.load(weights, map_location="cpu").items()})
    model = model.to(DEV).eval()
    for part in parts:
        if part["query_lo"] < 0:
            continue
        name = part["part"].replace("/", "_")
        dst = os.path.join(OUT, f"_nn_{name}.parquet")
        if os.path.exists(dst):
            continue
        t0 = time.time()
        log(f"  [{part['part']}] start: {part['pool_hi'] - part['pool_lo']:,} pool, {part['query_hi'] - part['query_lo']:,} queries")
        pool = embed_range(model, tok, rec_path, part["pool_lo"], part["pool_hi"], f"[{part['part']}] pool")
        qry = embed_range(model, tok, rec_path, part["query_lo"], part["query_hi"], f"[{part['part']}] queries")
        log(f"  [{part['part']}] embedded {len(pool):,} pool + {len(qry):,} queries ({time.time() - t0:.0f}s)")
        k = min(TOPK, len(pool))
        best_s = torch.empty((len(qry), k), dtype=EMB)
        best_i = torch.empty((len(qry), k), dtype=torch.int64)
        with torch.inference_mode():
            for a in range(0, len(qry), Q_CHUNK):
                qa = qry[a:a + Q_CHUNK]
                vals, idx = [], []
                for pb in range(0, len(pool), P_BLOCK):
                    s = qa @ pool[pb:pb + P_BLOCK].T
                    v, i = torch.topk(s, min(k, s.shape[1]), dim=1)
                    vals.append(v)
                    idx.append(i + pb)
                v, j = torch.topk(torch.cat(vals, 1), k, dim=1)
                best_s[a:a + Q_CHUNK] = v.cpu()
                best_i[a:a + Q_CHUNK] = torch.gather(torch.cat(idx, 1), 1, j).cpu()
        q = np.repeat(np.arange(part["query_lo"], part["query_hi"], dtype=np.int64), k)
        p = best_i.numpy().reshape(-1) + part["pool_lo"]
        pd.DataFrame({"q": q, "p": p, "score": best_s.numpy().reshape(-1).astype(np.float32),
                      "rank": np.tile(np.arange(k, dtype=np.int16), len(qry))}).to_parquet(dst + ".tmp", index=False)
        os.replace(dst + ".tmp", dst)
        log(f"  [{part['part']}] searched {len(qry):,} queries x {len(pool):,} pool ({time.time() - t0:.0f}s)")
        del pool, qry
        torch.cuda.empty_cache()


def main():
    os.makedirs(OUT, exist_ok=True)
    n_gpu = torch.cuda.device_count()
    log(f"torch {torch.__version__}, GPUs {[torch.cuda.get_device_name(i) for i in range(n_gpu)]}, amp {AMP}, model {MODEL}")
    if n_gpu == 0 and not SMOKE:
        sys.exit("no GPU: in Kaggle, Settings -> Accelerator -> GPU T4 x2")
    rec_path = find_input("records.parquet")
    parts = pd.read_parquet(find_input("partitions.parquet"))
    tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
    weights = os.path.join(OUT, "bienc_model", "bienc_fp16.pt")
    prev = find_input("bienc_model/bienc_fp16.pt")
    if prev and not os.path.exists(weights):
        os.makedirs(os.path.dirname(weights), exist_ok=True)
        import shutil
        shutil.copyfile(prev, weights)
        log(f"resumed fine-tuned weights from {prev}")
    if not os.path.exists(weights):
        # fine-tune in a separate process: when it exits, ALL its GPU memory is released (the caching allocator of a
        # process that trained keeps ~13 GB per T4 reserved, which starves the search workers)
        r = subprocess.run([sys.executable, os.path.abspath(__file__), "--train"], env={**os.environ, "DE_T_START": str(T_START)})
        if r.returncode != 0 or not os.path.exists(weights):
            log(f"fine-tuning failed (exit code {r.returncode})")
            sys.exit(1)
    for f in glob.glob("/kaggle/input/**/_nn_*.parquet", recursive=True):      # neighbours of an earlier run
        dst = os.path.join(OUT, os.path.basename(f))
        if not os.path.exists(dst):
            pd.read_parquet(f).to_parquet(dst, index=False)
    # balance partitions over the GPUs by work (pool size x queries)
    work = [(r.pool_hi - r.pool_lo) * max(1, r.query_hi - r.query_lo) ** 0.5 for r in parts.itertuples()]
    order = np.argsort(work)[::-1]
    n_workers = max(1, n_gpu)
    buckets, load = [[] for _ in range(n_workers)], [0.0] * n_workers
    for i in order:
        k = int(np.argmin(load))
        buckets[k].append(parts.iloc[i].to_dict())
        load[k] += work[i]
    if n_workers == 1:
        search_worker(json.dumps(buckets[0], default=int), rec_path, weights)
    else:
        procs = []
        for k in range(n_workers):
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(k), "DE_T_START": str(T_START)}
            procs.append(subprocess.Popen([sys.executable, os.path.abspath(__file__), "--worker",
                                           json.dumps(buckets[k], default=int), rec_path, weights], env=env))
        for pr in procs:
            pr.wait()
        log(f"search workers finished with exit codes {[pr.returncode for pr in procs]}")
    missing = [r["part"] for r in parts.to_dict("records") if r["query_lo"] >= 0
               and not os.path.exists(os.path.join(OUT, f"_nn_{r['part'].replace('/', '_')}.parquet"))]
    if missing:
        log(f"INCOMPLETE: no neighbours for {missing}. Finished partitions are kept in {OUT}/_nn_*.parquet; "
            f"attach this output to a new run to resume.")
        sys.exit(1)
    nn_files = sorted(glob.glob(os.path.join(OUT, "_nn_*.parquet")))
    nn = pd.concat([pd.read_parquet(f) for f in nn_files], ignore_index=True)
    ex = pd.read_parquet(find_input("existing.parquet"))
    ex["have"] = True
    nn = nn.merge(ex, on=["q", "p"], how="left")
    new = nn[nn["have"].isna()].drop(columns="have").sort_values(["q", "rank"])
    new["new_rank"] = new.groupby("q").cumcount()
    keep = new[new.new_rank < KEEP_NEW]
    keep[["q", "p", "score", "rank", "new_rank"]].to_parquet(os.path.join(OUT, "dense_new.parquet"), index=False)
    ev = pd.read_parquet(find_input("eval_pairs.parquet"))
    res = {"queries": int(nn.q.nunique()), "new_pairs_kept": len(keep), "new_per_query": len(keep) / max(1, nn.q.nunique())}
    if len(ev):
        ev = ev.merge(ex, on=["q", "p"], how="left")
        base_hit = ev["have"].notna()
        for kk in (1, 3, 5, 10):
            s = new[new.new_rank < kk][["q", "p"]].assign(hit=True)
            h = ev[["q", "p"]].merge(s, on=["q", "p"], how="left")["hit"].notna().to_numpy()
            res[f"recall_existing_plus_new@{kk}"] = float((base_hit.to_numpy() | h).mean())
        res["recall_existing"] = float(base_hit.mean())
        res["eval_true_pairs"] = len(ev)
        res["missed_recovered_share@10"] = float((~base_hit.to_numpy() & ev[["q", "p"]].merge(
            new[new.new_rank < 10][["q", "p"]].assign(hit=True), on=["q", "p"], how="left")["hit"].notna().to_numpy()).sum()
            / max(1, (~base_hit).sum()))
    json.dump(res, open(os.path.join(OUT, "dense_recall.json"), "w"), indent=2)
    log(f"dense_new.parquet written: {json.dumps(res)}")
    for f in nn_files:
        os.remove(f)
    log("COMPLETE")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        search_worker(sys.argv[2], sys.argv[3], sys.argv[4])
    elif len(sys.argv) > 1 and sys.argv[1] == "--train":
        train(AutoTokenizer.from_pretrained(MODEL, use_fast=True), find_input("records.parquet"), torch.cuda.device_count())
    else:
        main()
