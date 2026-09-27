#!/usr/bin/env python3
"""Cross-encoder stage on a cloud GPU (Kaggle "GPU T4 x2" notebook, or any CUDA machine: SageMaker, Colab, ...).

Self-contained: needs only torch, transformers, sentencepiece, pandas, pyarrow (all preinstalled on Kaggle) and the data
package written by export_ce_package.py:
    records.parquet      rid (int32), text (str)          serialized records ("name: ... address: ...")
    pairs_train.parquet  left, right (rid), label, domain  fine-tuning pairs (training entities only)
    pairs_eval.parquet   left, right, label, set           validation / hold-out pairs (monitoring only)
    pairs_score.parquet  pair_id, left, right, prio        every pair to score (val, hold-out, stack, test)

What it does (Ditto-style sequence-pair classification, "Business Entity Resolution Strategy" Stage 2):
  1. fine-tunes microsoft/mdeberta-v3-base (MIT, 280M params) as a cross-encoder on the training pairs, with a
     country-adversarial head behind a gradient-reversal layer (domain-invariant representation for the unseen France);
  2. scores every pair of pairs_score.parquet in priority order, writing resumable shards;
  3. writes /kaggle/working/ce_scores.parquet (pair_id, p_ce) + the fine-tuned weights (fp16) + a log.

Time safety: Kaggle kills a run at 12 h and then keeps NO output, so the script stops starting new work after
TIME_BUDGET_H hours and always writes what it has. A second run (attach the first run's output as an input dataset)
resumes: it reloads the fine-tuned model and skips the pairs already scored.

Usage in a Kaggle notebook cell:   !python /kaggle/input/<dataset-folder>/kaggle_ce.py
"""
from __future__ import annotations

import glob
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd

# ------------------------------------------------------------------ settings
MODEL_NAME = os.environ.get("CE_MODEL", "microsoft/mdeberta-v3-base")
MAX_LEN = 128
EPOCHS = float(os.environ.get("CE_EPOCHS", "1.0"))
LR = 3e-5
HEAD_LR = 1e-3
WEIGHT_DECAY = 0.01
WARMUP = 0.05
TRAIN_TOKENS_PER_GPU = 6144        # batch budget: rows x longest row, per GPU
SCORE_TOKENS_PER_GPU = 24576
ADV_WEIGHT = float(os.environ.get("CE_ADV", "0.1"))  # 0 disables the gradient-reversal country head
TIME_BUDGET_H = float(os.environ.get("CE_TIME_BUDGET_H", "11.0"))
SHARD_PAIRS = 500_000
SEED = 42
OUT = os.environ.get("CE_OUT", "/kaggle/working")
T_START = time.time()


def log(msg):
    line = f"[{(time.time() - T_START) / 60:7.1f} min] {msg}"
    print(line, flush=True)
    with open(os.path.join(OUT, "ce_log.txt"), "a", encoding="utf-8") as f:
        f.write(line + "\n")


def find_input(name):
    own = os.path.join(os.path.dirname(os.path.abspath(__file__)), name)
    if os.path.exists(own):                              # the package this script came with wins over other attached inputs
        return own
    extra = os.environ.get("CE_DATA")                    # local / SageMaker runs: the data folder (Kaggle: /kaggle/input)
    hits = sorted(glob.glob(f"/kaggle/input/**/{name}", recursive=True)) +         (sorted(glob.glob(os.path.join(extra, "**", name), recursive=True)) if extra else [])
    return hits[0] if hits else None


def time_left_h():
    return TIME_BUDGET_H - (time.time() - T_START) / 3600


# ------------------------------------------------------------------ model
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from transformers import AutoModel, AutoTokenizer  # noqa: E402

USE_BF16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8   # native bf16 only (Ampere+); T4 -> fp16
AMP_DTYPE = torch.bfloat16 if USE_BF16 else torch.float16
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SMOKE = os.environ.get("CE_SMOKE") == "1"          # tiny CPU functional test (never used for real scoring)


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


class CrossEncoder(nn.Module):
    def __init__(self, name, n_domains=2, adv=True):
        super().__init__()
        try:
            self.encoder = AutoModel.from_pretrained(name, dtype=torch.float32)
        except TypeError:                                    # transformers < 4.56 calls it torch_dtype
            self.encoder = AutoModel.from_pretrained(name, torch_dtype=torch.float32)
        self.encoder.float()
        h = self.encoder.config.hidden_size
        self.pool = nn.Sequential(nn.Dropout(0.1), nn.Linear(h, h), nn.GELU(), nn.Dropout(0.1))
        self.head = nn.Linear(h, 1)
        self.domain_head = nn.Sequential(nn.Linear(h, h // 2), nn.GELU(), nn.Linear(h // 2, n_domains)) if adv else None

    def forward(self, input_ids, attention_mask, adv_lambda=None):
        # autocast INSIDE forward: DataParallel runs replicas in side threads, where an outer autocast does not apply
        with torch.autocast("cuda", dtype=AMP_DTYPE, enabled=input_ids.is_cuda):
            cls = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
            logit = self.head(self.pool(cls)).squeeze(-1).float()
            if self.domain_head is not None and adv_lambda is not None and float(adv_lambda.flatten()[0]) > 0:
                dom = self.domain_head(GradReverse.apply(cls, float(adv_lambda.flatten()[0]))).float()
            else:
                dom = torch.zeros(input_ids.shape[0], 2, device=input_ids.device)
        return logit, dom


def pad(seqs, pad_id):
    n = max(len(s) for s in seqs)
    ids = np.full((len(seqs), n), pad_id, dtype=np.int64)
    mask = np.zeros((len(seqs), n), dtype=np.int64)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = s
        mask[i, :len(s)] = 1
    return torch.from_numpy(ids), torch.from_numpy(mask)


def batches(lengths, max_tokens, max_rows, rng=None):
    order = np.argsort(lengths + (rng.random(len(lengths)) * 4 if rng is not None else 0), kind="stable")
    out, cur, cur_max = [], [], 0
    for i in order:
        L = int(lengths[i])
        if cur and (max(cur_max, L) * (len(cur) + 1) > max_tokens or len(cur) >= max_rows):
            out.append(np.array(cur))
            cur, cur_max = [], 0
        cur.append(i)
        cur_max = max(cur_max, L)
    if cur:
        out.append(np.array(cur))
    if rng is not None:
        rng.shuffle(out)
    return out


def tokenize(tok, texts, left, right, chunk=200_000):
    # No [CLS]/[SEP]: the released model was fine-tuned on plain concatenated pairs (the Kaggle image's tokenizer added
    # no special tokens; reproduced exactly offline with add_special_tokens=False). Explicit, so any environment matches.
    seqs = []
    for a in range(0, len(left), chunk):
        l = [texts[i] for i in left[a:a + chunk]]
        r = [texts[i] for i in right[a:a + chunk]]
        enc = tok(l, r, truncation="longest_first", max_length=MAX_LEN, padding=False, add_special_tokens=False,
                  return_attention_mask=False, return_token_type_ids=False)
        seqs.extend(np.asarray(x, dtype=np.int32) for x in enc["input_ids"])
    return seqs


@torch.inference_mode()
def predict(model, seqs, pad_id, n_gpu, desc=""):
    lengths = np.fromiter((len(s) for s in seqs), dtype=np.int32, count=len(seqs))
    out = np.empty(len(seqs), dtype=np.float32)
    model.eval()
    t0 = last = time.time()
    done = 0
    for b in batches(lengths, SCORE_TOKENS_PER_GPU * max(1, n_gpu), 1024 * max(1, n_gpu)):
        ids, mask = pad([seqs[i] for i in b], pad_id)
        logit, _ = model(ids.to(DEV, non_blocking=True), mask.to(DEV, non_blocking=True))
        out[b] = torch.sigmoid(logit.float()).cpu().numpy()
        done += len(b)
        if time.time() - last > 120:
            last = time.time()
            log(f"  {desc} {done:,}/{len(seqs):,} ({done / (last - t0):,.0f} pairs/s)")
    return out


def auc(y, p):
    order = np.argsort(p)
    y = y[order]
    n1 = y.sum()
    n0 = len(y) - n1
    ranks = np.arange(1, len(y) + 1)
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / max(1, n0 * n1))


def main():
    os.makedirs(OUT, exist_ok=True)
    torch.manual_seed(SEED)
    n_gpu = torch.cuda.device_count()
    log(f"torch {torch.__version__}, GPUs: {[torch.cuda.get_device_name(i) for i in range(n_gpu)]}, amp {AMP_DTYPE}")
    if n_gpu == 0 and not SMOKE:
        sys.exit("no GPU: in Kaggle, Settings -> Accelerator -> GPU T4 x2")

    paths = {k: find_input(f"{k}.parquet") for k in ("records", "pairs_train", "pairs_eval", "pairs_score")}
    log(f"inputs: {paths}")
    rec = pd.read_parquet(paths["records"])
    texts = [""] * (int(rec["rid"].max()) + 1)
    for rid, t in zip(rec["rid"].to_numpy(), rec["text"].tolist()):
        texts[rid] = t
    del rec
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    pad_id = tok.pad_token_id

    # ---------------------------------------------------------- resume: fine-tuned model from an earlier run?
    prev_model = find_input("ce_model/ce_weights_fp16.pt")
    model = CrossEncoder(MODEL_NAME, adv=ADV_WEIGHT > 0)
    ev = pd.read_parquet(paths["pairs_eval"])
    if SMOKE:
        ev = ev.sample(n=min(len(ev), 300), random_state=SEED).reset_index(drop=True)
    seq_ev = tokenize(tok, texts, ev["left"].to_numpy(), ev["right"].to_numpy())
    y_ev = ev["label"].to_numpy().astype(np.int64)
    if prev_model:
        sd = torch.load(prev_model, map_location="cpu")
        model.load_state_dict({k: v.float() for k, v in sd.items()}, strict=False)
        model = model.to(DEV)
        dp = nn.DataParallel(model) if n_gpu > 1 else model
        log(f"resumed fine-tuned weights from {prev_model}")
    else:
        # ------------------------------------------------------ fine-tuning
        tr = pd.read_parquet(paths["pairs_train"])
        tr = tr.sample(frac=1.0 if not SMOKE else min(1.0, 400 / len(tr)), random_state=SEED).reset_index(drop=True)
        seq_tr = tokenize(tok, texts, tr["left"].to_numpy(), tr["right"].to_numpy())
        y_tr = torch.from_numpy(tr["label"].to_numpy().astype(np.float32))
        d_tr = torch.from_numpy(tr["domain"].to_numpy().astype(np.int64))
        lengths = np.fromiter((len(s) for s in seq_tr), dtype=np.int32, count=len(seq_tr))
        log(f"training pairs {len(seq_tr):,} (positives {float(y_tr.mean()):.3f}); tokens mean {lengths.mean():.1f} "
            f"p95 {np.percentile(lengths, 95):.0f}; eval pairs {len(seq_ev):,}")
        model = model.to(DEV)
        dp = nn.DataParallel(model) if n_gpu > 1 else model
        no_decay = lambda n: n.endswith(".bias") or "LayerNorm" in n  # noqa: E731
        groups = [
            {"params": [p for n, p in model.named_parameters() if n.startswith("encoder.") and not no_decay(n)], "lr": LR, "weight_decay": WEIGHT_DECAY},
            {"params": [p for n, p in model.named_parameters() if n.startswith("encoder.") and no_decay(n)], "lr": LR, "weight_decay": 0.0},
            {"params": [p for n, p in model.named_parameters() if not n.startswith("encoder.")], "lr": HEAD_LR, "weight_decay": 0.0},
        ]
        opt = torch.optim.AdamW(groups, lr=LR)
        base_lrs = [g["lr"] for g in opt.param_groups]
        scaler = torch.amp.GradScaler("cuda", enabled=(AMP_DTYPE == torch.float16 and DEV.type == "cuda"))
        rng = np.random.default_rng(SEED)
        n_dev = max(1, n_gpu)
        n_batches = len(batches(lengths, TRAIN_TOKENS_PER_GPU * n_dev, 64 * n_dev))
        total = max(1, int(math.ceil(n_batches * EPOCHS)))
        n_warm = max(1, int(total * WARMUP))
        bce, ce_loss = nn.BCEWithLogitsLoss(), nn.CrossEntropyLoss()
        step, t0, last, run_loss, run_dacc, run_n = 0, time.time(), time.time(), 0.0, 0.0, 0
        dp.train()
        log(f"fine-tuning {MODEL_NAME}: {total:,} steps ({EPOCHS} epoch(s)), adversarial weight {ADV_WEIGHT}")
        stop = False
        while step < total and not stop:
            for b in batches(lengths, TRAIN_TOKENS_PER_GPU * n_dev, 64 * n_dev, rng):
                if step >= total:
                    break
                if time_left_h() < 2.5:            # keep >= 2.5 h for scoring
                    log("time budget: stopping fine-tuning early")
                    stop = True
                    break
                frac = step / n_warm if step < n_warm else max(0.0, (total - step) / max(1, total - n_warm))
                for pg, blr in zip(opt.param_groups, base_lrs):
                    pg["lr"] = blr * frac
                ids, mask = pad([seq_tr[i] for i in b], pad_id)
                lam = ADV_WEIGHT * (2.0 / (1.0 + math.exp(-10 * step / total)) - 1.0) if ADV_WEIGHT > 0 else 0.0
                lam_t = torch.full((len(b), 1), lam)
                logit, dom = dp(ids.to(DEV, non_blocking=True), mask.to(DEV, non_blocking=True), lam_t.to(DEV))
                y = y_tr[b].to(DEV, non_blocking=True)
                loss = bce(logit, y)
                if lam > 0:
                    d = d_tr[b].to(DEV, non_blocking=True)
                    loss = loss + ce_loss(dom, d)
                    run_dacc += float((dom.argmax(1) == d).float().sum())
                if not torch.isfinite(loss):
                    opt.zero_grad(set_to_none=True)
                    step += 1
                    continue
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                step += 1
                run_loss += float(loss.detach()) * len(b)
                run_n += len(b)
                if time.time() - last > 300:
                    last = time.time()
                    log(f"  step {step:,}/{total:,} loss {run_loss / run_n:.4f} country-head acc {run_dacc / run_n:.3f} "
                        f"{run_n / (last - t0):,.0f} pairs/s, ETA {(total - step) * (last - t0) / step / 60:.0f} min")
                if step % max(1, total // 4) == 0:
                    sub = np.random.default_rng(0).choice(len(seq_ev), size=min(20000, len(seq_ev)), replace=False)
                    p = predict(dp, [seq_ev[i] for i in sub], pad_id, n_gpu)
                    log(f"  [eval @ step {step:,}] AUC {auc(y_ev[sub], p):.5f}")
                    dp.train()
        os.makedirs(os.path.join(OUT, "ce_model"), exist_ok=True)
        torch.save({k: v.half() for k, v in model.state_dict().items()}, os.path.join(OUT, "ce_model", "ce_weights_fp16.pt"))
        tok.save_pretrained(os.path.join(OUT, "ce_model"))
        json.dump({"model": MODEL_NAME, "max_len": MAX_LEN, "epochs": EPOCHS, "lr": LR, "adv_weight": ADV_WEIGHT,
                   "train_pairs": len(seq_tr), "steps": step}, open(os.path.join(OUT, "ce_model", "ce_config.json"), "w"), indent=2)
        log(f"fine-tuning done ({step:,} steps); weights saved")
        del seq_tr

    # ---------------------------------------------------------- evaluation on validation / hold-out pairs
    p_ev = predict(dp, seq_ev, pad_id, n_gpu, "eval")
    for s in sorted(ev["set"].unique()):
        m = (ev["set"] == s).to_numpy()
        log(f"[{s}] pairs {m.sum():,} AUC {auc(y_ev[m], p_ev[m]):.5f} logloss "
            f"{float(-np.mean(y_ev[m] * np.log(np.clip(p_ev[m], 1e-6, 1)) + (1 - y_ev[m]) * np.log(np.clip(1 - p_ev[m], 1e-6, 1)))):.4f}")

    # ---------------------------------------------------------- scoring, resumable shards, one worker process per GPU
    sc = pd.read_parquet(paths["pairs_score"]).sort_values(["prio", "pair_id"], kind="stable").reset_index(drop=True)
    if SMOKE:
        sc = sc.groupby("prio", group_keys=False).head(150).reset_index(drop=True)
    done_ids = set()
    extra = os.environ.get("CE_DATA")
    prev = sorted(glob.glob("/kaggle/input/**/ce_scores*.parquet", recursive=True)
                  + (glob.glob(os.path.join(extra, "**", "ce_scores*.parquet"), recursive=True) if extra else []))
    for i, f in enumerate(prev):                          # carry an earlier run's results of THIS package's pairs
        d = pd.read_parquet(f)
        d = d[d["pair_id"].isin(sc["pair_id"].to_numpy())]
        if len(d):
            done_ids.update(d["pair_id"].tolist())
            d.to_parquet(os.path.join(OUT, f"ce_scores_part_prev{i:03d}.parquet"), index=False)
    for f in glob.glob(os.path.join(OUT, "ce_scores_part*.parquet")):
        done_ids.update(pd.read_parquet(f, columns=["pair_id"])["pair_id"].tolist())
    if done_ids:
        sc = sc[~sc["pair_id"].isin(done_ids)].reset_index(drop=True)
    total_needed = len(pd.read_parquet(paths["pairs_score"], columns=["pair_id"])) if not SMOKE else len(sc) + len(done_ids)
    log(f"pairs to score: {len(sc):,} (already scored: {len(done_ids):,})")
    todo = os.path.join(OUT, "_todo.parquet")
    sc[["pair_id", "left", "right"]].to_parquet(todo, index=False)
    weights = os.path.join(OUT, "ce_model", "ce_weights_fp16.pt")
    if not os.path.exists(weights):                       # resumed run: keep the weights next to this run's output too
        os.makedirs(os.path.join(OUT, "ce_model"), exist_ok=True)
        torch.save({k: v.half() for k, v in model.state_dict().items()}, weights)
    del dp, model
    torch.cuda.empty_cache()
    n_workers = max(1, n_gpu) if not SMOKE else int(os.environ.get("CE_SMOKE_WORKERS", "1"))
    if n_workers == 1:
        score_worker(0, 1, todo, weights, paths["records"])
    else:
        import subprocess
        procs = []
        for k in range(n_workers):
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(k), "CE_T_START": str(T_START)}
            procs.append(subprocess.Popen([sys.executable, os.path.abspath(__file__), "--worker", str(k), str(n_workers),
                                           todo, weights, paths["records"]], env=env))
        for pr in procs:
            pr.wait()
        log(f"workers finished with exit codes {[pr.returncode for pr in procs]}")

    parts = sorted(glob.glob(os.path.join(OUT, "ce_scores_part*.parquet")))
    allp = pd.concat([pd.read_parquet(f) for f in parts], ignore_index=True).drop_duplicates("pair_id")
    allp.to_parquet(os.path.join(OUT, "ce_scores.parquet"), index=False)
    for f in parts + [todo]:
        os.remove(f)
    log(f"ce_scores.parquet: {len(allp):,} of {total_needed:,} pairs scored "
        f"({'COMPLETE' if len(allp) >= total_needed else 'INCOMPLETE - run again with this output attached'})")


def score_worker(k: int, n: int, todo_path: str, weights: str, records_path: str):
    """Score shards k, k+n, k+2n, ... of the pending pairs on the single visible GPU; one parquet file per shard."""
    rec = pd.read_parquet(records_path)
    texts = [""] * (int(rec["rid"].max()) + 1)
    for rid, t in zip(rec["rid"].to_numpy(), rec["text"].tolist()):
        texts[rid] = t
    del rec
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model = CrossEncoder(MODEL_NAME, adv=ADV_WEIGHT > 0)
    model.load_state_dict({kk: v.float() for kk, v in torch.load(weights, map_location="cpu").items()}, strict=False)
    model = model.to(DEV)
    sc = pd.read_parquet(todo_path)
    shards = list(range(0, len(sc), SHARD_PAIRS))[k::n]
    t0, n_done, n_mine = time.time(), 0, sum(min(SHARD_PAIRS, len(sc) - a) for a in shards)
    for j, a in enumerate(shards):
        if time_left_h() < 0.25:
            log(f"[worker {k}] time budget reached: stopping (attach this run's output to a new run to finish)")
            break
        part = sc.iloc[a:a + SHARD_PAIRS]
        seqs = tokenize(tok, texts, part["left"].to_numpy(), part["right"].to_numpy())
        p = predict(model, seqs, tok.pad_token_id, 1, f"worker {k} shard {j}")
        pd.DataFrame({"pair_id": part["pair_id"].to_numpy(), "p_ce": p}).to_parquet(
            os.path.join(OUT, f"ce_scores_part_w{k}_{a:010d}.parquet"), index=False)
        n_done += len(part)
        rate = n_done / (time.time() - t0)
        log(f"[worker {k}] {n_done:,}/{n_mine:,} scored ({rate:,.0f} pairs/s, ~{(n_mine - n_done) / max(rate, 1) / 60:.0f} min left)")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        T_START = float(os.environ.get("CE_T_START", T_START))
        score_worker(int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5], sys.argv[6])
    else:
        main()
