# Fine-tuned cross-encoder: business-record matcher

`microsoft/mdeberta-v3-base` (MIT licence, 278M parameters, multilingual), fine-tuned to decide whether two business
records (name + address) describe the same business. Trained by `src/kaggle_ce.py` on 1,169,095 labelled pairs from
the challenge training data (52.8% matches, including 40,000 "same name and house number, different street" negatives).
It uses a gradient-reversal country head (weight 0.1). Training ran for 1 epoch on 2x T4 (Kaggle), taking 96 min.
Pair AUC is 0.9985 on validation and on hold-out entities. On its own, it reaches a hold-out macro F0.5 of 0.9725 with
the challenge metric.

| file | |
|---|---|
| `ce_weights_fp16.pt` | state dict of the `CrossEncoder` module in `src/kaggle_ce.py`, fp16 (558 MB, stored with Git LFS) |
| `tokenizer.json`, `tokenizer_config.json` | the mDeBERTa tokenizer (unchanged) |
| `ce_config.json` | training settings |

**Download:** `git lfs install` then `git clone` (or `git lfs pull` in an existing clone). Without Git LFS you only get
a small pointer file.

**Input format:** each record becomes `name: <name> address: <address>`, lower-cased and trimmed (empty string when
missing). The two records are given as a text pair, truncated to 128 tokens together, **without special tokens**
(`add_special_tokens=False`). The model was fine-tuned on plain concatenated pairs, and the head reads the first
token. With the default `[CLS] … [SEP]` input, the scores come out visibly less confident. With
`add_special_tokens=False`, they reproduce the training run's scores exactly (mean absolute difference 0.0001 on
validation pairs, any transformers version from 4.57 to 5.17). The output is a match probability.

```python
import sys, torch
sys.path.insert(0, "code/business_entity_resolution/src")
from kaggle_ce import CrossEncoder, MAX_LEN            # the model class used in training
from transformers import AutoTokenizer

d = "code/business_entity_resolution/artifacts/ce/ce_model"
tok = AutoTokenizer.from_pretrained(d)
model = CrossEncoder("microsoft/mdeberta-v3-base", adv=True)   # downloads the base architecture once
sd = torch.load(f"{d}/ce_weights_fp16.pt", map_location="cpu")
model.load_state_dict({k: v.float() for k, v in sd.items()}, strict=False)
model.eval()

def text(name, address):
    return f"name: {(name or '').strip().lower()} address: {(address or '').strip().lower()}"

a = [text("Rev It Up Cafe", "248 Plaza Gardens Court, Unit 3B, Camdenton, MO")]
b = [text("REV IT-UP CAFE", "248 PLAZA GARDENS CT, CAMDENTON, MO")]
enc = tok(a, b, truncation="longest_first", max_length=MAX_LEN, padding=True, return_tensors="pt",
          add_special_tokens=False, return_token_type_ids=False)
with torch.inference_mode():
    logit, _ = model(enc["input_ids"], enc["attention_mask"])
print(torch.sigmoid(logit))      # match probability per pair
```

On a GPU, move `model` and the tensors to `"cuda"`: the forward pass then runs in fp16 (T4) or bf16 (Ampere+) automatically.
