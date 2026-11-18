"""Shared setup for the DSL ad-tone notebooks: load the ads, get LLM labels, and a few small helpers.

LLM labels are saved in data/ (one file per model + prompt), keyed by ad ID, so the LLM only runs
for ads that have no saved label yet.
"""
try:
    import torch  # import before scikit-learn: the other order can crash model loading on Macs
except ImportError:  # fine: torch is only needed to make new LLM labels
    torch = None

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

DATA_DIR = Path(__file__).resolve().parent / "data"
CHOICES = ["Promote", "Contrast", "Attack"]


def load_ads(n_corpus, seed):
    """A random sample of n_corpus ads, with Y = 1 if the expert coded the ad negative, X = 1 if Facebook."""
    ads = pd.read_csv(DATA_DIR / "ads.csv", dtype={"snapshot_id": str})
    ads = ads.sample(n_corpus, random_state=seed).reset_index(drop=True)
    ads["Y"] = ads.tone.isin(["Attack", "Contrast"]).astype(int)
    ads["X"] = (ads.platform == "Facebook").astype(int)
    return ads


def ols(y, x):
    """Slope and robust (HC1) standard error from regressing y on x."""
    fit = sm.OLS(np.asarray(y, dtype=float), sm.add_constant(np.asarray(x, dtype=float))).fit(cov_type="HC1")
    return fit.params[1], fit.bse[1]


def draw(n_corpus, n_expert, seed):
    """R[i] = 1 if ad i is in the random sample the experts label."""
    R = np.zeros(n_corpus, dtype=int)
    R[np.random.default_rng(seed).choice(n_corpus, n_expert, replace=False)] = 1
    return R


def labels_path(model, system, max_chars):
    key = hashlib.sha1(f"{system}|{max_chars}".encode()).hexdigest()[:8]
    return DATA_DIR / f"llm_labels_{model.split('/')[-1]}_{key}.csv"


def llm_labels(ads, model, system, max_chars=600, batch=64):
    """LLM tone label (Promote / Contrast / Attack) for every ad, using saved labels where they exist."""
    path = labels_path(model, system, max_chars)
    saved = pd.read_csv(path, dtype={"snapshot_id": str}) if path.exists() else pd.DataFrame(columns=["snapshot_id", "Q_tone"])
    known = dict(zip(saved.snapshot_id, saved.Q_tone))
    todo = ads[~ads.snapshot_id.isin(known)]
    if len(todo) == 0:
        print(f"Loaded saved LLM labels from data/{path.name}")
    else:
        print(f"{len(todo)} ads have no saved label for this model and prompt; running {model}")
        known.update(zip(todo.snapshot_id, _run_llm(todo.text.tolist(), model, system, max_chars, batch)))
        out = pd.DataFrame(sorted(known.items()), columns=["snapshot_id", "Q_tone"])
        out.to_csv(path, index=False)
        print(f"Saved to data/{path.name}")
    return ads.snapshot_id.map(known)


def _run_llm(texts, model_name, system, max_chars, batch):
    if torch is None:
        raise ImportError("No saved labels for this model and prompt, and the LLM packages are not installed.\n"
                          "Run `uv sync --extra llm`, restart the kernel, and run again.")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print("Running on", device, "(slow!)" if device == "cpu" else "")
    tok = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float32 if device == "cpu" else torch.float16).to(device).eval()
    first_tok = [tok.encode(c, add_special_tokens=False)[0] for c in CHOICES]
    assert len(set(first_tok)) == len(CHOICES)

    def label(chunk):
        # One forward pass: pick whichever answer's first token the model rates most likely.
        prompts = [tok.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": "Ad: " + t[:max_chars]}],
            tokenize=False, add_generation_prompt=True) for t in chunk]
        enc = tok(prompts, return_tensors="pt", padding=True).to(device)
        pos = (enc.attention_mask.cumsum(-1) - 1).clamp(min=0)  # correct positions with left padding
        with torch.no_grad():
            logits = model(**enc, position_ids=pos).logits[:, -1, first_tok]
        return [CHOICES[i] for i in logits.argmax(-1).tolist()]

    order = np.argsort([len(t) for t in texts], kind="stable")  # similar lengths per batch -> little padding
    out = [None] * len(texts)
    for i in range(0, len(order), batch):
        idx = order[i:i + batch]
        for j, lab in zip(idx, label([texts[j] for j in idx])):
            out[j] = lab
        print(f"{min(i + batch, len(order))}/{len(order)}", end="\r")
    print()
    return out
