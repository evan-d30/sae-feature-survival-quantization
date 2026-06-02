# Auto-exported from 02_pythia_phase2b_streaming_final.ipynb
# Some notebooks are intended for interactive/Colab use.


# %% [markdown]
# # QDM Phase 2B — Pythia-70M Streaming Notebook (Final)
# 
# Memory-safe Phase 2B notebook with:

# %% Cell 1
# Optional install cell. Run if the environment is missing packages.
# On Vast.ai, this may take a few minutes.

# !pip install -q pandas matplotlib numpy tqdm datasets transformer-lens sae-lens scikit-learn transformers accelerate bitsandbytes huggingface_hub safetensors

# %% Cell 2
import os
import gc
import glob
import json
import time
import math
import shutil
import getpass
from pathlib import Path

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from datasets import load_dataset

from transformer_lens import HookedTransformer
from sae_lens import SAE

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import login

pd.set_option("display.float_format", "{:.4f}".format)
torch.set_grad_enabled(False)

print("torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("VRAM GB:", torch.cuda.get_device_properties(0).total_memory / 1e9)

# %% Cell 3
# Optional Hugging Face login.
# Only run this if downloads are rate-limited or gated.

DO_HF_LOGIN = False

if DO_HF_LOGIN:
    hf_token = getpass.getpass("Paste your Hugging Face token: ")
    login(token=hf_token)
    os.environ["HF_TOKEN"] = hf_token
    os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token
    print("Hugging Face login complete.")
else:
    print("Skipping HF login. Set DO_HF_LOGIN=True if needed.")

# %% Cell 4
# =========================
# Configuration
# =========================

RUN_MODE = "test"   # "test" = 20k tokens; "full" = full run

TOKEN_BUDGETS = {
    "test": 20_000,
    "full": 200_000,   # increase to 500_000 if your instance is stable and runtime is acceptable
}

SEQ_LEN = 512
BATCH_SIZE_TL = 4
BATCH_SIZE_HF = 4

MODEL_TL_NAME = "pythia-70m-deduped"
MODEL_HF_NAME = "EleutherAI/pythia-70m-deduped"
SAE_RELEASE = "pythia-70m-deduped-res-sm"
LAYER = 4
HOOK_NAME = f"blocks.{LAYER}.hook_resid_post"
SAE_ID = HOOK_NAME

BITS_TO_TEST = [8, 7, 6, 5, 4]
FIRING_THRESHOLD = 0.001
SURVIVAL_THRESHOLD = 0.9
DAMAGE_THRESHOLD = 0.5

# Pruning will be matched to this RTN condition's perplexity delta.
PRUNE_MATCH_BITS = 6

TOKEN_BUDGET = TOKEN_BUDGETS[RUN_MODE]
OUTPUT_DIR = Path(f"phase2b_outputs_{RUN_MODE}_{TOKEN_BUDGET//1000}k_L{LAYER}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Put HF cache inside persistent workspace if available.
os.environ.setdefault("HF_HOME", str(Path.cwd() / "hf_cache"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(Path.cwd() / "hf_cache"))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("RUN_MODE:", RUN_MODE)
print("TOKEN_BUDGET:", TOKEN_BUDGET)
print("OUTPUT_DIR:", OUTPUT_DIR)
print("DEVICE:", DEVICE)

# %% Cell 5
# Safe cleanup. Use dry_run=True first.

def cleanup_phase2b_outputs(output_dir=OUTPUT_DIR, dry_run=True):
    output_dir = Path(output_dir)
    patterns = [
        "*.csv", "*.png", "*.json", "*.txt"
    ]

    files = []
    for p in patterns:
        files.extend(output_dir.glob(p))

    files = sorted(set(files))

    if not files:
        print("No old files found in", output_dir)
        return

    print("Matched files:")
    for f in files:
        print(" ", f)

    if dry_run:
        print("\ndry_run=True, nothing deleted.")
    else:
        for f in files:
            f.unlink()
            print("Deleted", f)

# Example:
# cleanup_phase2b_outputs(dry_run=True)
# cleanup_phase2b_outputs(dry_run=False)

# %% Cell 6
# =========================
# Load and tokenize dataset
# =========================

def build_tokens_2d(tokenizer, token_budget, seq_len):
    """
    Build a 2D token tensor for QDM.

    Uses WikiText-2 test split for small 20k smoke tests and train split for
    larger runs. We keep all non-empty lines so short lines are not discarded.
    """
    # Pick split based on token budget. Train has many more tokens than test.
    if token_budget <= 20_000:
        split = "test"
    else:
        split = "train"

    print(f"Using WikiText-2 split: {split}")

    ds = load_dataset(
        "Salesforce/wikitext",
        "wikitext-2-raw-v1",
        split=split,
    )

    # Keep all non-empty lines. Do not use len > 100; it discards too much text.
    full_text = "\n\n".join(x for x in ds["text"] if x.strip())
    print(f"Total characters: {len(full_text):,}")

    token_ids = tokenizer.encode(full_text, add_special_tokens=False)
    tokens = torch.tensor(token_ids, dtype=torch.long)
    print(f"Total available tokens: {tokens.shape[0]:,}")

    usable_tokens = min(token_budget, tokens.shape[0])
    n_seqs = usable_tokens // seq_len
    usable_tokens = n_seqs * seq_len

    if usable_tokens == 0:
        raise RuntimeError("Not enough tokens for one sequence.")

    tokens_2d = tokens[:usable_tokens].reshape(n_seqs, seq_len).to(DEVICE)

    print(f"Using tokens: {usable_tokens:,}")
    print(f"tokens_2d shape: {tuple(tokens_2d.shape)}")

    return tokens_2d

# %% Cell 7
# =========================
# Load TransformerLens models and SAE
# =========================

print("Loading TL reference model...")
model_ref = HookedTransformer.from_pretrained(MODEL_TL_NAME, device=DEVICE)
model_ref.eval()

print("Loading TL work model...")
model_work = HookedTransformer.from_pretrained(MODEL_TL_NAME, device=DEVICE)
model_work.eval()

print("Building token tensor...")
tokens_2d = build_tokens_2d(model_ref.tokenizer, TOKEN_BUDGET, SEQ_LEN)

print("Loading SAE...")
sae_obj = SAE.from_pretrained(release=SAE_RELEASE, sae_id=SAE_ID, device=DEVICE)
# Newer sae_lens returns SAE only; older versions may return a tuple.
if isinstance(sae_obj, tuple):
    sae = sae_obj[0]
else:
    sae = sae_obj
sae.eval()

print("Loaded SAE:", SAE_RELEASE, SAE_ID)
print("d_in:", sae.cfg.d_in, "d_sae:", sae.cfg.d_sae)

print("Saving original work-model state to CPU...")
original_state = {k: v.detach().cpu().clone() for k, v in model_work.state_dict().items()}
print("State saved.")

# %% Cell 8
# =========================
# Core helper functions
# =========================

def free_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def restore_model(model, original_state):
    model.load_state_dict({k: v.to(DEVICE) for k, v in original_state.items()})
    model.eval()
    free_memory()


def target_tl_weight(name):
    target_names = ["W_Q", "W_K", "W_V", "W_O", "W_in", "W_out"]
    return any(t in name for t in target_names)


def quantize_rtn_per_channel_tl(model, bits=8):
    """
    Simulated per-output-channel RTN quantization for TransformerLens weights.

    This matches the Phase 2A convention:
    - For weight shape (..., d_out), reduce over all dimensions except the last.
    - This gives one scale per output channel.
    """
    qmax = (2 ** (bits - 1)) - 1
    qmin = -(2 ** (bits - 1))

    n_tensors = 0
    n_params = 0

    for name, param in model.named_parameters():
        if target_tl_weight(name):
            w = param.data

            if w.ndim == 1:
                scale = w.abs().max() / qmax
            else:
                # Per-output-channel scaling: one scale for each final-dimension channel.
                scale = w.abs().amax(
                    dim=tuple(range(w.ndim - 1)),
                    keepdim=True,
                ) / qmax

            scale = torch.where(scale == 0, torch.ones_like(scale), scale)

            q = torch.round(w / scale).clamp(qmin, qmax)
            param.data = (q * scale).to(w.dtype)

            n_tensors += 1
            n_params += w.numel()

    return n_tensors, n_params


def apply_magnitude_pruning_tl(model, sparsity):
    """Per-tensor magnitude pruning on selected TL weight matrices."""
    n_tensors = 0
    n_pruned = 0
    n_total = 0
    for name, param in model.named_parameters():
        if target_tl_weight(name):
            w = param.data
            flat = w.abs().flatten()
            if flat.numel() == 0:
                continue
            # threshold for pruning smallest magnitudes
            k = int(math.floor(sparsity * flat.numel()))
            if k <= 0:
                continue
            if k >= flat.numel():
                thresh = flat.max() + 1
            else:
                thresh = torch.kthvalue(flat, k).values
            mask = (w.abs() > thresh).to(w.dtype)
            pruned_here = int((mask == 0).sum().item())
            param.data.mul_(mask)
            n_pruned += pruned_here
            n_total += w.numel()
            n_tensors += 1
    actual = n_pruned / max(n_total, 1)
    return n_tensors, n_pruned, n_total, actual


def compute_perplexity_tl(model, tokens_2d, batch_size=4, desc="Perplexity TL"):
    losses = []
    model.eval()
    for i in tqdm(range(0, tokens_2d.shape[0], batch_size), desc=desc):
        batch = tokens_2d[i:i+batch_size]
        with torch.no_grad():
            loss = model(batch, return_type="loss")
        losses.append(float(loss.item()))
    avg_loss = float(np.mean(losses))
    ppl = float(np.exp(avg_loss))
    return ppl, avg_loss


def tl_features(model, sae, batch, hook_name):
    with torch.no_grad():
        _, cache = model.run_with_cache(batch, names_filter=[hook_name])
        acts = cache[hook_name].detach().reshape(-1, cache[hook_name].shape[-1])
        feats = sae.encode(acts.to(DEVICE).float()).detach().cpu().to(torch.float64)
    del cache, acts
    return feats

# %% Cell 9
# =========================
# Streaming correlation engine
# =========================

class RunningFeatureStats:
    def __init__(self, d_sae):
        self.d_sae = d_sae
        self.n = 0
        self.sum_x = torch.zeros(d_sae, dtype=torch.float64)
        self.sum_y = torch.zeros(d_sae, dtype=torch.float64)
        self.sum_x2 = torch.zeros(d_sae, dtype=torch.float64)
        self.sum_y2 = torch.zeros(d_sae, dtype=torch.float64)
        self.sum_xy = torch.zeros(d_sae, dtype=torch.float64)
        self.fire_count = torch.zeros(d_sae, dtype=torch.float64)
        self.sum_x_activation = torch.zeros(d_sae, dtype=torch.float64)
        self.max_x_activation = torch.zeros(d_sae, dtype=torch.float64)

    def update(self, x, y):
        # x and y: (tokens_in_batch, d_sae), on CPU float64
        self.n += x.shape[0]
        self.sum_x += x.sum(dim=0)
        self.sum_y += y.sum(dim=0)
        self.sum_x2 += (x ** 2).sum(dim=0)
        self.sum_y2 += (y ** 2).sum(dim=0)
        self.sum_xy += (x * y).sum(dim=0)
        self.fire_count += (x > 0).sum(dim=0)
        self.sum_x_activation += x.sum(dim=0)
        self.max_x_activation = torch.maximum(self.max_x_activation, x.max(dim=0).values)

    def finalize(self, condition, bits, layer, firing_threshold=0.001):
        n = self.n
        numerator = self.sum_xy - (self.sum_x * self.sum_y / n)
        denom_x = self.sum_x2 - (self.sum_x ** 2 / n)
        denom_y = self.sum_y2 - (self.sum_y ** 2 / n)
        denominator = torch.sqrt(torch.clamp(denom_x * denom_y, min=1e-12))
        corr = torch.clamp(numerator / denominator, -1.0, 1.0)

        firing_rate = self.fire_count / n
        mean_activation = self.sum_x_activation / n
        active_mask = firing_rate > firing_threshold
        active_corrs = corr[active_mask]
        if active_corrs.numel() == 0:
            raise RuntimeError("No active features found; lower firing_threshold.")

        summary = {
            "condition": condition,
            "bits": bits,
            "layer": layer,
            "n_tokens": int(n),
            "n_active_features": int(active_mask.sum().item()),
            "mean_corr": float(active_corrs.mean().item()),
            "median_corr": float(active_corrs.median().item()),
            "survived_>0.9_pct": float((active_corrs > SURVIVAL_THRESHOLD).double().mean().item() * 100),
            "damaged_<0.5_pct": float((active_corrs < DAMAGE_THRESHOLD).double().mean().item() * 100),
        }
        per_feature = pd.DataFrame({
            "feature_id": np.arange(self.d_sae),
            "condition": condition,
            "bits": bits,
            "layer": layer,
            "corr": corr.numpy(),
            "firing_rate": firing_rate.numpy(),
            "mean_activation": mean_activation.numpy(),
            "max_activation": self.max_x_activation.numpy(),
            "active": active_mask.numpy(),
            "survived_>0.9": ((corr > SURVIVAL_THRESHOLD) & active_mask).numpy(),
            "damaged_<0.5": ((corr < DAMAGE_THRESHOLD) & active_mask).numpy(),
        })
        return summary, per_feature


def stream_compare_feature_fns(tokens_2d, ref_fn, cond_fn, d_sae, condition, bits, layer, batch_size, firing_threshold, desc):
    stats = RunningFeatureStats(d_sae)
    for i in tqdm(range(0, tokens_2d.shape[0], batch_size), desc=desc):
        batch = tokens_2d[i:i+batch_size]
        x = ref_fn(batch)
        y = cond_fn(batch)
        stats.update(x, y)
        del x, y, batch
        free_memory()
    return stats.finalize(condition, bits, layer, firing_threshold=firing_threshold)

# %% Cell 10
# =========================
# TL condition runner: FP16, RTN, pruning
# =========================

all_summaries = []
all_per_feature_paths = []


def save_condition_outputs(summary, per_feature, prefix):
    summary_path = OUTPUT_DIR / f"{prefix}_summary.csv"
    pf_path = OUTPUT_DIR / f"{prefix}_per_feature.csv"
    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    per_feature.to_csv(pf_path, index=False)
    print("Saved", summary_path)
    print("Saved", pf_path)
    return summary_path, pf_path


def append_master(summary, per_feature_path):
    all_summaries.append(summary)
    all_per_feature_paths.append(str(per_feature_path))
    master = pd.DataFrame(all_summaries)
    master.to_csv(OUTPUT_DIR / "phase2b_summary_partial.csv", index=False)
    with open(OUTPUT_DIR / "phase2b_per_feature_files.json", "w") as f:
        json.dump(all_per_feature_paths, f, indent=2)
    display(master)


def run_tl_condition(condition, bits=None, prune_sparsity=None, batch_size=BATCH_SIZE_TL):
    restore_model(model_work, original_state)

    if bits is not None:
        n_tensors, n_params = quantize_rtn_per_channel_tl(model_work, bits=bits)
        print(f"Applied RTN INT{bits}: {n_tensors} tensors, {n_params:,} params")
    if prune_sparsity is not None:
        n_tensors, n_pruned, n_total, actual = apply_magnitude_pruning_tl(model_work, prune_sparsity)
        print(f"Applied pruning target={prune_sparsity:.4f}, actual={actual:.4f}, tensors={n_tensors}")

    ppl, loss = compute_perplexity_tl(model_work, tokens_2d, batch_size=batch_size, desc=f"PPL {condition}")

    ref_fn = lambda batch: tl_features(model_ref, sae, batch, HOOK_NAME)
    cond_fn = lambda batch: tl_features(model_work, sae, batch, HOOK_NAME)

    summary, per_feature = stream_compare_feature_fns(
        tokens_2d=tokens_2d,
        ref_fn=ref_fn,
        cond_fn=cond_fn,
        d_sae=sae.cfg.d_sae,
        condition=condition,
        bits=16 if bits is None else bits,
        layer=LAYER,
        batch_size=batch_size,
        firing_threshold=FIRING_THRESHOLD,
        desc=f"Features {condition}",
    )

    summary["perplexity"] = ppl
    summary["loss"] = loss
    summary["method"] = "TL"
    if prune_sparsity is not None:
        summary["prune_sparsity"] = prune_sparsity
    else:
        summary["prune_sparsity"] = np.nan

    prefix = condition.replace(" ", "_").replace("/", "_").replace(".", "p")
    summary_path, pf_path = save_condition_outputs(summary, per_feature, prefix)
    append_master(summary, pf_path)

    restore_model(model_work, original_state)
    del per_feature
    free_memory()
    return summary

# %% Cell 11
# =========================
# Run core simulated RTN sweep: conditions 1-6
# =========================

summary_fp16 = run_tl_condition("FP16 baseline", bits=None)
PPL_FP16 = summary_fp16["perplexity"]

rtn_summaries = {}
for bits in BITS_TO_TEST:
    s = run_tl_condition(f"RTN INT{bits}", bits=bits)
    s["ppl_delta_pct"] = (s["perplexity"] / PPL_FP16 - 1) * 100
    rtn_summaries[bits] = s

# Add ppl_delta to all summaries and resave master.
for s in all_summaries:
    s["ppl_delta_pct"] = (s["perplexity"] / PPL_FP16 - 1) * 100
pd.DataFrame(all_summaries).to_csv(OUTPUT_DIR / "phase2b_summary_partial.csv", index=False)
display(pd.DataFrame(all_summaries))

# %% Cell 12
# =========================
# Pruning matched to RTN condition perplexity
# =========================

TARGET_PPL = rtn_summaries[PRUNE_MATCH_BITS]["perplexity"]
TARGET_DELTA = (TARGET_PPL / PPL_FP16 - 1) * 100
print(f"Matching pruning to RTN INT{PRUNE_MATCH_BITS}: PPL={TARGET_PPL:.3f}, delta={TARGET_DELTA:+.2f}%")


def eval_prune_ppl(sparsity):
    restore_model(model_work, original_state)
    apply_magnitude_pruning_tl(model_work, sparsity)
    ppl, loss = compute_perplexity_tl(model_work, tokens_2d, batch_size=BATCH_SIZE_TL, desc=f"PPL prune {sparsity:.3f}")
    restore_model(model_work, original_state)
    return ppl

# Binary search sparsity. Keep iterations low for runtime.
lo, hi = 0.0, 0.8
search_rows = []
for step in range(6):
    mid = (lo + hi) / 2
    ppl_mid = eval_prune_ppl(mid)
    search_rows.append({"step": step, "sparsity": mid, "ppl": ppl_mid, "delta_pct": (ppl_mid/PPL_FP16 - 1)*100})
    print(search_rows[-1])
    if ppl_mid < TARGET_PPL:
        lo = mid
    else:
        hi = mid

search_df = pd.DataFrame(search_rows)
search_df.to_csv(OUTPUT_DIR / "pruning_match_search.csv", index=False)
display(search_df)

best_idx = (search_df["ppl"] - TARGET_PPL).abs().idxmin()
BEST_PRUNE_SPARSITY = float(search_df.loc[best_idx, "sparsity"])
print("Best pruning sparsity:", BEST_PRUNE_SPARSITY)

summary_prune = run_tl_condition(f"Magnitude pruning matched INT{PRUNE_MATCH_BITS}", prune_sparsity=BEST_PRUNE_SPARSITY)
summary_prune["ppl_delta_pct"] = (summary_prune["perplexity"] / PPL_FP16 - 1) * 100
pd.DataFrame(all_summaries).to_csv(OUTPUT_DIR / "phase2b_summary_partial.csv", index=False)

# %% Cell 13
# =========================
# HF / bitsandbytes helper functions
# =========================

# Note: HF hidden_states[layer+1] should correspond roughly to post-block residual stream.
# We compare HF FP16 vs HF quantized, so the basis is internally consistent for BNB conditions.

hf_tokenizer = AutoTokenizer.from_pretrained(MODEL_HF_NAME)
if hf_tokenizer.pad_token is None:
    hf_tokenizer.pad_token = hf_tokenizer.eos_token


def load_hf_model_fp16():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_HF_NAME,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map={"": 0} if DEVICE == "cuda" else None,
    )
    model.eval()
    return model


def load_hf_model_bnb_int8():
    qconfig = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_HF_NAME,
        quantization_config=qconfig,
        device_map={"": 0} if DEVICE == "cuda" else None,
    )
    model.eval()
    return model


def load_hf_model_bnb_nf4():
    qconfig = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_HF_NAME,
        quantization_config=qconfig,
        device_map={"": 0} if DEVICE == "cuda" else None,
    )
    model.eval()
    return model


def hf_features(model, sae, batch, layer_idx):
    with torch.no_grad():
        out = model(batch, output_hidden_states=True, use_cache=False)
        acts = out.hidden_states[layer_idx + 1].detach().reshape(-1, out.hidden_states[layer_idx + 1].shape[-1])
        feats = sae.encode(acts.to(DEVICE).float()).detach().cpu().to(torch.float64)
    del out, acts
    return feats


def compute_perplexity_hf(model, tokens_2d, batch_size=4, desc="HF PPL"):
    losses = []
    for i in tqdm(range(0, tokens_2d.shape[0], batch_size), desc=desc):
        batch = tokens_2d[i:i+batch_size]
        with torch.no_grad():
            out = model(batch, labels=batch, use_cache=False)
        losses.append(float(out.loss.item()))
        del out
        free_memory()
    avg_loss = float(np.mean(losses))
    ppl = float(np.exp(avg_loss))
    return ppl, avg_loss


def run_hf_bnb_condition(condition, load_cond_model_fn, bits, batch_size=BATCH_SIZE_HF):
    print("Loading HF FP16 reference model...")
    hf_ref = load_hf_model_fp16()
    ppl_ref, loss_ref = compute_perplexity_hf(hf_ref, tokens_2d, batch_size=batch_size, desc="HF FP16 PPL")
    print("HF FP16 PPL:", ppl_ref)

    print(f"Loading condition model: {condition}")
    hf_cond = load_cond_model_fn()
    ppl_cond, loss_cond = compute_perplexity_hf(hf_cond, tokens_2d, batch_size=batch_size, desc=f"{condition} PPL")

    ref_fn = lambda batch: hf_features(hf_ref, sae, batch, LAYER)
    cond_fn = lambda batch: hf_features(hf_cond, sae, batch, LAYER)

    summary, per_feature = stream_compare_feature_fns(
        tokens_2d=tokens_2d,
        ref_fn=ref_fn,
        cond_fn=cond_fn,
        d_sae=sae.cfg.d_sae,
        condition=condition,
        bits=bits,
        layer=LAYER,
        batch_size=batch_size,
        firing_threshold=FIRING_THRESHOLD,
        desc=f"Features {condition}",
    )

    summary["perplexity"] = ppl_cond
    summary["loss"] = loss_cond
    summary["hf_ref_perplexity"] = ppl_ref
    summary["hf_ref_loss"] = loss_ref
    summary["ppl_delta_pct"] = (ppl_cond / ppl_ref - 1) * 100
    summary["method"] = "HF-bitsandbytes"
    summary["prune_sparsity"] = np.nan

    prefix = condition.replace(" ", "_").replace("/", "_").replace(".", "p")
    summary_path, pf_path = save_condition_outputs(summary, per_feature, prefix)
    append_master(summary, pf_path)

    del per_feature, hf_ref, hf_cond
    free_memory()
    return summary

# %% Cell 14
# =========================
# Optional HF calibration: HF FP16 vs TL FP16 on the same SAE
# =========================
# This checks whether HuggingFace hidden states are close enough to TransformerLens
# residual-stream activations for BNB results to be interpreted alongside TL/RTN results.
# It is not counted as one of the 9 main conditions.
#
# Default:
#   - test mode: skip calibration for speed
#   - full mode: run calibration for rigor

RUN_HF_TL_CALIBRATION = (RUN_MODE == "full")

if RUN_HF_TL_CALIBRATION:
    print("Running HF-vs-TL FP16 calibration...")
    hf_ref_cal = load_hf_model_fp16()

    ref_fn = lambda batch: tl_features(model_ref, sae, batch, HOOK_NAME)
    cond_fn = lambda batch: hf_features(hf_ref_cal, sae, batch, LAYER)

    cal_summary, cal_pf = stream_compare_feature_fns(
        tokens_2d=tokens_2d,
        ref_fn=ref_fn,
        cond_fn=cond_fn,
        d_sae=sae.cfg.d_sae,
        condition="HF FP16 vs TL FP16 calibration",
        bits=16,
        layer=LAYER,
        batch_size=BATCH_SIZE_HF,
        firing_threshold=FIRING_THRESHOLD,
        desc="HF-TL calibration",
    )

    cal_summary["method"] = "calibration"
    save_condition_outputs(cal_summary, cal_pf, "HF_TL_FP16_calibration")

    display(pd.DataFrame([cal_summary]))

    # For paper interpretation: if this is low, BNB absolute damage rates are not
    # directly comparable to TL-based RTN rates.
    print("Calibration mean corr:", cal_summary["mean_corr"])
    print("Calibration median corr:", cal_summary["median_corr"])
    print("Calibration survived >0.9 %:", cal_summary["survived_>0.9_pct"])

    del cal_pf, hf_ref_cal
    free_memory()
else:
    print("Skipping HF-vs-TL calibration. It runs automatically when RUN_MODE == 'full'.")
    print("Set RUN_HF_TL_CALIBRATION=True manually if you want to run it in test mode.")

# %% Cell 15
# =========================
# Run bitsandbytes conditions: conditions 7-8
# =========================
# These can fail if bitsandbytes/CUDA setup is incompatible. If so, keep RTN/pruning results and debug BNB separately.

RUN_BNB_CONDITIONS = True

if RUN_BNB_CONDITIONS:
    try:
        summary_bnb8 = run_hf_bnb_condition("bitsandbytes LLM.int8", load_hf_model_bnb_int8, bits=8)
    except Exception as e:
        print("BNB INT8 failed:", repr(e))
        with open(OUTPUT_DIR / "bnb_int8_error.txt", "w") as f:
            f.write(repr(e))

    try:
        summary_nf4 = run_hf_bnb_condition("bitsandbytes NF4 4bit", load_hf_model_bnb_nf4, bits=4)
    except Exception as e:
        print("BNB NF4 failed:", repr(e))
        with open(OUTPUT_DIR / "bnb_nf4_error.txt", "w") as f:
            f.write(repr(e))
else:
    print("Skipping BNB conditions. Set RUN_BNB_CONDITIONS=True to run them.")

pd.DataFrame(all_summaries).to_csv(OUTPUT_DIR / "phase2b_summary_partial.csv", index=False)
display(pd.DataFrame(all_summaries))

# %% Cell 16
# =========================
# Final summary and plots
# =========================

summary_df = pd.DataFrame(all_summaries)

# Fill ppl deltas where missing. Note: HF BNB uses HF reference PPL, TL uses TL FP16 PPL.
if "ppl_delta_pct" not in summary_df.columns:
    summary_df["ppl_delta_pct"] = np.nan

summary_df.to_csv(OUTPUT_DIR / "phase2b_summary_final.csv", index=False)
print("Saved", OUTPUT_DIR / "phase2b_summary_final.csv")
display(summary_df)

# Main simulated RTN sweep plot
rtn_df = summary_df[summary_df["condition"].str.startswith("RTN INT")].copy()
if len(rtn_df) > 0:
    rtn_df = rtn_df.sort_values("bits", ascending=False)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(rtn_df["bits"], rtn_df["ppl_delta_pct"], marker="o", linewidth=2)
    axes[0].invert_xaxis()
    axes[0].set_xlabel("Bits")
    axes[0].set_ylabel("Perplexity delta (%)")
    axes[0].set_title("Task degradation vs bitwidth")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(rtn_df["bits"], rtn_df["survived_>0.9_pct"], marker="o", linewidth=2, label="Survived >0.9")
    axes[1].plot(rtn_df["bits"], rtn_df["damaged_<0.5_pct"], marker="s", linewidth=2, label="Damaged <0.5")
    axes[1].invert_xaxis()
    axes[1].set_xlabel("Bits")
    axes[1].set_ylabel("Feature percentage")
    axes[1].set_title("Feature survival/damage vs bitwidth")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase2b_rtn_sweep.png", dpi=150, bbox_inches="tight")
    plt.show()

# Compare all conditions
plot_df = summary_df.copy()
if len(plot_df) > 0:
    fig, ax = plt.subplots(figsize=(12, 6))
    labels = plot_df["condition"].tolist()
    x = np.arange(len(labels))
    ax.bar(x, plot_df["survived_>0.9_pct"].values, label="Survived >0.9")
    ax.bar(x, plot_df["damaged_<0.5_pct"].values, label="Damaged <0.5")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Feature percentage")
    ax.set_title("Feature survival and damage across Phase 2B conditions")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase2b_all_conditions_bar.png", dpi=150, bbox_inches="tight")
    plt.show()

# %% Cell 17
# =========================
# Inspect most disrupted features from any saved per-feature CSV
# =========================

def inspect_top_damaged(per_feature_csv, top_k=20):
    pf = pd.read_csv(per_feature_csv)
    active = pf[pf["active"] == True].sort_values("corr")
    cols = ["feature_id", "condition", "bits", "layer", "corr", "firing_rate", "mean_activation", "max_activation", "survived_>0.9", "damaged_<0.5"]
    display(active[cols].head(top_k))
    return active[cols].head(top_k)

print("Available per-feature files:")
for p in sorted(OUTPUT_DIR.glob("*_per_feature.csv")):
    print(p)

# Example usage after run:
# inspect_top_damaged(OUTPUT_DIR / "RTN_INT6_per_feature.csv", top_k=20)

# %% Cell 18
# =========================
# Matched-perplexity comparison: quantization vs pruning
# This is the Borobia differentiation analysis
# =========================

rtn_match_pf_path = OUTPUT_DIR / f"RTN_INT{PRUNE_MATCH_BITS}_per_feature.csv"
prune_pf_path = OUTPUT_DIR / f"Magnitude_pruning_matched_INT{PRUNE_MATCH_BITS}_per_feature.csv"

if rtn_match_pf_path.exists() and prune_pf_path.exists():
    pf_rtn = pd.read_csv(rtn_match_pf_path)
    pf_prune = pd.read_csv(prune_pf_path)

    # Restrict to active features (active in FP16)
    active_rtn = pf_rtn[pf_rtn["active"] == True].set_index("feature_id")
    active_prune = pf_prune[pf_prune["active"] == True].set_index("feature_id")

    # Inner join on feature_id
    shared = active_rtn.join(active_prune, lsuffix="_rtn", rsuffix="_prune", how="inner")
    print(f"Shared active features: {len(shared)}")

    damaged_rtn = shared["damaged_<0.5_rtn"]
    damaged_prune = shared["damaged_<0.5_prune"]

    n_rtn = int(damaged_rtn.sum())
    n_prune = int(damaged_prune.sum())
    n_both = int((damaged_rtn & damaged_prune).sum())
    n_either = int((damaged_rtn | damaged_prune).sum())
    jaccard = n_both / max(n_either, 1)

    print(f"\n=== Damage overlap at matched perplexity ===")
    print(f"  Damaged by RTN only:    {n_rtn - n_both}")
    print(f"  Damaged by pruning only: {n_prune - n_both}")
    print(f"  Damaged by both:        {n_both}")
    print(f"  Jaccard overlap:        {jaccard:.3f}")

    # Damage-score correlation
    damage_rtn_scores = 1 - shared["corr_rtn"]
    damage_prune_scores = 1 - shared["corr_prune"]
    pearson_r = damage_rtn_scores.corr(damage_prune_scores, method="pearson")
    spearman_r = damage_rtn_scores.corr(damage_prune_scores, method="spearman")
    print(f"  Per-feature damage-score Pearson:  {pearson_r:.3f}")
    print(f"  Per-feature damage-score Spearman: {spearman_r:.3f}")

    # Save the comparison
    comparison = pd.DataFrame({
        "feature_id": shared.index,
        "corr_RTN": shared["corr_rtn"],
        "corr_PRUNE": shared["corr_prune"],
        "damaged_RTN": damaged_rtn,
        "damaged_PRUNE": damaged_prune,
        "fp16_firing_rate": shared["firing_rate_rtn"],
    })
    comparison.to_csv(OUTPUT_DIR / "matched_perplexity_overlap.csv", index=False)

    # 3-panel plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].scatter(shared["corr_rtn"], shared["corr_prune"], alpha=0.3, s=8)
    axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
    axes[0].axvline(0.5, color='red', alpha=0.3, linestyle=':')
    axes[0].axhline(0.5, color='red', alpha=0.3, linestyle=':')
    axes[0].set_xlabel(f"Correlation under RTN INT{PRUNE_MATCH_BITS}")
    axes[0].set_ylabel("Correlation under matched-sparsity pruning")
    axes[0].set_title(f"Per-feature damage scatter\n(Pearson r={pearson_r:.2f})")
    axes[0].set_xlim(0, 1); axes[0].set_ylim(0, 1)
    axes[0].grid(True, alpha=0.3)

    categories = ['RTN only', 'Both', 'Pruning only', 'Neither']
    counts = [n_rtn - n_both, n_both, n_prune - n_both, len(shared) - n_either]
    axes[1].bar(categories, counts, color=['steelblue', 'purple', 'crimson', 'lightgrey'], edgecolor='black')
    for i, c in enumerate(counts):
        axes[1].text(i, c, str(c), ha='center', va='bottom')
    axes[1].set_ylabel("Number of features")
    axes[1].set_title(f"Damage overlap\nJaccard = {jaccard:.3f}")
    axes[1].grid(True, alpha=0.3, axis='y')

    # Damage by firing-rate decile
    deciles = pd.qcut(shared["firing_rate_rtn"], q=10, labels=False, duplicates='drop')
    decile_df = pd.DataFrame({
        "decile": deciles,
        "damaged_rtn": damaged_rtn.values,
        "damaged_prune": damaged_prune.values,
    })
    agg = decile_df.groupby("decile").agg({"damaged_rtn": "mean", "damaged_prune": "mean"}).reset_index()
    x_d = np.arange(len(agg))
    w = 0.4
    axes[2].bar(x_d - w/2, agg["damaged_rtn"] * 100, w, label=f"RTN INT{PRUNE_MATCH_BITS}", color='steelblue')
    axes[2].bar(x_d + w/2, agg["damaged_prune"] * 100, w, label="Pruning", color='crimson')
    axes[2].set_xlabel("Firing-rate decile (low → high)")
    axes[2].set_ylabel("% damaged in decile")
    axes[2].set_title("Damage rate by feature rarity")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase2b_matched_perplexity_comparison.png", dpi=150, bbox_inches='tight')
    plt.show()
else:
    print("Skipping matched-perplexity comparison: one or both CSVs missing.")
    print(f"  RTN: {rtn_match_pf_path.exists()}")
    print(f"  Prune: {prune_pf_path.exists()}")

# %% [markdown]
# ## Notes
# 
# - Start with `RUN_MODE = "test"` for the 20k-token end-to-end check.
