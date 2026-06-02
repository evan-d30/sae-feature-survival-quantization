# Auto-exported from 01_pythia_phase2a_bitwidth_sweep.ipynb
# Some notebooks are intended for interactive/Colab use.


# %% [markdown]
# # QDM Phase 2A + 2A.5: Pythia-70M Full Bit-Width Sweep
# 
# **What changed from the smoke test:**

# %% [markdown]
# ## 1. Mount Drive for checkpointing

# %% Cell 2
import os
from pathlib import Path

# Save everything to the persistent workspace folder on vast.ai
RESULTS_DIR = Path("/workspace/qdm_phase2a_70m_results")

# Create folder if it does not exist
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print(f"Results directory: {RESULTS_DIR}")
print(f"Directory exists: {RESULTS_DIR.exists()}")
print(f"Existing files: {os.listdir(RESULTS_DIR) if RESULTS_DIR.exists() else 'none'}")

# %% [markdown]
# ## 2. Install + imports

# %% Cell 4
# !pip install -q transformer_lens sae-lens datasets matplotlib pandas
print("Installed.")

# %% Cell 5
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gc
from pathlib import Path
from datasets import load_dataset
from transformer_lens import HookedTransformer
from sae_lens import SAE
from tqdm.auto import tqdm

assert torch.cuda.is_available(), "No GPU"
print("Device:", torch.cuda.get_device_name(0))
device = "cuda"
torch.set_grad_enabled(False)

# %% [markdown]
# ## 3. Config

# %% Cell 7
MODEL_NAME = "pythia-70m-deduped"
SAE_RELEASE = "pythia-70m-deduped-res-sm"

# Pythia-70M has 6 layers (0-5).
# Layer 4 was the smoke test (worked). Layer 2 is the secondary robustness check.
PRIMARY_LAYER = 4
SECONDARY_LAYER = 2

TOKEN_BUDGET = 200_000
SEQ_LEN = 512
BATCH_SIZE = 16

BITS_TO_TEST = [8, 7, 6, 5]  # plus FP16 baseline

CKPT = lambda name: os.path.join(RESULTS_DIR, name)

print(f"Model: {MODEL_NAME}")
print(f"Layers: primary={PRIMARY_LAYER}, secondary={SECONDARY_LAYER}")
print(f"Tokens: {TOKEN_BUDGET:,}")
print(f"Bit-widths: FP16 + {BITS_TO_TEST}")

# %% [markdown]
# ## 4. Load model + tokens

# %% Cell 9
print("Loading Pythia-70m-deduped...")
model = HookedTransformer.from_pretrained(MODEL_NAME, device=device)
model.eval()
print(f"  n_layers={model.cfg.n_layers}, d_model={model.cfg.d_model}")

original_state = {k: v.clone() for k, v in model.state_dict().items()}

# Tokens via raw tokenizer (model.to_tokens truncates at n_ctx)
print("\nTokenizing WikiText-2 train split...")
ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
full_text = "\n\n".join(x for x in ds["text"] if x.strip())
tokens = model.tokenizer(full_text, return_tensors="pt", truncation=False)["input_ids"][0]
print(f"Total tokens available: {tokens.shape[0]:,}")

assert tokens.shape[0] >= TOKEN_BUDGET, f"Need {TOKEN_BUDGET}, have {tokens.shape[0]}"

n_seqs = TOKEN_BUDGET // SEQ_LEN
tokens_2d = tokens[:n_seqs * SEQ_LEN].reshape(n_seqs, SEQ_LEN).to(device)
print(f"Token tensor: {tokens_2d.shape}")

# %% [markdown]
# ## 5. Helper functions
# 
# Quantization, caching, metrics. Same as before.

# %% Cell 11
WEIGHT_KEYWORDS = ["W_Q", "W_K", "W_V", "W_O", "W_in", "W_out"]

def is_quantizable(name):
    return any(s in name for s in WEIGHT_KEYWORDS)

def quantize_rtn_per_channel(model, bits):
    q_max = 2**(bits - 1) - 1
    q_min = -(2**(bits - 1))
    count = 0
    for name, param in model.named_parameters():
        if not is_quantizable(name):
            continue
        w = param.data
        scale = w.abs().amax(dim=tuple(range(w.ndim - 1)), keepdim=True) / q_max
        scale = torch.clamp(scale, min=1e-12)
        q = torch.round(w / scale).clamp(q_min, q_max)
        param.data = (q * scale).to(w.dtype)
        count += 1
    return count

def restore(model, original_state):
    model.load_state_dict(original_state)

def cache_activations(model, tokens_2d, hook_name, batch_size):
    storage = []
    n_seqs = tokens_2d.shape[0]
    for i in tqdm(range(0, n_seqs, batch_size), desc=f"caching {hook_name}", leave=False):
        batch = tokens_2d[i:i+batch_size]
        _, cache = model.run_with_cache(batch, names_filter=[hook_name])
        storage.append(cache[hook_name].cpu())
    acts = torch.cat(storage, dim=0)
    return acts.reshape(-1, acts.shape[-1])

def compute_perplexity(model, tokens_2d, batch_size):
    losses = []
    for i in range(0, tokens_2d.shape[0], batch_size):
        loss = model(tokens_2d[i:i+batch_size], return_type="loss")
        losses.append(loss.item())
    return float(np.exp(float(np.mean(losses))))

def sae_encode_batched(sae, acts, device, batch=8192):
    out = []
    for i in tqdm(range(0, acts.shape[0], batch), desc="SAE encoding", leave=False):
        chunk = acts[i:i+batch].to(device).float()
        out.append(sae.encode(chunk).cpu())
    return torch.cat(out, dim=0)

def per_feature_pearson(a, b, eps=1e-8):
    a_c = a - a.mean(dim=0, keepdim=True)
    b_c = b - b.mean(dim=0, keepdim=True)
    num = (a_c * b_c).sum(dim=0)
    den = torch.sqrt((a_c**2).sum(dim=0) * (b_c**2).sum(dim=0)) + eps
    return num / den

def summarize_correlations(corrs, firing_rates, threshold=0.001):
    active_mask = firing_rates > threshold
    active = corrs[active_mask]
    return {
        "n_total_features": int(corrs.numel()),
        "n_active_features": int(active_mask.sum().item()),
        "mean_corr": float(active.mean()),
        "median_corr": float(active.median()),
        "survived_pct": float((active > 0.9).float().mean()) * 100,
        "degraded_pct": float(((active > 0.5) & (active <= 0.9)).float().mean()) * 100,
        "damaged_pct": float((active < 0.5).float().mean()) * 100,
    }

print("Helpers ready.")

# %% [markdown]
# ## 6. Sweep runner with checkpointing

# %% Cell 13
def run_sweep_for_layer(model, sae, hook_name, layer_idx, tokens_2d,
                       original_state, bits_list, ckpt_prefix):
    summary_path = CKPT(f"{ckpt_prefix}_summary.csv")

    if os.path.exists(summary_path):
        existing_df = pd.read_csv(summary_path)
        completed = set(existing_df["condition"].tolist())
        print(f"Resuming. Already done: {completed}")
        summary_rows = existing_df.to_dict("records")
    else:
        completed = set()
        summary_rows = []

    # FP16 baseline
    if "FP16" not in completed:
        print(f"\n[{ckpt_prefix}] === FP16 baseline ===")
        restore(model, original_state)
        ppl_fp16 = compute_perplexity(model, tokens_2d, BATCH_SIZE)
        print(f"  perplexity: {ppl_fp16:.3f}")

        acts_fp16 = cache_activations(model, tokens_2d, hook_name, BATCH_SIZE)
        features_fp16 = sae_encode_batched(sae, acts_fp16, device)
        fp16_firing_rate = (features_fp16 > 0).float().mean(dim=0)
        n_active = int((fp16_firing_rate > 0.001).sum().item())
        print(f"  active features: {n_active} / {features_fp16.shape[1]}")

        torch.save(features_fp16, CKPT(f"{ckpt_prefix}_features_FP16.pt"))
        torch.save(fp16_firing_rate, CKPT(f"{ckpt_prefix}_firing_rate_FP16.pt"))
        np.save(CKPT(f"{ckpt_prefix}_ppl_FP16.npy"), np.array([ppl_fp16]))

        summary_rows.append({
            "condition": "FP16", "bits": 16, "perplexity": ppl_fp16, "ppl_delta_pct": 0.0,
            "n_total_features": features_fp16.shape[1], "n_active_features": n_active,
            "mean_corr": 1.0, "median_corr": 1.0,
            "survived_pct": 100.0, "degraded_pct": 0.0, "damaged_pct": 0.0,
        })
        pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
        del acts_fp16; gc.collect(); torch.cuda.empty_cache()
    else:
        ppl_fp16 = float(np.load(CKPT(f"{ckpt_prefix}_ppl_FP16.npy"))[0])
        features_fp16 = torch.load(CKPT(f"{ckpt_prefix}_features_FP16.pt"))
        fp16_firing_rate = torch.load(CKPT(f"{ckpt_prefix}_firing_rate_FP16.pt"))
        print(f"[{ckpt_prefix}] FP16 already cached. ppl={ppl_fp16:.3f}")

    # Quantized conditions
    for bits in bits_list:
        cond = f"RTN_INT{bits}"
        if cond in completed:
            print(f"[{ckpt_prefix}] {cond} already done.")
            continue

        print(f"\n[{ckpt_prefix}] === {cond} ===")
        restore(model, original_state)
        n_q = quantize_rtn_per_channel(model, bits=bits)
        ppl = compute_perplexity(model, tokens_2d, BATCH_SIZE)
        ppl_delta = (ppl / ppl_fp16 - 1) * 100
        print(f"  quantized {n_q} tensors; ppl {ppl:.3f} (Δ {ppl_delta:+.2f}%)")

        acts = cache_activations(model, tokens_2d, hook_name, BATCH_SIZE)
        features = sae_encode_batched(sae, acts, device)
        corrs = per_feature_pearson(features_fp16, features)
        summary = summarize_correlations(corrs, fp16_firing_rate)

        row = {"condition": cond, "bits": bits, "perplexity": ppl,
               "ppl_delta_pct": ppl_delta, **summary}
        summary_rows.append(row)
        print(f"  survived {summary['survived_pct']:.1f}% | "
              f"degraded {summary['degraded_pct']:.1f}% | "
              f"damaged {summary['damaged_pct']:.1f}%")

        torch.save(corrs, CKPT(f"{ckpt_prefix}_corrs_{cond}.pt"))
        pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
        del acts, features, corrs; gc.collect(); torch.cuda.empty_cache()

    restore(model, original_state)
    return pd.DataFrame(summary_rows), features_fp16, fp16_firing_rate

print("Sweep runner ready.")

# %% [markdown]
# ## 7. Phase 2A: primary layer (4)

# %% Cell 15
hook_primary = f"blocks.{PRIMARY_LAYER}.hook_resid_post"
sae_primary = SAE.from_pretrained(
    release=SAE_RELEASE, sae_id=hook_primary, device=device
)
sae_primary.eval()
print(f"SAE loaded for {hook_primary}: d_in={sae_primary.cfg.d_in}, d_sae={sae_primary.cfg.d_sae}")

# %% Cell 16
results_2a, features_fp16_2a, firing_2a = run_sweep_for_layer(
    model=model, sae=sae_primary, hook_name=hook_primary, layer_idx=PRIMARY_LAYER,
    tokens_2d=tokens_2d, original_state=original_state,
    bits_list=BITS_TO_TEST, ckpt_prefix=f"phase2a_L{PRIMARY_LAYER}"
)
pd.set_option("display.float_format", "{:.3f}".format)
print("\n=== Phase 2A summary (layer", PRIMARY_LAYER, ") ===")
print(results_2a.to_string(index=False))

# %% [markdown]
# ## 8. Phase 2A plots: sweep curve + histograms

# %% Cell 18
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
quant = results_2a[results_2a["bits"] < 16].sort_values("bits", ascending=False)

axes[0].plot(quant["bits"], quant["ppl_delta_pct"],
             marker='o', markersize=10, linewidth=2, color='steelblue',
             label='Perplexity Δ (%)')
axes[0].set_xlabel("Bits"); axes[0].set_ylabel("Perplexity Δ (%)", color='steelblue')
axes[0].tick_params(axis='y', labelcolor='steelblue')
axes[0].invert_xaxis(); axes[0].grid(True, alpha=0.3)
axes[0].set_title(f"Pythia-70M layer {PRIMARY_LAYER}: bit-width sweep")
ax2 = axes[0].twinx()
ax2.plot(quant["bits"], quant["damaged_pct"],
         marker='s', markersize=10, linewidth=2, color='crimson',
         label='Damaged (%)')
ax2.set_ylabel("Damaged features (%)", color='crimson')
ax2.tick_params(axis='y', labelcolor='crimson')

axes[1].plot(quant["bits"], quant["survived_pct"],
             marker='o', markersize=10, linewidth=2, color='seagreen')
axes[1].set_xlabel("Bits"); axes[1].set_ylabel("Survived (>0.9) (%)")
axes[1].invert_xaxis(); axes[1].grid(True, alpha=0.3)
axes[1].set_title("Feature survival vs bit-width")
axes[1].set_ylim(0, 105)

plt.tight_layout()
plt.savefig(CKPT(f"phase2a_L{PRIMARY_LAYER}_sweep.png"), dpi=140, bbox_inches='tight')
plt.show()

# %% Cell 19
fig, axes = plt.subplots(1, len(BITS_TO_TEST), figsize=(5*len(BITS_TO_TEST), 5), sharey=True)
if len(BITS_TO_TEST) == 1:
    axes = [axes]

active_mask = firing_2a > 0.001

for ax, bits in zip(axes, BITS_TO_TEST):
    cond = f"RTN_INT{bits}"
    corrs_path = CKPT(f"phase2a_L{PRIMARY_LAYER}_corrs_{cond}.pt")
    corrs = torch.load(corrs_path)
    active_corrs = corrs[active_mask]
    row = results_2a[results_2a["condition"] == cond].iloc[0]
    ax.hist(active_corrs.numpy(), bins=60, edgecolor='black', alpha=0.8)
    ax.set_xlabel("Pearson correlation (vs FP16)")
    ax.set_title(f"{cond}\nppl Δ: {row['ppl_delta_pct']:+.2f}%, "
                 f"damaged: {row['damaged_pct']:.1f}%")
    ax.axvline(0.9, color='green', linestyle='--', alpha=0.6)
    ax.axvline(0.5, color='red', linestyle='--', alpha=0.6)
    ax.set_xlim(0, 1.0)
    ax.grid(True, alpha=0.3)

axes[0].set_ylabel("Number of active features")
plt.tight_layout()
plt.savefig(CKPT(f"phase2a_L{PRIMARY_LAYER}_histograms.png"), dpi=140, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 9. Inspect disrupted features (Phase 2A primary layer)

# %% Cell 21
# Pick the condition with the most interesting damage band (between 1% and 15%)
damage_band = results_2a[(results_2a["damaged_pct"] >= 0.5) & (results_2a["damaged_pct"] <= 15.0)]
if len(damage_band) == 0:
    damage_band = results_2a[(results_2a["bits"] < 16) & (results_2a["damaged_pct"] > 0)]

if len(damage_band) > 0:
    target_cond = damage_band.iloc[0]["condition"]
    corrs = torch.load(CKPT(f"phase2a_L{PRIMARY_LAYER}_corrs_{target_cond}.pt"))

    active_idx = torch.where(firing_2a > 0.001)[0]
    sorted_active = active_idx[corrs[active_idx].argsort()]
    flat_tokens = tokens_2d.flatten().cpu()

    print(f"=== 10 most disrupted features at {target_cond} (layer {PRIMARY_LAYER}) ===\n")
    for rank, fi in enumerate(sorted_active[:10]):
        fi = int(fi)
        print(f"#{rank+1}  feat={fi}  corr={float(corrs[fi]):+.3f}  firing={float(firing_2a[fi]):.5f}")
        feat_acts = features_fp16_2a[:, fi]
        top_positions = feat_acts.argsort(descending=True)[:3]
        for pos in top_positions:
            pos = int(pos)
            start = max(0, pos - 10)
            end = min(len(flat_tokens), pos + 3)
            toks = model.to_str_tokens(flat_tokens[start:end])
            mp = pos - start
            marked = "".join(f"[{t}]" if i == mp else t for i, t in enumerate(toks))
            print(f"   act={float(feat_acts[pos]):.2f}: {marked}")
        print()
else:
    print("No condition shows damaged features.")

# %% [markdown]
# ## 10. Phase 2A.5: secondary layer (2)
# 
# Free primary SAE first, then load the secondary.

# %% Cell 23
del sae_primary, features_fp16_2a, firing_2a
gc.collect()
torch.cuda.empty_cache()
print("Cleared primary SAE.")

# %% Cell 24
hook_secondary = f"blocks.{SECONDARY_LAYER}.hook_resid_post"
sae_secondary = SAE.from_pretrained(
    release=SAE_RELEASE, sae_id=hook_secondary, device=device
)
sae_secondary.eval()
print(f"SAE loaded for {hook_secondary}: d_in={sae_secondary.cfg.d_in}, d_sae={sae_secondary.cfg.d_sae}")

# %% Cell 25
results_2a5, features_fp16_2a5, firing_2a5 = run_sweep_for_layer(
    model=model, sae=sae_secondary, hook_name=hook_secondary, layer_idx=SECONDARY_LAYER,
    tokens_2d=tokens_2d, original_state=original_state,
    bits_list=BITS_TO_TEST, ckpt_prefix=f"phase2a5_L{SECONDARY_LAYER}"
)
print("\n=== Phase 2A.5 summary (layer", SECONDARY_LAYER, ") ===")
print(results_2a5.to_string(index=False))

# %% [markdown]
# ## 11. Overlay: layer comparison
# 
# The robustness check. If both layers show the same bit-width pattern, the result is layer-stable. If they diverge, layer-specificity is a finding in itself.

# %% Cell 27
results_2a_reload = pd.read_csv(CKPT(f"phase2a_L{PRIMARY_LAYER}_summary.csv"))

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
q2a = results_2a_reload[results_2a_reload["bits"] < 16].sort_values("bits", ascending=False)
q2a5 = results_2a5[results_2a5["bits"] < 16].sort_values("bits", ascending=False)

axes[0].plot(q2a["bits"], q2a["ppl_delta_pct"],
             marker='o', markersize=10, linewidth=2, label=f"L{PRIMARY_LAYER}")
axes[0].plot(q2a5["bits"], q2a5["ppl_delta_pct"],
             marker='s', markersize=10, linewidth=2, linestyle='--', label=f"L{SECONDARY_LAYER}")
axes[0].set_xlabel("Bits"); axes[0].set_ylabel("Perplexity Δ (%)")
axes[0].set_title("Task degradation\n(should match — whole-model metric)")
axes[0].invert_xaxis(); axes[0].grid(True, alpha=0.3); axes[0].legend()

axes[1].plot(q2a["bits"], q2a["survived_pct"],
             marker='o', markersize=10, linewidth=2, label=f"L{PRIMARY_LAYER}")
axes[1].plot(q2a5["bits"], q2a5["survived_pct"],
             marker='s', markersize=10, linewidth=2, linestyle='--', label=f"L{SECONDARY_LAYER}")
axes[1].set_xlabel("Bits"); axes[1].set_ylabel("Survived (>0.9) (%)")
axes[1].set_title("Feature survival by layer")
axes[1].invert_xaxis(); axes[1].grid(True, alpha=0.3); axes[1].legend()
axes[1].set_ylim(0, 105)

axes[2].plot(q2a["bits"], q2a["damaged_pct"],
             marker='o', markersize=10, linewidth=2, label=f"L{PRIMARY_LAYER}")
axes[2].plot(q2a5["bits"], q2a5["damaged_pct"],
             marker='s', markersize=10, linewidth=2, linestyle='--', label=f"L{SECONDARY_LAYER}")
axes[2].set_xlabel("Bits"); axes[2].set_ylabel("Damaged (<0.5) (%)")
axes[2].set_title("Feature damage by layer")
axes[2].invert_xaxis(); axes[2].grid(True, alpha=0.3); axes[2].legend()

plt.tight_layout()
plt.savefig(CKPT("phase2a_layer_comparison.png"), dpi=140, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 12. Combined results table

# %% Cell 29
results_2a_reload["layer"] = PRIMARY_LAYER
results_2a5["layer"] = SECONDARY_LAYER
combined = pd.concat([results_2a_reload, results_2a5], ignore_index=True)
combined = combined[["layer", "condition", "bits", "perplexity", "ppl_delta_pct",
                     "n_active_features", "mean_corr", "median_corr",
                     "survived_pct", "degraded_pct", "damaged_pct"]]
combined.to_csv(CKPT("phase2a_combined.csv"), index=False)

pd.set_option("display.max_columns", None)
print(combined.to_string(index=False))
print(f"\nSaved: {CKPT('phase2a_combined.csv')}")

# %% [markdown]
# ## 13. Next steps
# 
# After this finishes, send me:
