# Auto-exported from 00_smoke_test_v2.ipynb
# Some notebooks are intended for interactive/Colab use.


# %% [markdown]
# # QDM Smoke Test v2: Less Aggressive Quantization
# 
# **Goal:** Reproduce the v1 smoke test, but with quantization mild enough that perplexity changes <5%. The interesting finding for the paper is "mechanistic damage appears even when task-level damage doesn't" — but to claim that, we need a quantization regime where perplexity barely moves.

# %% [markdown]
# ## 1. Install + imports

# %% Cell 2
# !pip install -q transformer_lens sae-lens datasets matplotlib
print("Done.")

# %% Cell 3
import torch
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformer_lens import HookedTransformer
from sae_lens import SAE
from tqdm.auto import tqdm

assert torch.cuda.is_available(), "No GPU. Set Runtime → Change runtime type → GPU."
print("Device:", torch.cuda.get_device_name(0))

device = "cuda"
torch.set_grad_enabled(False)

# %% [markdown]
# ## 2. Load model, SAE, and tokens (same as v1)

# %% Cell 5
MODEL_NAME = "pythia-70m-deduped"
LAYER = 4
HOOK_NAME = f"blocks.{LAYER}.hook_resid_post"
SMOKE_TOKENS = 50_000
SEQ_LEN = 512

model = HookedTransformer.from_pretrained(MODEL_NAME, device=device)
model.eval()

sae = SAE.from_pretrained(
    release="pythia-70m-deduped-res-sm",
    sae_id=HOOK_NAME,
    device=device,
)
sae.eval()

# Tokens — use the raw tokenizer (model.to_tokens truncates at n_ctx=2048)
ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
full_text = "\n\n".join(x for x in ds["text"] if x.strip())
print(f"Total characters: {len(full_text):,}")

tokens = model.tokenizer(full_text, return_tensors="pt", truncation=False)["input_ids"][0]
print(f"Total tokens available: {tokens.shape[0]:,}")

assert tokens.shape[0] >= SMOKE_TOKENS, f"Only {tokens.shape[0]} tokens, need {SMOKE_TOKENS}"

n_seqs = SMOKE_TOKENS // SEQ_LEN
tokens_smoke = tokens[:n_seqs * SEQ_LEN].reshape(n_seqs, SEQ_LEN).to(device)

# Save original weights once
original_state = {k: v.clone() for k, v in model.state_dict().items()}

print(f"Model: {MODEL_NAME}, layer {LAYER}")
print(f"SAE features: {sae.cfg.d_sae}")
print(f"Tokens: {tokens_smoke.shape}")

# %% [markdown]
# ## 3. Helper functions
# 
# Three quantization variants and the metric/caching machinery.

# %% Cell 7
WEIGHT_KEYWORDS = ["W_Q", "W_K", "W_V", "W_O", "W_in", "W_out"]
ATTN_OUTPUT_KEYWORDS = ["W_O"]

def is_quantizable(name, exclude_attn_output=False):
    if not any(s in name for s in WEIGHT_KEYWORDS):
        return False
    if exclude_attn_output and any(s in name for s in ATTN_OUTPUT_KEYWORDS):
        return False
    return True


def quantize_rtn(model, bits=8, per_channel=True, exclude_attn_output=False):
    """Apply RTN quantization in place. Returns count of quantized tensors."""
    q_max = 2**(bits - 1) - 1
    q_min = -(2**(bits - 1))
    count = 0
    for name, param in model.named_parameters():
        if not is_quantizable(name, exclude_attn_output):
            continue
        w = param.data
        if per_channel:
            # Per-output-channel scaling: scale along all dims except the last
            scale = w.abs().amax(dim=tuple(range(w.ndim - 1)), keepdim=True) / q_max
            scale = torch.clamp(scale, min=1e-12)
        else:
            scale_val = max(w.abs().max().item() / q_max, 1e-12)
            scale = torch.tensor(scale_val, device=w.device, dtype=w.dtype)
        q = torch.round(w / scale).clamp(q_min, q_max)
        param.data = (q * scale).to(w.dtype)
        count += 1
    return count


def restore(model, original_state):
    model.load_state_dict(original_state)


def cache_activations(model, tokens_2d, hook_name, batch_size=16):
    storage = []
    n_seqs = tokens_2d.shape[0]
    for i in tqdm(range(0, n_seqs, batch_size), desc="  caching", leave=False):
        batch = tokens_2d[i:i+batch_size]
        _, cache = model.run_with_cache(batch, names_filter=[hook_name])
        storage.append(cache[hook_name].cpu())
    acts = torch.cat(storage, dim=0)
    return acts.reshape(-1, acts.shape[-1])


def compute_perplexity(model, tokens_2d, batch_size=16):
    losses = []
    for i in range(0, tokens_2d.shape[0], batch_size):
        batch = tokens_2d[i:i+batch_size]
        loss = model(batch, return_type="loss")
        losses.append(loss.item())
    avg_loss = float(np.mean(losses))
    return float(np.exp(avg_loss)), avg_loss


def sae_encode_batched(sae, acts, device, batch=8192):
    out = []
    for i in tqdm(range(0, acts.shape[0], batch), desc="  encoding", leave=False):
        chunk = acts[i:i+batch].to(device).float()
        out.append(sae.encode(chunk).cpu())
    return torch.cat(out, dim=0)


def per_feature_pearson(a, b, eps=1e-8):
    a_c = a - a.mean(dim=0, keepdim=True)
    b_c = b - b.mean(dim=0, keepdim=True)
    num = (a_c * b_c).sum(dim=0)
    den = torch.sqrt((a_c**2).sum(dim=0) * (b_c**2).sum(dim=0)) + eps
    return num / den

print("Helpers defined.")

# %% [markdown]
# ## 4. Cache FP16 baseline once
# 
# We'll reuse this across all three quantization conditions.

# %% Cell 9
restore(model, original_state)
print("Computing FP16 perplexity...")
ppl_fp16, _ = compute_perplexity(model, tokens_smoke)
print(f"FP16 perplexity: {ppl_fp16:.3f}")

print("\nCaching FP16 activations...")
acts_fp16 = cache_activations(model, tokens_smoke, HOOK_NAME)
print(f"FP16 acts: {acts_fp16.shape}")

print("\nEncoding FP16 activations with SAE...")
features_fp16 = sae_encode_batched(sae, acts_fp16, device)
fp16_firing_rate = (features_fp16 > 0).float().mean(dim=0)
active_mask = fp16_firing_rate > 0.001
print(f"Active features (firing > 0.1%): {active_mask.sum().item()} / {features_fp16.shape[1]}")

# %% [markdown]
# ## 5. Run condition 1: per-channel RTN INT8
# 
# This is the v1 quantization with per-channel scaling instead of per-tensor. Should be massively less aggressive.

# %% Cell 11
restore(model, original_state)
n = quantize_rtn(model, bits=8, per_channel=True, exclude_attn_output=False)
ppl_c1, _ = compute_perplexity(model, tokens_smoke)
print(f"[per-channel INT8] Quantized {n} tensors")
print(f"[per-channel INT8] Perplexity: {ppl_c1:.3f}  (delta: {(ppl_c1/ppl_fp16-1)*100:+.2f}%)")

acts_c1 = cache_activations(model, tokens_smoke, HOOK_NAME)
features_c1 = sae_encode_batched(sae, acts_c1, device)
corrs_c1 = per_feature_pearson(features_fp16, features_c1)
active_corrs_c1 = corrs_c1[active_mask]
print(f"[per-channel INT8] Mean corr: {active_corrs_c1.mean():.3f}, median: {active_corrs_c1.median():.3f}")
print(f"[per-channel INT8] Survived >0.9: {(active_corrs_c1 > 0.9).float().mean():.1%}")
print(f"[per-channel INT8] Damaged <0.5: {(active_corrs_c1 < 0.5).float().mean():.1%}")

# %% [markdown]
# ## 6. Run condition 2: per-channel INT8, excluding attention output projections
# 
# If condition 1 perplexity is already <5%, skip this and try INT6 instead. If condition 1 is still too aggressive, this should help by leaving W_O at full precision (attention output projections are quantization-sensitive).

# %% Cell 13
restore(model, original_state)
n = quantize_rtn(model, bits=8, per_channel=True, exclude_attn_output=True)
ppl_c2, _ = compute_perplexity(model, tokens_smoke)
print(f"[per-channel INT8, no W_O] Quantized {n} tensors")
print(f"[per-channel INT8, no W_O] Perplexity: {ppl_c2:.3f}  (delta: {(ppl_c2/ppl_fp16-1)*100:+.2f}%)")

acts_c2 = cache_activations(model, tokens_smoke, HOOK_NAME)
features_c2 = sae_encode_batched(sae, acts_c2, device)
corrs_c2 = per_feature_pearson(features_fp16, features_c2)
active_corrs_c2 = corrs_c2[active_mask]
print(f"[per-channel INT8, no W_O] Mean corr: {active_corrs_c2.mean():.3f}, median: {active_corrs_c2.median():.3f}")
print(f"[per-channel INT8, no W_O] Survived >0.9: {(active_corrs_c2 > 0.9).float().mean():.1%}")
print(f"[per-channel INT8, no W_O] Damaged <0.5: {(active_corrs_c2 < 0.5).float().mean():.1%}")

# %% [markdown]
# ## 7. Run condition 3: per-channel RTN INT6
# 
# If INT8 with per-channel scaling preserves perplexity nicely, push to INT6 to find the regime where damage actually shows up. INT6 is 64 levels per channel — usually meaningful perturbation but not catastrophic.

# %% Cell 15
restore(model, original_state)
n = quantize_rtn(model, bits=6, per_channel=True, exclude_attn_output=False)
ppl_c3, _ = compute_perplexity(model, tokens_smoke)
print(f"[per-channel INT6] Quantized {n} tensors")
print(f"[per-channel INT6] Perplexity: {ppl_c3:.3f}  (delta: {(ppl_c3/ppl_fp16-1)*100:+.2f}%)")

acts_c3 = cache_activations(model, tokens_smoke, HOOK_NAME)
features_c3 = sae_encode_batched(sae, acts_c3, device)
corrs_c3 = per_feature_pearson(features_fp16, features_c3)
active_corrs_c3 = corrs_c3[active_mask]
print(f"[per-channel INT6] Mean corr: {active_corrs_c3.mean():.3f}, median: {active_corrs_c3.median():.3f}")
print(f"[per-channel INT6] Survived >0.9: {(active_corrs_c3 > 0.9).float().mean():.1%}")
print(f"[per-channel INT6] Damaged <0.5: {(active_corrs_c3 < 0.5).float().mean():.1%}")

# %% [markdown]
# ## 8. Summary table
# 
# Side-by-side comparison of the three conditions plus v1 (per-tensor INT8) reference numbers from your earlier run.

# %% Cell 17
import pandas as pd

results = pd.DataFrame([
    {
        "condition": "FP16 (baseline)",
        "perplexity": ppl_fp16,
        "ppl_delta_pct": 0.0,
        "mean_corr": 1.0,
        "median_corr": 1.0,
        "survived_>0.9_pct": 100.0,
        "damaged_<0.5_pct": 0.0,
    },
    {
        "condition": "per-channel INT8",
        "perplexity": ppl_c1,
        "ppl_delta_pct": (ppl_c1/ppl_fp16 - 1) * 100,
        "mean_corr": float(active_corrs_c1.mean()),
        "median_corr": float(active_corrs_c1.median()),
        "survived_>0.9_pct": float((active_corrs_c1 > 0.9).float().mean()) * 100,
        "damaged_<0.5_pct": float((active_corrs_c1 < 0.5).float().mean()) * 100,
    },
    {
        "condition": "per-channel INT8 (no W_O)",
        "perplexity": ppl_c2,
        "ppl_delta_pct": (ppl_c2/ppl_fp16 - 1) * 100,
        "mean_corr": float(active_corrs_c2.mean()),
        "median_corr": float(active_corrs_c2.median()),
        "survived_>0.9_pct": float((active_corrs_c2 > 0.9).float().mean()) * 100,
        "damaged_<0.5_pct": float((active_corrs_c2 < 0.5).float().mean()) * 100,
    },
    {
        "condition": "per-channel INT6",
        "perplexity": ppl_c3,
        "ppl_delta_pct": (ppl_c3/ppl_fp16 - 1) * 100,
        "mean_corr": float(active_corrs_c3.mean()),
        "median_corr": float(active_corrs_c3.median()),
        "survived_>0.9_pct": float((active_corrs_c3 > 0.9).float().mean()) * 100,
        "damaged_<0.5_pct": float((active_corrs_c3 < 0.5).float().mean()) * 100,
    },
])

pd.set_option("display.float_format", "{:.3f}".format)
pd.set_option("display.max_columns", None)
print(results.to_string(index=False))

# %% [markdown]
# ## 9. Side-by-side histograms
# 
# The headline plot. Four panels: FP16-vs-FP16 noise floor (sanity), then the three quantization conditions. Look for: conditions where the title shows small perplexity delta but the histogram still has a damage tail.

# %% Cell 19
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

conditions = [
    ("per-channel INT8", active_corrs_c1, ppl_c1),
    ("per-channel INT8 (no W_O)", active_corrs_c2, ppl_c2),
    ("per-channel INT6", active_corrs_c3, ppl_c3),
]

for ax, (label, corrs, ppl) in zip(axes, conditions):
    delta = (ppl/ppl_fp16 - 1) * 100
    ax.hist(corrs.numpy(), bins=60, edgecolor='black', alpha=0.8)
    ax.set_xlabel("Per-feature Pearson correlation (vs FP16)")
    ax.set_title(f"{label}\nppl delta: {delta:+.2f}%, damaged: {(corrs < 0.5).float().mean()*100:.1f}%")
    ax.axvline(0.9, color='green', linestyle='--', alpha=0.6, label='Survive (>0.9)')
    ax.axvline(0.5, color='red', linestyle='--', alpha=0.6, label='Damage (<0.5)')
    ax.set_xlim(0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')

axes[0].set_ylabel("Number of active features")
plt.tight_layout()
plt.savefig("smoke_v2_histograms.png", dpi=140, bbox_inches='tight')
plt.show()
print("Saved smoke_v2_histograms.png")

# %% [markdown]
# ## 10. Pick the regime to scale up
# 
# Look at the table from cell 8 and the plots from cell 9. The ideal condition for Phase 2 is one with:

# %% Cell 21
# Auto-pick the most informative regime
candidates = []
for label, corrs, ppl in conditions:
    delta = (ppl/ppl_fp16 - 1) * 100
    damaged = float((corrs < 0.5).float().mean()) * 100
    candidates.append({"label": label, "ppl_delta": delta, "damaged_pct": damaged, "corrs": corrs})

# "Best" = perplexity within 5% AND has at least 1% damaged features
# If multiple qualify, pick the one with the largest damage tail
qualified = [c for c in candidates if c["ppl_delta"] < 5.0 and c["damaged_pct"] >= 1.0]

if qualified:
    best = max(qualified, key=lambda c: c["damaged_pct"])
    print(f"BEST REGIME: {best['label']}")
    print(f"  ppl delta:    {best['ppl_delta']:+.2f}%  (target: < 5%)")
    print(f"  damaged tail: {best['damaged_pct']:.1f}%  (target: >= 1%)")
    print()
    print("This is the regime to use for Phase 2.")
    headline_corrs = best["corrs"]
    headline_label = best["label"]
elif any(c["ppl_delta"] < 5.0 for c in candidates):
    # Have a low-ppl-delta regime but no damage tail
    mild = [c for c in candidates if c["ppl_delta"] < 5.0]
    best = min(mild, key=lambda c: c["ppl_delta"])
    print(f"BEST WITHIN BUDGET: {best['label']}")
    print(f"  ppl delta:    {best['ppl_delta']:+.2f}%")
    print(f"  damaged tail: {best['damaged_pct']:.1f}%")
    print()
    print("No regime has both <5% ppl AND >=1% damage. Likely need INT5 or INT4 to see damage.")
    print("Recommendation: scale up with multiple bitwidths, including this one as 'mild' baseline.")
    headline_corrs = best["corrs"]
    headline_label = best["label"]
else:
    print("All conditions exceeded 5% perplexity. The quantization scheme is still too aggressive.")
    print("Consider: smaller exclusion set, group-wise scaling, or accepting a higher ppl threshold.")
    headline_corrs = candidates[0]["corrs"]
    headline_label = candidates[0]["label"]

# %% [markdown]
# ## 11. Inspect disrupted features in the chosen regime
# 
# Make sure the damaged features are coherent (look like real features), not noise. If they have semantically meaningful top-activating contexts, the result is real.

# %% Cell 23
# Use whichever condition got picked above
# Recompute corrs and features for inspection
if headline_label == "per-channel INT8":
    feat_used, corrs_used = features_c1, corrs_c1
elif headline_label == "per-channel INT8 (no W_O)":
    feat_used, corrs_used = features_c2, corrs_c2
else:
    feat_used, corrs_used = features_c3, corrs_c3

active_idx = torch.where(active_mask)[0]
sorted_active = active_idx[corrs_used[active_idx].argsort()]
flat_tokens = tokens_smoke.flatten().cpu()

print(f"=== 8 most disrupted features in: {headline_label} ===\n")
for rank, fi in enumerate(sorted_active[:8]):
    fi = int(fi)
    c = float(corrs_used[fi])
    fr = float(fp16_firing_rate[fi])
    print(f"#{rank+1}  feat={fi}  corr={c:+.3f}  firing_rate={fr:.5f}")
    feat_acts = features_fp16[:, fi]
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

# %% Cell 24
restore(model, original_state)
n = quantize_rtn(model, bits=7, per_channel=True, exclude_attn_output=False)
ppl_c4, _ = compute_perplexity(model, tokens_smoke)
print(f"[per-channel INT7] Quantized {n} tensors")
print(f"[per-channel INT7] Perplexity: {ppl_c4:.3f}  (delta: {(ppl_c4/ppl_fp16-1)*100:+.2f}%)")

acts_c4 = cache_activations(model, tokens_smoke, HOOK_NAME)
features_c4 = sae_encode_batched(sae, acts_c4, device)
corrs_c4 = per_feature_pearson(features_fp16, features_c4)
active_corrs_c4 = corrs_c4[active_mask]
print(f"[per-channel INT7] Mean corr: {active_corrs_c4.mean():.3f}, median: {active_corrs_c4.median():.3f}")
print(f"[per-channel INT7] Survived >0.9: {(active_corrs_c4 > 0.9).float().mean():.1%}")
print(f"[per-channel INT7] Damaged <0.5: {(active_corrs_c4 < 0.5).float().mean():.1%}")

# %% Cell 25
restore(model, original_state)
n = quantize_rtn(model, bits=5, per_channel=True, exclude_attn_output=False)
ppl_c5, _ = compute_perplexity(model, tokens_smoke)
print(f"[per-channel INT5] Quantized {n} tensors")
print(f"[per-channel INT5] Perplexity: {ppl_c5:.3f}  (delta: {(ppl_c5/ppl_fp16-1)*100:+.2f}%)")

acts_c5 = cache_activations(model, tokens_smoke, HOOK_NAME)
features_c5 = sae_encode_batched(sae, acts_c5, device)
corrs_c5 = per_feature_pearson(features_fp16, features_c5)
active_corrs_c5 = corrs_c5[active_mask]
print(f"[per-channel INT5] Mean corr: {active_corrs_c5.mean():.3f}, median: {active_corrs_c5.median():.3f}")
print(f"[per-channel INT5] Survived >0.9: {(active_corrs_c5 > 0.9).float().mean():.1%}")
print(f"[per-channel INT5] Damaged <0.5: {(active_corrs_c5 < 0.5).float().mean():.1%}")

# %% Cell 26
import pandas as pd

results = pd.DataFrame([
    {
        "condition": "FP16 (baseline)",
        "bits": 16,
        "perplexity": ppl_fp16,
        "ppl_delta_pct": 0.0,
        "mean_corr": 1.0, "median_corr": 1.0,
        "survived_>0.9_pct": 100.0, "damaged_<0.5_pct": 0.0,
    },
    {
        "condition": "per-channel INT8",
        "bits": 8,
        "perplexity": ppl_c1,
        "ppl_delta_pct": (ppl_c1/ppl_fp16 - 1) * 100,
        "mean_corr": float(active_corrs_c1.mean()),
        "median_corr": float(active_corrs_c1.median()),
        "survived_>0.9_pct": float((active_corrs_c1 > 0.9).float().mean()) * 100,
        "damaged_<0.5_pct": float((active_corrs_c1 < 0.5).float().mean()) * 100,
    },
    {
        "condition": "per-channel INT7",
        "bits": 7,
        "perplexity": ppl_c4,
        "ppl_delta_pct": (ppl_c4/ppl_fp16 - 1) * 100,
        "mean_corr": float(active_corrs_c4.mean()),
        "median_corr": float(active_corrs_c4.median()),
        "survived_>0.9_pct": float((active_corrs_c4 > 0.9).float().mean()) * 100,
        "damaged_<0.5_pct": float((active_corrs_c4 < 0.5).float().mean()) * 100,
    },
    {
        "condition": "per-channel INT6",
        "bits": 6,
        "perplexity": ppl_c3,
        "ppl_delta_pct": (ppl_c3/ppl_fp16 - 1) * 100,
        "mean_corr": float(active_corrs_c3.mean()),
        "median_corr": float(active_corrs_c3.median()),
        "survived_>0.9_pct": float((active_corrs_c3 > 0.9).float().mean()) * 100,
        "damaged_<0.5_pct": float((active_corrs_c3 < 0.5).float().mean()) * 100,
    },
    {
        "condition": "per-channel INT5",
        "bits": 5,
        "perplexity": ppl_c5,
        "ppl_delta_pct": (ppl_c5/ppl_fp16 - 1) * 100,
        "mean_corr": float(active_corrs_c5.mean()),
        "median_corr": float(active_corrs_c5.median()),
        "survived_>0.9_pct": float((active_corrs_c5 > 0.9).float().mean()) * 100,
        "damaged_<0.5_pct": float((active_corrs_c5 < 0.5).float().mean()) * 100,
    },
])

pd.set_option("display.float_format", "{:.3f}".format)
pd.set_option("display.max_columns", None)
print(results.to_string(index=False))
results.to_csv("smoke_v2_results.csv", index=False)

# %% Cell 27
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

quant_results = results[results["bits"] < 16].sort_values("bits", ascending=False)

axes[0].plot(quant_results["bits"], quant_results["ppl_delta_pct"],
             marker='o', markersize=10, linewidth=2, color='steelblue', label='Perplexity Δ (%)')
axes[0].set_xlabel("Bits")
axes[0].set_ylabel("Perplexity Δ (%)", color='steelblue')
axes[0].tick_params(axis='y', labelcolor='steelblue')
axes[0].invert_xaxis()
axes[0].grid(True, alpha=0.3)
axes[0].set_title("Task degradation vs bit-width")

ax2 = axes[0].twinx()
ax2.plot(quant_results["bits"], quant_results["damaged_<0.5_pct"],
         marker='s', markersize=10, linewidth=2, color='crimson', label='Damaged features (%)')
ax2.set_ylabel("Damaged features (%)", color='crimson')
ax2.tick_params(axis='y', labelcolor='crimson')

axes[1].plot(quant_results["bits"], quant_results["survived_>0.9_pct"],
             marker='o', markersize=10, linewidth=2, color='seagreen')
axes[1].set_xlabel("Bits")
axes[1].set_ylabel("Survived features (>0.9) (%)")
axes[1].invert_xaxis()
axes[1].grid(True, alpha=0.3)
axes[1].set_title("Feature survival vs bit-width")
axes[1].set_ylim(0, 105)

plt.tight_layout()
plt.savefig("smoke_v2_sweep.png", dpi=140, bbox_inches='tight')
plt.show()
print("Saved smoke_v2_sweep.png")

# %% Cell 28
fig, axes = plt.subplots(1, 4, figsize=(22, 5), sharey=True)

conditions = [
    ("per-channel INT8", active_corrs_c1, ppl_c1),
    ("per-channel INT7", active_corrs_c4, ppl_c4),
    ("per-channel INT6", active_corrs_c3, ppl_c3),
    ("per-channel INT5", active_corrs_c5, ppl_c5),
]

for ax, (label, corrs, ppl) in zip(axes, conditions):
    delta = (ppl/ppl_fp16 - 1) * 100
    damaged_pct = float((corrs < 0.5).float().mean()) * 100
    ax.hist(corrs.numpy(), bins=60, edgecolor='black', alpha=0.8)
    ax.set_xlabel("Pearson correlation (vs FP16)")
    ax.set_title(f"{label}\nppl Δ: {delta:+.2f}%, damaged: {damaged_pct:.1f}%")
    ax.axvline(0.9, color='green', linestyle='--', alpha=0.6, label='Survive (>0.9)')
    ax.axvline(0.5, color='red', linestyle='--', alpha=0.6, label='Damage (<0.5)')
    ax.set_xlim(0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=9)

axes[0].set_ylabel("Number of active features")
plt.tight_layout()
plt.savefig("smoke_v2_histograms_5panel.png", dpi=140, bbox_inches='tight')
plt.show()
print("Saved smoke_v2_histograms_5panel.png")

# %% [markdown]
# ## What to look for in the results
# 
# **Best case for your paper:** Per-channel INT8 has ppl delta < 2% **and** a damage tail of 2-5% of features. The disrupted features have coherent semantic identity. This means:
