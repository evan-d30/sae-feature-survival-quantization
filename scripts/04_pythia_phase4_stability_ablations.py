# Auto-exported from 04_pythia_phase4_stability_ablations.ipynb
# Some notebooks are intended for interactive/Colab use.


# %% [markdown]
# # QDM Phase 4 (v2): Stability Ablations — Corrected
# 
# **What this notebook does:**

# %% [markdown]
# ## 1. Install + imports

# %% Cell 2
# !pip install -q transformer-lens sae-lens datasets matplotlib pandas tqdm
print("Installed.")

# %% Cell 3
import os
import gc
import math
import json
from pathlib import Path

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from datasets import load_dataset

from transformer_lens import HookedTransformer
from sae_lens import SAE

assert torch.cuda.is_available(), "No GPU"
DEVICE = "cuda"
torch.set_grad_enabled(False)
pd.set_option("display.float_format", "{:.4f}".format)
print("Device:", torch.cuda.get_device_name(0))

# %% [markdown]
# ## 2. Config

# %% Cell 5
MODEL_TL_NAME = "pythia-70m-deduped"
SAE_RELEASE = "pythia-70m-deduped-res-sm"
LAYER = 4
HOOK_NAME = f"blocks.{LAYER}.hook_resid_post"
SAE_ID = HOOK_NAME

SEQ_LEN = 512
BATCH_SIZE = 16
TOKEN_BUDGETS = [50_000, 100_000, 200_000]
SEEDS = [0, 1, 2]
DAMAGE_BITS = 6  # the bit-width to use as the "informative damage" condition

FIRING_THRESHOLD = 0.001
SURVIVAL_THRESHOLD = 0.9
DAMAGE_THRESHOLD = 0.5

OUTPUT_DIR = Path(f"phase4_outputs_L{LAYER}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Output: {OUTPUT_DIR}")

# %% [markdown]
# ## 3. Load models, SAE, tokens

# %% Cell 7
print("Loading TL reference model...")
model_ref = HookedTransformer.from_pretrained(MODEL_TL_NAME, device=DEVICE)
model_ref.eval()

print("Loading TL work model...")
model_work = HookedTransformer.from_pretrained(MODEL_TL_NAME, device=DEVICE)
model_work.eval()

original_state = {k: v.detach().cpu().clone() for k, v in model_work.state_dict().items()}

print("Loading SAE...")
sae_obj = SAE.from_pretrained(release=SAE_RELEASE, sae_id=SAE_ID, device=DEVICE)
sae = sae_obj[0] if isinstance(sae_obj, tuple) else sae_obj
sae.eval()
print(f"SAE d_in={sae.cfg.d_in}, d_sae={sae.cfg.d_sae}")

print("Loading tokens from WikiText-2 train...")
ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
full_text = "\n\n".join(x for x in ds["text"] if x.strip())
token_ids = model_ref.tokenizer.encode(full_text, add_special_tokens=False)
all_tokens = torch.tensor(token_ids, dtype=torch.long)
print(f"Total tokens available: {all_tokens.shape[0]:,}")

# We need enough tokens that random subset sampling can draw NON-OVERLAPPING samples
# at the largest budget. Check this.
min_needed = max(TOKEN_BUDGETS) * 2  # heuristic: 2x the biggest budget for diverse sampling
if all_tokens.shape[0] < min_needed:
    print(f"⚠ Warning: only {all_tokens.shape[0]:,} tokens total; "
          f"random subset sampling may produce overlapping subsets at large budgets")
print("Setup complete.")

# %% [markdown]
# ## 4. Helpers (streaming Pearson, RTN, token subset builders)

# %% Cell 9
def target_tl_weight(name):
    return any(t in name for t in ["W_Q", "W_K", "W_V", "W_O", "W_in", "W_out"])

def restore_model(model, original_state):
    model.load_state_dict({k: v.to(DEVICE) for k, v in original_state.items()})
    model.eval()
    gc.collect(); torch.cuda.empty_cache()

def quantize_rtn_per_channel_tl(model, bits):
    qmax = (2 ** (bits - 1)) - 1
    qmin = -(2 ** (bits - 1))
    for name, param in model.named_parameters():
        if target_tl_weight(name):
            w = param.data
            if w.ndim == 1:
                scale = w.abs().max() / qmax
            else:
                scale = w.abs().amax(dim=tuple(range(w.ndim - 1)), keepdim=True) / qmax
            scale = torch.where(scale == 0, torch.ones_like(scale), scale)
            q = torch.round(w / scale).clamp(qmin, qmax)
            param.data = (q * scale).to(w.dtype)

def tl_features(model, sae, batch, hook_name):
    with torch.no_grad():
        _, cache = model.run_with_cache(batch, names_filter=[hook_name])
        acts = cache[hook_name].detach().reshape(-1, cache[hook_name].shape[-1])
        feats = sae.encode(acts.to(DEVICE).float()).detach().cpu().to(torch.float64)
    del cache, acts
    return feats

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

    def update(self, x, y):
        self.n += x.shape[0]
        self.sum_x += x.sum(dim=0); self.sum_y += y.sum(dim=0)
        self.sum_x2 += (x**2).sum(dim=0); self.sum_y2 += (y**2).sum(dim=0)
        self.sum_xy += (x * y).sum(dim=0)
        self.fire_count += (x > 0).sum(dim=0)

    def finalize(self):
        n = self.n
        numerator = self.sum_xy - (self.sum_x * self.sum_y / n)
        denom_x = self.sum_x2 - (self.sum_x ** 2 / n)
        denom_y = self.sum_y2 - (self.sum_y ** 2 / n)
        denominator = torch.sqrt(torch.clamp(denom_x * denom_y, min=1e-12))
        corr = torch.clamp(numerator / denominator, -1.0, 1.0)
        firing_rate = self.fire_count / n
        active = firing_rate > FIRING_THRESHOLD
        active_corr = corr[active]
        return {
            "n_tokens": int(n),
            "n_active": int(active.sum().item()),
            "mean_corr": float(active_corr.mean().item()),
            "median_corr": float(active_corr.median().item()),
            "survived_pct": float((active_corr > SURVIVAL_THRESHOLD).double().mean().item()) * 100,
            "damaged_pct": float((active_corr < DAMAGE_THRESHOLD).double().mean().item()) * 100,
        }, corr, firing_rate, active

def build_sequential_tokens(all_tokens, n_tokens, seq_len):
    """Take the FIRST n_tokens tokens. Used for budget-stability test."""
    n_seqs = n_tokens // seq_len
    usable = n_seqs * seq_len
    sliced = all_tokens[:usable].reshape(n_seqs, seq_len)
    return sliced.to(DEVICE)

def build_random_subset(all_tokens, n_tokens, seq_len, seed):
    """Sample a RANDOM subset of sequences from the full corpus.

    This is the correct version of seed variation: different seeds produce
    different subsets of sequences (different actual tokens), not just the
    same tokens in different order.
    """
    total_seqs_available = all_tokens.shape[0] // seq_len
    all_seq = all_tokens[:total_seqs_available * seq_len].reshape(total_seqs_available, seq_len)

    n_seqs_needed = n_tokens // seq_len
    if n_seqs_needed > total_seqs_available:
        raise ValueError(f"Need {n_seqs_needed} sequences but only {total_seqs_available} available")

    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    idx = torch.randperm(total_seqs_available, generator=g)[:n_seqs_needed]
    subset = all_seq[idx].contiguous()
    return subset.to(DEVICE)

def run_comparison(ref_fn, cond_fn, tokens_2d, batch_size, desc=""):
    """Stream-compare two activation pipelines."""
    stats = RunningFeatureStats(sae.cfg.d_sae)
    for i in tqdm(range(0, tokens_2d.shape[0], batch_size), desc=desc, leave=False):
        batch = tokens_2d[i:i+batch_size]
        x = ref_fn(batch)
        y = cond_fn(batch)
        stats.update(x, y)
        del x, y
    return stats.finalize()

print("Helpers ready.")

# %% [markdown]
# ## 5. Ablation A — Token-budget stability
# 
# Run RTN INT6 at 50k, 100k, 200k tokens. If survival/damaged percentages agree to within ~1-2 points, your chosen budget is statistically sufficient.

# %% Cell 11
ablation_A_rows = []

for tb in TOKEN_BUDGETS:
    print(f"\n=== Token budget = {tb:,} ===")
    tokens_2d = build_sequential_tokens(all_tokens, tb, SEQ_LEN)

    restore_model(model_work, original_state)
    quantize_rtn_per_channel_tl(model_work, bits=DAMAGE_BITS)

    ref_fn = lambda b: tl_features(model_ref, sae, b, HOOK_NAME)
    cond_fn = lambda b: tl_features(model_work, sae, b, HOOK_NAME)

    summary, _, _, _ = run_comparison(
        ref_fn, cond_fn, tokens_2d, BATCH_SIZE,
        desc=f"INT{DAMAGE_BITS} @ {tb//1000}k"
    )
    restore_model(model_work, original_state)

    row = {"token_budget": tb, "condition": f"RTN_INT{DAMAGE_BITS}", **summary}
    ablation_A_rows.append(row)
    print(f"  active={summary['n_active']}, mean_corr={summary['mean_corr']:.4f}, "
          f"survived={summary['survived_pct']:.2f}%, damaged={summary['damaged_pct']:.2f}%")

ablation_A_df = pd.DataFrame(ablation_A_rows)
ablation_A_df.to_csv(OUTPUT_DIR / "ablation_A_token_budget.csv", index=False)
print()
print("=== Ablation A: token-budget stability ===")
print(ablation_A_df.to_string(index=False))

max_dev_surv = ablation_A_df["survived_pct"].max() - ablation_A_df["survived_pct"].min()
max_dev_dmg = ablation_A_df["damaged_pct"].max() - ablation_A_df["damaged_pct"].min()
print(f"\nSurvived %% range: {max_dev_surv:.2f} pp")
print(f"Damaged %% range:  {max_dev_dmg:.2f} pp")
if max_dev_surv < 2.0 and max_dev_dmg < 2.0:
    print("✓ Stable: token budget does not materially affect results.")
else:
    print("⚠ Token budget affects results. Use the largest budget and document the sensitivity.")

# %% [markdown]
# ## 6. Ablation B — Seed stability (TRUE token-sampling variation)
# 
# Same condition (RTN INT6), same budget (100k tokens), but with 3 different **random sequence subsets** drawn from the full corpus. Each seed produces *different actual text*, not just the same text in a different order.

# %% Cell 13
STABILITY_BUDGET = 100_000
ablation_B_rows = []

for seed in SEEDS:
    print(f"\n=== Seed = {seed} (random subset) ===")
    tokens_2d = build_random_subset(all_tokens, STABILITY_BUDGET, SEQ_LEN, seed=seed)

    restore_model(model_work, original_state)
    quantize_rtn_per_channel_tl(model_work, bits=DAMAGE_BITS)

    ref_fn = lambda b: tl_features(model_ref, sae, b, HOOK_NAME)
    cond_fn = lambda b: tl_features(model_work, sae, b, HOOK_NAME)

    summary, _, _, _ = run_comparison(
        ref_fn, cond_fn, tokens_2d, BATCH_SIZE,
        desc=f"INT{DAMAGE_BITS} seed={seed}"
    )
    restore_model(model_work, original_state)

    row = {
        "seed": seed,
        "token_budget": STABILITY_BUDGET,
        "condition": f"RTN_INT{DAMAGE_BITS}",
        **summary
    }
    ablation_B_rows.append(row)
    print(f"  seed={seed}: mean_corr={summary['mean_corr']:.4f}, "
          f"survived={summary['survived_pct']:.2f}%, damaged={summary['damaged_pct']:.2f}%")

ablation_B_df = pd.DataFrame(ablation_B_rows)
ablation_B_df.to_csv(OUTPUT_DIR / "ablation_B_seed_stability.csv", index=False)

print()
print("=== Ablation B: seed stability (random subsets) ===")
print(ablation_B_df.to_string(index=False))

survived_std = ablation_B_df["survived_pct"].std()
damaged_std = ablation_B_df["damaged_pct"].std()
print(f"\nSurvived %% std across seeds: {survived_std:.3f} pp")
print(f"Damaged %%  std across seeds: {damaged_std:.3f} pp")

if survived_std < 2.0 and damaged_std < 1.0:
    print("✓ Stable: random sampling of tokens does not materially affect results.")
    print("  Variance across seeds is small, so observed effects are not artifacts of token choice.")
else:
    print("⚠ Seed variation is non-trivial. Report mean ± std in the paper.")

# %% [markdown]
# ## 7. Ablation C — Random baseline (FP16 vs FP16, identical inputs)
# 
# The strict null: same model, same tokens. Mean per-feature correlation must be 1.0 and damaged % must be 0. If it's not, the pipeline is generating false damage from nothing.

# %% Cell 15
print("=== FP16 vs FP16: strict null test ===")
tokens_2d_null = build_sequential_tokens(all_tokens, STABILITY_BUDGET, SEQ_LEN)

# Both pipelines use model_ref (always FP16). Same model, same tokens, same order.
# Should produce mean_corr = 1.0 exactly.
ref_fn = lambda b: tl_features(model_ref, sae, b, HOOK_NAME)
cond_fn = lambda b: tl_features(model_ref, sae, b, HOOK_NAME)
summary_strict, corr_strict, firing_strict, active_strict = run_comparison(
    ref_fn, cond_fn, tokens_2d_null, BATCH_SIZE, desc="strict null"
)
print(f"  mean_corr={summary_strict['mean_corr']:.6f}")
print(f"  survived={summary_strict['survived_pct']:.4f}%")
print(f"  damaged={summary_strict['damaged_pct']:.4f}%")
print(f"  Expected: mean_corr=1.000000, survived=100.0%, damaged=0.0%")

ablation_C_rows = [{
    "test": "strict null (same model, same tokens)",
    "n_active": summary_strict["n_active"],
    "mean_corr": summary_strict["mean_corr"],
    "survived_pct": summary_strict["survived_pct"],
    "damaged_pct": summary_strict["damaged_pct"],
}]
ablation_C_df = pd.DataFrame(ablation_C_rows)
ablation_C_df.to_csv(OUTPUT_DIR / "ablation_C_random_baseline.csv", index=False)

print()
if summary_strict["mean_corr"] > 0.999 and summary_strict["damaged_pct"] < 0.1:
    print("✓ Pipeline integrity confirmed: identical inputs produce ~1.0 correlations.")
else:
    print("⚠ STOP: pipeline generates spurious damage from identical inputs.")
    print("  All prior phase comparisons are suspect. Debug before continuing.")

# %% [markdown]
# ## 8. Combined methodology table

# %% Cell 17
print("=" * 80)
print("PHASE 4 SUMMARY: Methodology Ablations")
print("=" * 80)

print("\nAblation A — Token-budget stability (RTN INT6, varying tokens):")
print(ablation_A_df[["token_budget", "n_active", "mean_corr", "survived_pct", "damaged_pct"]].to_string(index=False))

print("\nAblation B — Seed stability (RTN INT6, 100k tokens, RANDOM subsets):")
print(ablation_B_df[["seed", "n_active", "mean_corr", "survived_pct", "damaged_pct"]].to_string(index=False))

print("\nAblation C — Pipeline integrity (FP16 vs FP16, same tokens):")
print(ablation_C_df.to_string(index=False))

print("\n\n=== Summary statistics for the paper ===")
print(f"Token-budget sensitivity (survived %% range across 50k/100k/200k): "
      f"{ablation_A_df['survived_pct'].max() - ablation_A_df['survived_pct'].min():.2f} pp")
print(f"Token-sample sensitivity (survived %% std across 3 random subsets): "
      f"{ablation_B_df['survived_pct'].std():.3f} pp")
print(f"Pipeline null baseline correlation: {ablation_C_df['mean_corr'].iloc[0]:.6f}")

combined = pd.concat([
    ablation_A_df.assign(ablation="A_token_budget"),
    ablation_B_df.assign(ablation="B_seed_stability"),
    ablation_C_df.assign(ablation="C_pipeline_integrity"),
], ignore_index=True, sort=False)
combined.to_csv(OUTPUT_DIR / "phase4_combined_summary.csv", index=False)
print(f"\nSaved: {OUTPUT_DIR / 'phase4_combined_summary.csv'}")

# %% [markdown]
# ## 9. Visualization for the paper

# %% Cell 19
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Panel A: token-budget stability
a = ablation_A_df.copy()
axes[0].plot(a["token_budget"]/1000, a["survived_pct"], marker='o', markersize=10,
             linewidth=2, color='seagreen', label='Survived >0.9')
axes[0].plot(a["token_budget"]/1000, a["damaged_pct"], marker='s', markersize=10,
             linewidth=2, color='crimson', label='Damaged <0.5')
axes[0].set_xlabel("Token budget (thousands)")
axes[0].set_ylabel("Feature %")
axes[0].set_title(f"A. Token-budget stability\n(RTN INT{DAMAGE_BITS}, layer {LAYER})")
axes[0].grid(True, alpha=0.3)
axes[0].legend()

# Panel B: seed stability (random subsets)
b = ablation_B_df.copy()
x = b["seed"].values
axes[1].bar(x - 0.2, b["survived_pct"], width=0.4, color='seagreen', label='Survived >0.9')
axes[1].bar(x + 0.2, b["damaged_pct"], width=0.4, color='crimson', label='Damaged <0.5')
axes[1].set_xlabel("Seed (random sequence subset)")
axes[1].set_ylabel("Feature %")
axes[1].set_title(f"B. Seed stability (random subsets)\n(RTN INT{DAMAGE_BITS}, 100k tokens)")
axes[1].set_xticks(x)
axes[1].grid(True, alpha=0.3, axis='y')
axes[1].legend()

# Panel C: pipeline integrity
c_corr = ablation_C_df["mean_corr"].iloc[0]
c_dmg = ablation_C_df["damaged_pct"].iloc[0]
axes[2].bar([0, 1], [c_corr, 1.0], width=0.6,
            color=['steelblue', 'lightgrey'],
            edgecolor='black')
axes[2].set_xticks([0, 1])
axes[2].set_xticklabels(["Observed mean corr", "Expected (1.0)"])
axes[2].set_ylabel("Mean feature correlation")
axes[2].set_ylim(min(c_corr, 0.98), 1.005)
axes[2].set_title(f"C. Pipeline integrity\n(FP16 vs FP16, damaged={c_dmg:.3f}%)")
axes[2].grid(True, alpha=0.3, axis='y')
axes[2].axhline(1.0, color='red', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "phase4_stability_ablations.png", dpi=150, bbox_inches='tight')
plt.show()
print(f"Saved: {OUTPUT_DIR / 'phase4_stability_ablations.png'}")

# %% [markdown]
# ## 10. Bonus: Threshold-sensitivity (post-hoc, no rerun needed)
# 
# The paper uses survival > 0.9 and damaged < 0.5 as thresholds. Reviewers will ask whether the qualitative finding depends on those specific values. This cell uses Ablation A's data to recompute at several thresholds; if the patterns hold across thresholds, the conclusion is robust.

# %% Cell 21
print("Recomputing INT6 at 200k tokens to extract per-feature correlations...")
tokens_2d = build_sequential_tokens(all_tokens, 200_000, SEQ_LEN)
restore_model(model_work, original_state)
quantize_rtn_per_channel_tl(model_work, bits=DAMAGE_BITS)

ref_fn = lambda b: tl_features(model_ref, sae, b, HOOK_NAME)
cond_fn = lambda b: tl_features(model_work, sae, b, HOOK_NAME)
summary, corr_arr, firing_arr, active_arr = run_comparison(
    ref_fn, cond_fn, tokens_2d, BATCH_SIZE,
    desc="INT6 @ 200k (for thresholds)"
)
restore_model(model_work, original_state)

active_corrs = corr_arr[active_arr]
print(f"Active features: {active_corrs.numel()}")

survival_thresholds = [0.80, 0.85, 0.90, 0.95]
damage_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]

threshold_rows = []
for st in survival_thresholds:
    for dt in damage_thresholds:
        if dt >= st:
            continue  # damage threshold must be below survival threshold
        survived = float((active_corrs > st).double().mean().item()) * 100
        damaged = float((active_corrs < dt).double().mean().item()) * 100
        degraded = 100 - survived - damaged
        threshold_rows.append({
            "survival_threshold": st,
            "damage_threshold": dt,
            "survived_pct": survived,
            "degraded_pct": degraded,
            "damaged_pct": damaged,
        })

threshold_df = pd.DataFrame(threshold_rows)
threshold_df.to_csv(OUTPUT_DIR / "ablation_D_threshold_sensitivity.csv", index=False)

print()
print("=== Threshold sensitivity (RTN INT6, 200k tokens) ===")
print(threshold_df.to_string(index=False))

# Check: does qualitative pattern (most features survive, small tail damaged) hold across thresholds?
print(f"\nAt the strictest survival threshold ({max(survival_thresholds)}): "
      f"{threshold_df[threshold_df['survival_threshold']==max(survival_thresholds)]['survived_pct'].iloc[0]:.1f}% survived")
print(f"At the most lenient survival threshold ({min(survival_thresholds)}): "
      f"{threshold_df[threshold_df['survival_threshold']==min(survival_thresholds)]['survived_pct'].iloc[0]:.1f}% survived")
print("\nIf qualitative pattern is preserved across thresholds, your conclusions are robust to threshold choice.")

# %% [markdown]
# ## 11. Paper paragraph (ready to adapt)
# 
# > **Statistical Reliability.** We verified three properties of our measurement pipeline. (i) **Token-budget stability**: applying RTN INT6 to Pythia-70M layer 4 at 50k, 100k, and 200k tokens yielded survival rates within X.XX percentage points of each other, indicating sufficient statistical power at our chosen budget. (ii) **Token-sample stability**: re-running the same condition on three independent random subsets of 100k tokens drawn from the full corpus produced a standard deviation of X.XXX percentage points in survival rate, substantially smaller than between-condition differences in our main results. (iii) **Pipeline integrity**: comparing FP16 to FP16 on identical inputs yielded a mean per-feature correlation of X.XXXXXX with X.XX% features classified as damaged, confirming our measurement does not generate spurious damage signals. We additionally verify (Appendix N) that our qualitative findings hold across survival thresholds in {0.80, 0.85, 0.90, 0.95} and damage thresholds in {0.3, 0.4, 0.5, 0.6, 0.7}, demonstrating robustness to threshold choice.
