# Auto-exported from 05_phase5_feature_class_predictor.ipynb
# Some notebooks are intended for interactive/Colab use.


# %% [markdown]
# # QDM Phase 5: Feature-Class Vulnerability Analysis
# 
# **What this does:** Post-hoc analysis on per-feature CSVs from Phases 2/3. No new model runs. The question is:

# %% [markdown]
# ## 1. Setup

# %% Cell 2
# !pip install -q pandas numpy matplotlib scikit-learn
print("Installed.")

# %% Cell 3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve

pd.set_option("display.float_format", "{:.4f}".format)
pd.set_option("display.max_columns", None)
print("Imports OK.")

# %% Cell 4
from pathlib import Path

search_root = Path("/workspace")

matches = list(search_root.rglob("*INT6_per_feature.csv"))

for m in matches:
    print(m)

# %% [markdown]
# ## 2. Config — adjust paths to your saved CSVs

# %% Cell 6
from pathlib import Path

# =========================
# Phase 5A file paths
# =========================
# Use FULL-RUN outputs only.
# Do NOT use:
#   phase2b_outputs_test_20k_L4/
#   phase3_gemma_outputs_3a_test_20k_L12/

PYTHIA_PF_PATH = Path("/workspace/phase2b_outputs_full_200k_L4/RTN_INT6_per_feature.csv")
GEMMA_PF_PATH = Path("/workspace/phase3_gemma_outputs_3b_full_500k_L12/RTN_INT6_per_feature.csv")

# Survival threshold: feature is counted as "survived" if corr > 0.9
SURVIVAL_THRESHOLD = 0.9

# Output directory for Phase 5A artifacts
OUT_DIR = Path("/workspace/phase5_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Check files exist
print(f"Pythia CSV: {PYTHIA_PF_PATH.exists()} — {PYTHIA_PF_PATH}")
print(f"Gemma CSV:  {GEMMA_PF_PATH.exists()} — {GEMMA_PF_PATH}")
print(f"Output dir: {OUT_DIR.exists()} — {OUT_DIR}")

if not PYTHIA_PF_PATH.exists():
    raise FileNotFoundError(f"Pythia per-feature CSV not found: {PYTHIA_PF_PATH}")

if not GEMMA_PF_PATH.exists():
    raise FileNotFoundError(f"Gemma per-feature CSV not found: {GEMMA_PF_PATH}")

# %% [markdown]
# ## 3. Load + inspect
# 
# Your per-feature CSVs should have at minimum: `feature_id`, `corr`, `firing_rate`, `mean_activation`, `max_activation`, `active`. If your column names differ, adjust below.

# %% Cell 8
def load_pf(path, label):
    df = pd.read_csv(path)
    print(f"\n=== {label}: {path.name} ===")
    print(f"  rows: {len(df):,}")
    print(f"  columns: {df.columns.tolist()}")
    if "active" in df.columns:
        n_active = df["active"].sum()
        print(f"  active features: {n_active:,}")
    return df

pythia_df = load_pf(PYTHIA_PF_PATH, "Pythia-70M")
gemma_df = load_pf(GEMMA_PF_PATH, "Gemma-2-2B")

# %% [markdown]
# ## 4. Feature engineering
# 
# For each active feature, derive properties from the FP16 measurements that might predict fragility:

# %% Cell 10
def engineer_features(df, label):
    """Returns X (feature matrix), y (survival binary), feature_names."""
    # Restrict to features active in FP16
    if "active" in df.columns:
        df = df[df["active"] == True].copy()
    else:
        df = df[df["firing_rate"] > 0.001].copy()

    # Binary outcome: survived (corr > threshold) or not
    df["survived"] = (df["corr"] > SURVIVAL_THRESHOLD).astype(int)

    # Derived properties
    df["rarity"] = -np.log(df["firing_rate"].clip(lower=1e-8))
    df["log_mean_activation"] = np.log(df["mean_activation"].clip(lower=1e-8))
    df["log_max_activation"] = np.log(df["max_activation"].clip(lower=1e-8))
    df["activation_concentration"] = df["max_activation"] / df["mean_activation"].clip(lower=1e-8)
    df["log_concentration"] = np.log(df["activation_concentration"].clip(lower=1.0))

    feature_cols = [
        "rarity",
        "log_mean_activation",
        "log_max_activation",
        "log_concentration",
    ]

    # Drop any rows with NaN/inf
    df_clean = df[feature_cols + ["survived", "corr", "firing_rate"]].copy()
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan).dropna()

    X = df_clean[feature_cols].values
    y = df_clean["survived"].values

    print(f"\n=== {label} engineered ===")
    print(f"  Active features used: {len(df_clean):,}")
    print(f"  Survived (corr > {SURVIVAL_THRESHOLD}): {y.sum():,} ({y.mean()*100:.1f}%)")
    print(f"  Damaged/degraded: {(1-y).sum():,} ({(1-y).mean()*100:.1f}%)")

    return X, y, feature_cols, df_clean

X_pythia, y_pythia, feature_cols, pythia_clean = engineer_features(pythia_df, "Pythia-70M")
X_gemma, y_gemma, _, gemma_clean = engineer_features(gemma_df, "Gemma-2-2B")

# %% [markdown]
# ## 5. Fit vulnerability predictor
# 
# Logistic regression with L2 regularization, standardized inputs, 5-fold cross-validated AUC. The AUC tells you how well feature properties predict survival; the coefficients tell you which properties matter.

# %% Cell 12
def fit_vulnerability_predictor(X, y, feature_cols, label):
    """Returns dict with AUC, coefficients, and the trained model."""
    if len(np.unique(y)) < 2:
        print(f"⚠ {label}: only one class present, can't fit predictor")
        return None

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 5-fold cross-validated AUC
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    clf = LogisticRegression(max_iter=1000, C=1.0)
    cv_aucs = cross_val_score(clf, X_scaled, y, cv=cv, scoring="roc_auc")

    # Fit on all data for coefficient inspection
    clf.fit(X_scaled, y)
    coefs = pd.DataFrame({
        "property": feature_cols,
        "standardized_coef": clf.coef_[0],
        "abs_coef": np.abs(clf.coef_[0]),
    }).sort_values("abs_coef", ascending=False).reset_index(drop=True)

    print(f"\n=== {label} vulnerability predictor ===")
    print(f"  5-fold CV AUC: {cv_aucs.mean():.3f} ± {cv_aucs.std():.3f}")
    print(f"  Class balance: {y.mean()*100:.1f}% survived")
    print(f"\n  Standardized coefficients (positive = predicts survival):")
    print(coefs.to_string(index=False))

    return {
        "label": label,
        "cv_auc_mean": float(cv_aucs.mean()),
        "cv_auc_std": float(cv_aucs.std()),
        "cv_aucs": cv_aucs.tolist(),
        "coefficients": coefs,
        "model": clf,
        "scaler": scaler,
        "n_samples": len(y),
        "frac_survived": float(y.mean()),
    }

results_pythia = fit_vulnerability_predictor(X_pythia, y_pythia, feature_cols, "Pythia-70M")
results_gemma = fit_vulnerability_predictor(X_gemma, y_gemma, feature_cols, "Gemma-2-2B")

# %% [markdown]
# ## 6. Cross-model comparison of coefficients
# 
# If the same properties predict fragility in both Pythia and Gemma (same sign, similar magnitude), the finding is general. If the coefficients diverge, fragility predictors are model-specific.

# %% Cell 14
if results_pythia and results_gemma:
    coef_compare = pd.merge(
        results_pythia["coefficients"][["property", "standardized_coef"]].rename(
            columns={"standardized_coef": "coef_pythia"}
        ),
        results_gemma["coefficients"][["property", "standardized_coef"]].rename(
            columns={"standardized_coef": "coef_gemma"}
        ),
        on="property",
    )
    coef_compare["sign_agreement"] = (
        np.sign(coef_compare["coef_pythia"]) == np.sign(coef_compare["coef_gemma"])
    )
    coef_compare["abs_avg"] = (
        coef_compare["coef_pythia"].abs() + coef_compare["coef_gemma"].abs()
    ) / 2
    coef_compare = coef_compare.sort_values("abs_avg", ascending=False).reset_index(drop=True)

    print("=== Cross-model coefficient comparison ===")
    print(coef_compare.to_string(index=False))

    n_agree = coef_compare["sign_agreement"].sum()
    n_total = len(coef_compare)
    print(f"\n  Sign agreement: {n_agree}/{n_total} properties")
    if n_agree == n_total:
        print("  ✓ All properties agree in sign across models — predictors generalize")
    else:
        print("  ⚠ Some properties differ in sign — predictors are partially model-specific")

    coef_compare.to_csv(OUT_DIR / "coefficient_comparison.csv", index=False)

# %% [markdown]
# ## 7. Survival rate by feature property (quartile binning)
# 
# Visualizes the relationship between each property and survival. If "rare features die first," you should see survival rate decrease monotonically as rarity increases.

# %% Cell 16
def survival_by_quartile(df_clean, prop, label, ax):
    """Plot survival rate by quartiles of `prop`."""
    df_clean = df_clean.copy()
    df_clean[f"{prop}_q"] = pd.qcut(df_clean[prop], q=4, labels=["Q1 (low)", "Q2", "Q3", "Q4 (high)"], duplicates="drop")
    grouped = df_clean.groupby(f"{prop}_q", observed=True).agg(
        survival=("survived", "mean"),
        n=("survived", "size"),
    ).reset_index()
    grouped["survival_pct"] = grouped["survival"] * 100

    ax.bar(range(len(grouped)), grouped["survival_pct"], color="seagreen", edgecolor="black")
    for i, (rate, n) in enumerate(zip(grouped["survival_pct"], grouped["n"])):
        ax.text(i, rate + 1, f"{rate:.0f}%\nn={n}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(range(len(grouped)))
    ax.set_xticklabels(grouped[f"{prop}_q"], rotation=0, fontsize=9)
    ax.set_ylabel("Survival rate (%)")
    ax.set_title(f"{label}: survival by {prop}", fontsize=10)
    ax.set_ylim(0, max(105, grouped["survival_pct"].max() + 10))
    ax.grid(True, alpha=0.3, axis="y")

# 4 properties × 2 models = 8 panels (2 rows × 4 cols)
properties_to_plot = ["rarity", "log_mean_activation", "log_max_activation", "log_concentration"]

fig, axes = plt.subplots(2, 4, figsize=(20, 9))
for j, prop in enumerate(properties_to_plot):
    survival_by_quartile(pythia_clean, prop, "Pythia-70M", axes[0, j])
    survival_by_quartile(gemma_clean, prop, "Gemma-2-2B", axes[1, j])
plt.tight_layout()
plt.savefig(OUT_DIR / "survival_by_quartile.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 8. ROC curves (visualizing the predictor)

# %% Cell 18
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for ax, results, X, y in [(axes[0], results_pythia, X_pythia, y_pythia),
                          (axes[1], results_gemma, X_gemma, y_gemma)]:
    if results is None:
        ax.text(0.5, 0.5, "No predictor (single class)", ha="center", va="center")
        continue
    X_scaled = results["scaler"].transform(X)
    y_score = results["model"].predict_proba(X_scaled)[:, 1]
    fpr, tpr, _ = roc_curve(y, y_score)
    auc = roc_auc_score(y, y_score)
    ax.plot(fpr, tpr, linewidth=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"{results['label']}: feature-survival ROC\n(CV AUC = {results['cv_auc_mean']:.3f} ± {results['cv_auc_std']:.3f})")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_DIR / "roc_curves.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 9. Coefficient comparison bar chart

# %% Cell 20
if results_pythia and results_gemma:
    cc = coef_compare.copy()
    x = np.arange(len(cc))
    w = 0.4

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - w/2, cc["coef_pythia"], width=w, color="steelblue", label="Pythia-70M")
    ax.bar(x + w/2, cc["coef_gemma"], width=w, color="orange", label="Gemma-2-2B")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(cc["property"], rotation=15, ha="right")
    ax.set_ylabel("Standardized coefficient\n(positive = predicts survival)")
    ax.set_title("Feature properties predicting survival under RTN INT6")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "coefficient_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()

# %% [markdown]
# ## 10. Summary for paper
# 
# The interpretation guide depending on what you see:

# %% Cell 22
print("=" * 70)
print("PHASE 5 SUMMARY")
print("=" * 70)

if results_pythia:
    print(f"\nPythia-70M:")
    print(f"  Active features analyzed: {results_pythia['n_samples']:,}")
    print(f"  Fraction surviving INT6: {results_pythia['frac_survived']*100:.1f}%")
    print(f"  CV AUC for survival prediction: {results_pythia['cv_auc_mean']:.3f} ± {results_pythia['cv_auc_std']:.3f}")

if results_gemma:
    print(f"\nGemma-2-2B:")
    print(f"  Active features analyzed: {results_gemma['n_samples']:,}")
    print(f"  Fraction surviving INT6: {results_gemma['frac_survived']*100:.1f}%")
    print(f"  CV AUC for survival prediction: {results_gemma['cv_auc_mean']:.3f} ± {results_gemma['cv_auc_std']:.3f}")

if results_pythia and results_gemma:
    print(f"\nCross-model coefficient sign agreement: "
          f"{coef_compare['sign_agreement'].sum()}/{len(coef_compare)}")

# Save the combined summary
summary = {
    "pythia": {
        "n_samples": int(results_pythia["n_samples"]) if results_pythia else None,
        "frac_survived": float(results_pythia["frac_survived"]) if results_pythia else None,
        "cv_auc_mean": float(results_pythia["cv_auc_mean"]) if results_pythia else None,
        "cv_auc_std": float(results_pythia["cv_auc_std"]) if results_pythia else None,
        "coefficients": results_pythia["coefficients"].to_dict("records") if results_pythia else None,
    },
    "gemma": {
        "n_samples": int(results_gemma["n_samples"]) if results_gemma else None,
        "frac_survived": float(results_gemma["frac_survived"]) if results_gemma else None,
        "cv_auc_mean": float(results_gemma["cv_auc_mean"]) if results_gemma else None,
        "cv_auc_std": float(results_gemma["cv_auc_std"]) if results_gemma else None,
        "coefficients": results_gemma["coefficients"].to_dict("records") if results_gemma else None,
    },
}

import json as _json
with open(OUT_DIR / "phase5_summary.json", "w") as f:
    _json.dump(summary, f, indent=2)
print(f"\nSaved: {OUT_DIR / 'phase5_summary.json'}")

# %% Cell 23
# ============================================================
# Phase 5B: RTN INT6 vs matched-pruning per-feature overlap
# ============================================================

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# File paths
# -------------------------
PYTHIA_RTN_PATH = Path("/workspace/phase2b_outputs_full_200k_L4/RTN_INT6_per_feature.csv")
PYTHIA_PRUNE_PATH = Path("/workspace/phase2b_outputs_full_200k_L4/Magnitude_pruning_matched_INT6_per_feature.csv")

GEMMA_RTN_PATH = Path("/workspace/phase3_gemma_outputs_3b_full_500k_L12/RTN_INT6_per_feature.csv")
GEMMA_PRUNE_PATH = Path("/workspace/phase3_gemma_outputs_3b_full_500k_L12/Magnitude_pruning_matched_INT6_per_feature.csv")

OUT_DIR = Path("/workspace/phase5_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("Pythia RTN:", PYTHIA_RTN_PATH.exists(), PYTHIA_RTN_PATH)
print("Pythia Prune:", PYTHIA_PRUNE_PATH.exists(), PYTHIA_PRUNE_PATH)
print("Gemma RTN:", GEMMA_RTN_PATH.exists(), GEMMA_RTN_PATH)
print("Gemma Prune:", GEMMA_PRUNE_PATH.exists(), GEMMA_PRUNE_PATH)

for path in [PYTHIA_RTN_PATH, PYTHIA_PRUNE_PATH, GEMMA_RTN_PATH, GEMMA_PRUNE_PATH]:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

# %% Cell 24
# ============================================================
# Helper functions
# ============================================================

def jaccard_from_masks(df, mask_a, mask_b, id_col="feature_id"):
    """
    Jaccard overlap = |A ∩ B| / |A ∪ B|
    """
    a = set(df.loc[mask_a, id_col])
    b = set(df.loc[mask_b, id_col])

    union = a | b
    inter = a & b

    if len(union) == 0:
        return np.nan

    return len(inter) / len(union)


def load_and_merge_overlap(model_name, rtn_path, prune_path):
    """
    Load RTN INT6 and matched-pruning per-feature CSVs,
    merge by feature_id, keep active features, and compute overlap metrics.
    """
    rtn = pd.read_csv(rtn_path)
    prune = pd.read_csv(prune_path)

    print(f"\n=== {model_name}: loaded files ===")
    print("RTN rows:", len(rtn), "columns:", list(rtn.columns))
    print("Prune rows:", len(prune), "columns:", list(prune.columns))

    merged = rtn.merge(
        prune,
        on="feature_id",
        suffixes=("_rtn", "_prune")
    )

    # Keep features active in the FP16 reference.
    # Usually both files have the same active mask; use RTN as source of truth.
    if "active_rtn" in merged.columns:
        merged = merged[merged["active_rtn"] == True].copy()
    elif "active" in merged.columns:
        merged = merged[merged["active"] == True].copy()
    else:
        print("Warning: no active column found; using all merged features.")

    # Standardize expected columns
    if "corr_rtn" not in merged.columns or "corr_prune" not in merged.columns:
        raise ValueError("Expected corr_rtn and corr_prune after merge. Check CSV columns.")

    # Damage scores: larger = more feature drift
    merged["damage_rtn"] = 1.0 - merged["corr_rtn"]
    merged["damage_prune"] = 1.0 - merged["corr_prune"]

    # Survival/degradation/damage masks
    merged["rtn_survived"] = merged["corr_rtn"] > 0.9
    merged["prune_survived"] = merged["corr_prune"] > 0.9

    merged["rtn_non_survived"] = merged["corr_rtn"] <= 0.9
    merged["prune_non_survived"] = merged["corr_prune"] <= 0.9

    merged["rtn_damaged"] = merged["corr_rtn"] < 0.5
    merged["prune_damaged"] = merged["corr_prune"] < 0.5

    merged["rtn_degraded"] = (merged["corr_rtn"] >= 0.5) & (merged["corr_rtn"] <= 0.9)
    merged["prune_degraded"] = (merged["corr_prune"] >= 0.5) & (merged["corr_prune"] <= 0.9)

    # Overlap metrics
    summary = {
        "model": model_name,
        "n_active_features": len(merged),

        "rtn_survived_pct": merged["rtn_survived"].mean() * 100,
        "prune_survived_pct": merged["prune_survived"].mean() * 100,

        "rtn_non_survived_pct": merged["rtn_non_survived"].mean() * 100,
        "prune_non_survived_pct": merged["prune_non_survived"].mean() * 100,

        "rtn_degraded_pct": merged["rtn_degraded"].mean() * 100,
        "prune_degraded_pct": merged["prune_degraded"].mean() * 100,

        "rtn_damaged_pct": merged["rtn_damaged"].mean() * 100,
        "prune_damaged_pct": merged["prune_damaged"].mean() * 100,

        # Main overlap metric: meaningful because many features are below corr <= 0.9
        "jaccard_non_survived_corr_le_0.9": jaccard_from_masks(
            merged,
            merged["rtn_non_survived"],
            merged["prune_non_survived"]
        ),

        # Secondary overlap metric: may be unstable if very few features are <0.5
        "jaccard_damaged_corr_lt_0.5": jaccard_from_masks(
            merged,
            merged["rtn_damaged"],
            merged["prune_damaged"]
        ),

        # Damage-score agreement
        "pearson_damage_score_corr": merged["damage_rtn"].corr(
            merged["damage_prune"],
            method="pearson"
        ),
        "spearman_damage_score_corr": merged["damage_rtn"].corr(
            merged["damage_prune"],
            method="spearman"
        ),

        # Raw feature-correlation agreement
        "pearson_feature_corr": merged["corr_rtn"].corr(
            merged["corr_prune"],
            method="pearson"
        ),
        "spearman_feature_corr": merged["corr_rtn"].corr(
            merged["corr_prune"],
            method="spearman"
        ),
    }

    summary_df = pd.DataFrame([summary])

    return merged, summary_df

# %% Cell 25
# ============================================================
# Run overlap analysis for Pythia and Gemma
# ============================================================

pythia_overlap, pythia_overlap_summary = load_and_merge_overlap(
    model_name="Pythia-70M",
    rtn_path=PYTHIA_RTN_PATH,
    prune_path=PYTHIA_PRUNE_PATH
)

gemma_overlap, gemma_overlap_summary = load_and_merge_overlap(
    model_name="Gemma-2-2B",
    rtn_path=GEMMA_RTN_PATH,
    prune_path=GEMMA_PRUNE_PATH
)

overlap_summary = pd.concat(
    [pythia_overlap_summary, gemma_overlap_summary],
    ignore_index=True
)

pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", "{:.4f}".format)

display(overlap_summary)

# Save outputs
pythia_overlap.to_csv(OUT_DIR / "phase5b_pythia_rtn_int6_vs_pruning_per_feature_overlap.csv", index=False)
gemma_overlap.to_csv(OUT_DIR / "phase5b_gemma_rtn_int6_vs_pruning_per_feature_overlap.csv", index=False)
overlap_summary.to_csv(OUT_DIR / "phase5b_rtn_int6_vs_pruning_overlap_summary.csv", index=False)

print("Saved:")
print(OUT_DIR / "phase5b_pythia_rtn_int6_vs_pruning_per_feature_overlap.csv")
print(OUT_DIR / "phase5b_gemma_rtn_int6_vs_pruning_per_feature_overlap.csv")
print(OUT_DIR / "phase5b_rtn_int6_vs_pruning_overlap_summary.csv")

# %% Cell 26
# ============================================================
# Scatter plots: RTN correlation vs pruning correlation
# ============================================================

def plot_overlap_scatter(df, model_name, out_path):
    plt.figure(figsize=(6, 6))

    plt.scatter(
        df["corr_rtn"],
        df["corr_prune"],
        s=8,
        alpha=0.25
    )

    plt.axvline(0.9, linestyle="--", alpha=0.7, label="survival threshold 0.9")
    plt.axhline(0.9, linestyle="--", alpha=0.7)
    plt.axvline(0.5, linestyle=":", alpha=0.7, label="damage threshold 0.5")
    plt.axhline(0.5, linestyle=":", alpha=0.7)

    plt.xlabel("RTN INT6 feature correlation")
    plt.ylabel("Matched-pruning feature correlation")
    plt.title(f"{model_name}: RTN INT6 vs matched pruning")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(out_path, dpi=180)
    plt.show()

    print("Saved:", out_path)


plot_overlap_scatter(
    pythia_overlap,
    "Pythia-70M layer 4",
    OUT_DIR / "phase5b_pythia_rtn_vs_pruning_scatter.png"
)

plot_overlap_scatter(
    gemma_overlap,
    "Gemma-2-2B layer 12",
    OUT_DIR / "phase5b_gemma_rtn_vs_pruning_scatter.png"
)

# %% Cell 27
# ============================================================
# Firing-rate decile comparison
# ============================================================

def firing_rate_decile_analysis(df, model_name, out_prefix):
    """
    Compare RTN vs pruning vulnerability across FP16 firing-rate deciles.
    """
    # Find the firing-rate column.
    # After merge, it is usually firing_rate_rtn and firing_rate_prune.
    if "firing_rate_rtn" in df.columns:
        firing_col = "firing_rate_rtn"
    elif "firing_rate" in df.columns:
        firing_col = "firing_rate"
    else:
        raise ValueError("No firing_rate column found.")

    work = df.copy()

    # qcut can fail if too many repeated values, so duplicates='drop'
    work["firing_decile"] = pd.qcut(
        work[firing_col],
        q=10,
        labels=False,
        duplicates="drop"
    )

    decile_df = work.groupby("firing_decile").agg(
        n_features=("feature_id", "count"),
        mean_firing_rate=(firing_col, "mean"),

        rtn_survived_pct=("rtn_survived", lambda x: x.mean() * 100),
        prune_survived_pct=("prune_survived", lambda x: x.mean() * 100),

        rtn_non_survived_pct=("rtn_non_survived", lambda x: x.mean() * 100),
        prune_non_survived_pct=("prune_non_survived", lambda x: x.mean() * 100),

        rtn_damaged_pct=("rtn_damaged", lambda x: x.mean() * 100),
        prune_damaged_pct=("prune_damaged", lambda x: x.mean() * 100),

        rtn_mean_damage=("damage_rtn", "mean"),
        prune_mean_damage=("damage_prune", "mean"),
    ).reset_index()

    display(decile_df)

    csv_path = OUT_DIR / f"{out_prefix}_firing_rate_deciles.csv"
    decile_df.to_csv(csv_path, index=False)

    # Plot non-survived by firing-rate decile
    plt.figure(figsize=(7, 5))

    plt.plot(
        decile_df["firing_decile"],
        decile_df["rtn_non_survived_pct"],
        marker="o",
        label="RTN INT6"
    )

    plt.plot(
        decile_df["firing_decile"],
        decile_df["prune_non_survived_pct"],
        marker="o",
        label="Matched pruning"
    )

    plt.xlabel("FP16 firing-rate decile")
    plt.ylabel("Non-survived features (%)")
    plt.title(f"{model_name}: vulnerability by firing-rate decile")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plot_path = OUT_DIR / f"{out_prefix}_firing_rate_deciles.png"
    plt.savefig(plot_path, dpi=180)
    plt.show()

    print("Saved:", csv_path)
    print("Saved:", plot_path)

    return decile_df


pythia_deciles = firing_rate_decile_analysis(
    pythia_overlap,
    "Pythia-70M layer 4",
    "phase5b_pythia_rtn_vs_pruning"
)

gemma_deciles = firing_rate_decile_analysis(
    gemma_overlap,
    "Gemma-2-2B layer 12",
    "phase5b_gemma_rtn_vs_pruning"
)

# %% Cell 28
# ============================================================
# Optional: overlap counts table for paper
# ============================================================

def overlap_counts(df, model_name):
    rows = []

    for label, rtn_mask, prune_mask in [
        ("non_survived_corr_le_0.9", df["rtn_non_survived"], df["prune_non_survived"]),
        ("damaged_corr_lt_0.5", df["rtn_damaged"], df["prune_damaged"]),
        ("degraded_0.5_to_0.9", df["rtn_degraded"], df["prune_degraded"]),
    ]:
        rtn_set = set(df.loc[rtn_mask, "feature_id"])
        prune_set = set(df.loc[prune_mask, "feature_id"])

        intersection = rtn_set & prune_set
        union = rtn_set | prune_set

        rows.append({
            "model": model_name,
            "set_type": label,
            "rtn_count": len(rtn_set),
            "prune_count": len(prune_set),
            "intersection_count": len(intersection),
            "union_count": len(union),
            "jaccard": len(intersection) / len(union) if len(union) > 0 else np.nan,
            "rtn_only_count": len(rtn_set - prune_set),
            "prune_only_count": len(prune_set - rtn_set),
        })

    return pd.DataFrame(rows)


pythia_counts = overlap_counts(pythia_overlap, "Pythia-70M")
gemma_counts = overlap_counts(gemma_overlap, "Gemma-2-2B")

overlap_counts_df = pd.concat([pythia_counts, gemma_counts], ignore_index=True)

display(overlap_counts_df)

overlap_counts_df.to_csv(
    OUT_DIR / "phase5b_overlap_counts_table.csv",
    index=False
)

print("Saved:", OUT_DIR / "phase5b_overlap_counts_table.csv")

# %% Cell 29
import pandas as pd
from pathlib import Path

summary_paths = [
    Path("/workspace/phase2b_outputs_full_200k_L4/phase2b_summary_final.csv"),
    Path("/workspace/phase2b_outputs_full_200k_L4/pruning_match_search.csv"),
    Path("/workspace/phase3_gemma_outputs_3b_full_500k_L12/phase3_summary_final.csv"),
]

for p in summary_paths:
    print("\n===", p, "===")
    print("exists:", p.exists())
    if p.exists():
        df = pd.read_csv(p)
        print(df.columns.tolist())
        display(df)

# %% Cell 30
from pathlib import Path
import os

ROOT = Path("/workspace")

folders = [
    "qdm_phase2a_results",
    "phase2b_outputs_full_200k_L4",
    "phase3_gemma_outputs_3b_full_500k_L12",
    "phase4_outputs_L4",
    "phase5_outputs",
]

for folder in folders:
    path = ROOT / folder
    print("\n" + "=" * 90)
    print(f"{folder}")
    print("=" * 90)

    if not path.exists():
        print("MISSING:", path)
        continue

    files = sorted([p for p in path.rglob("*") if p.is_file()])

    for f in files:
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"{str(f):90s}  {size_mb:8.3f} MB")

print("\nDone.")

# %% Cell 31
from pathlib import Path
import shutil

DELIVERABLES = Path("/workspace/deliverables")
DELIVERABLES.mkdir(parents=True, exist_ok=True)

files_by_section = {
    "01_core_summaries": [
        "/workspace/qdm_phase2a_results/phase2a_combined.csv",
        "/workspace/phase2b_outputs_full_200k_L4/phase2b_summary_final.csv",
        "/workspace/phase3_gemma_outputs_3b_full_500k_L12/phase3_summary_final.csv",
        "/workspace/phase4_outputs_L4/phase4_combined_summary.csv",
        "/workspace/phase4_outputs_L4/ablation_D_threshold_sensitivity.csv",
        "/workspace/phase5_outputs/phase5_summary.json",
        "/workspace/phase5_outputs/phase5b_rtn_int6_vs_pruning_overlap_summary.csv",
        "/workspace/phase5_outputs/phase5b_overlap_counts_table.csv",
    ],

    "02_main_figures": [
        "/workspace/qdm_phase2a_results/phase2a_layer_comparison.png",
        "/workspace/phase2b_outputs_full_200k_L4/phase2b_rtn_sweep.png",
        "/workspace/phase2b_outputs_full_200k_L4/phase2b_all_conditions_bar.png",
        "/workspace/phase3_gemma_outputs_3b_full_500k_L12/phase3_rtn_sweep.png",
        "/workspace/phase3_gemma_outputs_3b_full_500k_L12/phase3_all_conditions_bar.png",
        "/workspace/phase4_outputs_L4/phase4_stability_ablations.png",
        "/workspace/phase5_outputs/survival_by_quartile.png",
        "/workspace/phase5_outputs/roc_curves.png",
        "/workspace/phase5_outputs/coefficient_comparison.png",
        "/workspace/phase5_outputs/phase5b_pythia_rtn_vs_pruning_scatter.png",
        "/workspace/phase5_outputs/phase5b_gemma_rtn_vs_pruning_scatter.png",
        "/workspace/phase5_outputs/phase5b_pythia_rtn_vs_pruning_firing_rate_deciles.png",
        "/workspace/phase5_outputs/phase5b_gemma_rtn_vs_pruning_firing_rate_deciles.png",
    ],

    "03_int7_diagnostics": [
        "/workspace/phase3_gemma_outputs_3b_full_500k_L12/int7_investigation_corrected_ppl_sweep_50k.csv",
        "/workspace/phase3_gemma_outputs_3b_full_500k_L12/int7_subset_reproducibility.csv",
        "/workspace/phase3_gemma_outputs_3b_full_500k_L12/int7_quantization_weight_diagnostics_summary.csv",
        "/workspace/phase3_gemma_outputs_3b_full_500k_L12/int7_logit_drift_check_50k_summary.csv",
    ],

    "04_pruning_calibration": [
        "/workspace/phase2b_outputs_full_200k_L4/pruning_match_search.csv",
        "/workspace/phase3_gemma_outputs_3b_full_500k_L12/pruning_match_search.csv",
    ],

    "05_phase5b_decile_csvs": [
        "/workspace/phase5_outputs/phase5b_pythia_rtn_vs_pruning_firing_rate_deciles.csv",
        "/workspace/phase5_outputs/phase5b_gemma_rtn_vs_pruning_firing_rate_deciles.csv",
    ],

    "06_optional_verification": [
        "/workspace/qdm_phase2a_results/phase2a_L4_summary.csv",
        "/workspace/qdm_phase2a_results/phase2a5_L2_summary.csv",
        "/workspace/phase2b_outputs_full_200k_L4/RTN_INT6_summary.csv",
        "/workspace/phase2b_outputs_full_200k_L4/Magnitude_pruning_matched_INT6_summary.csv",
        "/workspace/phase3_gemma_outputs_3b_full_500k_L12/RTN_INT6_summary.csv",
        "/workspace/phase3_gemma_outputs_3b_full_500k_L12/Magnitude_pruning_matched_INT6_summary.csv",
        "/workspace/phase5_outputs/coefficient_comparison.csv",
    ],
}

missing = []
copied = []

for section, file_list in files_by_section.items():
    section_dir = DELIVERABLES / section
    section_dir.mkdir(parents=True, exist_ok=True)

    for file_path in file_list:
        src = Path(file_path)

        if not src.exists():
            missing.append(str(src))
            print("MISSING:", src)
            continue

        dst = section_dir / src.name

        # Avoid filename collisions by prefixing parent folder if needed
        if dst.exists():
            dst = section_dir / f"{src.parent.name}__{src.name}"

        shutil.copy2(src, dst)
        copied.append(str(dst))
        print("Copied:", src, "->", dst)

print("\n" + "=" * 80)
print(f"Copied {len(copied)} files into {DELIVERABLES}")
print(f"Missing {len(missing)} files")

if missing:
    print("\nMissing files:")
    for m in missing:
        print(" -", m)

# %% [markdown]
# ## 11. Paper paragraph (adapt with your actual numbers)
# 
# > **Feature-class vulnerability.** To test whether feature fragility under quantization is structured or random, we fit logistic regressions predicting feature survival (correlation > 0.9 with FP16 features under RTN INT6) from four properties measured in FP16: rarity (negative log firing rate), mean activation magnitude, peak activation, and activation concentration (max/mean ratio). On Pythia-70M, 5-fold cross-validated AUC was [X.XXX ± X.XXX]; on Gemma-2-2B, AUC was [X.XXX ± X.XXX]. [N/4] standardized coefficients agreed in sign across models. The dominant predictor in both models was [X], with positive/negative sign indicating that [more rare/more frequent] features were systematically more likely to [survive/degrade]. These results indicate that quantization-induced feature damage is non-random: certain feature classes are predictably more fragile, which has implications for selecting interpretability targets that remain robust under deployment quantization.
