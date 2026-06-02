# QDM Feature Survival Under Quantization
This repository contains the notebook and analysis artifacts for **How Quantization Changes Interpretable Features in Language Models**

The project asks whether sparse-autoencoder (SAE) features extracted from a full precision language model remain "faithful" after a model is quantized. The core measurement uses a frozen SAE as a fixed basis: FP16 and compressed activations are computed on identical tokens, encoded by the same SAE, and compared feature-by-feature with Pearson correlation.

## Main claims reproduced by this repo

- SAE feature survival degrades gradually as RTN bit-width decreases from INT8 to INT4.
- INT6 feature survival is predictable from full-precision feature statistics.
- Perplexity can remain stable or improve while SAE feature fidelity degrades.
- RTN quantization and matched-perplexity magnitude pruning damage strongly overlapping feature sets.

## Repository structure

```text
notebooks/
  00_smoke_test.ipynb                         # Tiny end-to-end QDM pipeline check
  00_smoke_test_v2.ipynb                      # Less aggressive smoke test
  01_pythia_phase2a_bitwidth_sweep.ipynb      # Early Pythia bit-width sweep
  02_pythia_phase2b_streaming_final.ipynb     # Main Pythia streaming run
  03_gemma_phase3_streaming_sweep.ipynb       # Main Gemma streaming run
  04_pythia_phase4_stability_ablations.ipynb  # Token budget, seed, null, layer checks
  05_phase5_feature_class_predictor.ipynb     # Logistic predictor + pruning overlap analysis
  06_gemma_sliding_window_ppl_check.ipynb     # Sliding-window perplexity robustness check
  legacy/                                     # Earlier exploratory notebooks retained for provenance
scripts/                                      # Auto-exported .py versions of curated notebooks
src/qdm_feature_survival/metrics.py           # Small reusable metric utilities
results/tables/                               # CSV/JSON summary tables
results/figures/                              # Figures used for analysis and appendix plots
results/per_feature/                          # Per-feature overlap CSVs
paper/draft_anonymous.pdf                     # Latest anonymized PDF draft included for context
```
