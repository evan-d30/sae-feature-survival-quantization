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
  legacy/
results
  results/tables/                               # CSV/JSON summary tables
  results/figures/                              # Figures used for analysis and appendix plots
  results/per_feature/                          # Per-feature overlap CSVs                                  
scripts/                                      # Auto-exported .py versions of curated notebooks
src/qdm_feature_survival/metrics.py           # Small reusable metric utilities
```

## Setup

Create an environment with pip:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

or conda

```bash
conda env create -f environment.yml
conda activate qdm-feature-survival
```

Note that for some Gemma runs may require accepting the relevant model license on HF and logging in locally or an API key. Notebook is written so HF login in optional.


## Reproduction order

Before running the main pipeline, we recommend a smoke test as we did for our original experiment.

For a quick check:

1. Run `notebooks/00_smoke_test.ipynb`.
2. Run `notebooks/00_smoke_test_v2.ipynb`.

For the main paper pipeline:

1. Run `notebooks/02_pythia_phase2b_streaming_final.ipynb` for Pythia-70M.
2. Run `notebooks/03_gemma_phase3_streaming_sweep.ipynb` for Gemma-2-2B.
3. Run `notebooks/04_pythia_phase4_stability_ablations.ipynb` for stability checks.
4. Run `notebooks/05_phase5_feature_class_predictor.ipynb` for feature-statistics prediction and pruning-overlap analysis.
5. Run `notebooks/06_gemma_sliding_window_ppl_check.ipynb` for the sliding-window perplexity check.

The notebooks include test/full run modes where applicable. Gemma runs require substantial ammount of time and computing power.

The scripts in `scripts/` are the recommended entry points for reproduction. The notebooks in `notebooks/` are included for transparency and exploratory provenance.

## Data and model dependencies

This repo does not include model weights, SAE weights, or raw WikiText data. The notebooks download models, SAEs, and WikiText-2 through standard libraries:

- Hugging Face `datasets`
- `transformer_lens`
- `sae_lens`
- Hugging Face model hub

## Included result artifacts

The `results/` directory contains the summary tables, feature-class prediction outputs, pruning-overlap tables, and figures used to assemble the paper's reported numbers. Large model checkpoints are intentionally excluded.

## License

Code is released under the MIT License. Dataset/model/SAE artifacts remain governed by their original licenses.
