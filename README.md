# QDM Feature Survival Under Quantization
This repository contains the notebook and analysis artifacts for **How Quantization Changes Interpretable Features in Language Models**

The project asks whether sparse-autoencoder (SAE) features extracted from a full precision language model remain "faithful" after a model is quantized. The core measurement uses a frozen SAE as a fixed basis: FP16 and compressed activations are computed on identical tokens, encoded by the same SAE, and compared feature-by-feature with Pearson correlation.

