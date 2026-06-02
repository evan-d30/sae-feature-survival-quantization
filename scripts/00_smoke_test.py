# Auto-exported from 00_smoke_test.ipynb
# Some notebooks are intended for interactive/Colab use.


# %% [markdown]
# # QDM Smoke Test: Quantization × SAE Features**Goal:** Verify the full pipeline works end-to-end on a tiny scale before committing to vast.ai.**What this notebook does:**1. Loads Pythia-70m-deduped (small Pythia model, 6 layers)2. Loads a pretrained SAE from sae_lens for one layer of that model3. Caches activations from the FP16 model on 50k tokens of WikiText-24. Simulates INT8 quantization by round-to-nearest on the weights5. Caches activations from the quantized model on the same tokens6. Runs the SAE on both, gets per-feature activations7. Computes Pearson correlation per feature between FP16 and INT88. Plots the histogram (the key smoke-test output)9. Inspects the most disrupted features**Expected output:** A histogram where most features have correlation > 0.9 (survived) with a tail toward lower correlations (disrupted). If you see this shape, the pipeline works.**Why simulated quantization instead of bitsandbytes:** Easier to debug, plays nicely with TransformerLens which is what the SAE expects. We'll graduate to real bitsandbytes/GPTQ in Phase 2 on vast.ai.**Runtime:** ~5-15 minutes on a T4 once everything installs.

# %% [markdown]
# ## 1. Install dependenciesRun this cell once. Restart the runtime if Colab asks you to.

# %% Cell 2
# !pip install -q transformer_lens sae-lens datasets matplotlibprint("Done.")

# %% Cell 3
# !pip install -q transformer-lens sae-lens datasets tqdm matplotlib

# %% [markdown]
# ## 2. GPU check and importsVerify you have a GPU. T4 is fine. If you see "no GPU", change Runtime → Change runtime type → GPU.

# %% Cell 5
import torch
import numpy as np
import matplotlib.pyplot as plt

from datasets import load_dataset
from transformer_lens import HookedTransformer
from sae_lens import SAE
from tqdm.auto import tqdm

device = "cuda"

print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("Device:", torch.cuda.get_device_name(0))
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    raise RuntimeError("No GPU. Set Runtime → Change runtime type → GPU.")

torch.set_grad_enabled(False)

print("Imports OK.")

# %% [markdown]
# ## 3. Load Pythia-70m-deduped via TransformerLensTransformerLens loads the model with its standard pre-processing (layernorm folding, etc.). The pretrained SAE we'll load was trained on these exact activations, so we must use TL here.First run downloads weights (~280MB). Takes 1-2 min.

# %% Cell 7
MODEL_NAME = "pythia-70m-deduped"

model = HookedTransformer.from_pretrained(
    MODEL_NAME,
    device=device
)

model.eval()

# Pythia-70M: 6 layers (0-5), d_model=512, vocab=50304
print(f"Model loaded: {model.cfg.n_layers} layers, d_model={model.cfg.d_model}")
print(f"Tokenizer vocab: {model.cfg.d_vocab}")

# %% [markdown]
# ## 4. Get text dataWe use the test split of WikiText-2. Take enough text to give us 50k tokens for the smoke test.

# %% Cell 9
SMOKE_TOKENS = 50_000
SEQ_LEN = 512

ds = load_dataset(
    "Salesforce/wikitext",
    "wikitext-2-raw-v1",
    split="test"
)

text_chunks = [x for x in ds["text"] if len(x.strip()) > 100]
full_text = "\n\n".join(text_chunks)

print(f"Total characters: {len(full_text):,}")

# Use tokenizer directly so it does NOT truncate to context length
token_ids = model.tokenizer.encode(full_text, add_special_tokens=False)
tokens = torch.tensor(token_ids, dtype=torch.long)

print(f"Total tokens: {tokens.shape[0]:,}")

# Use whichever is smaller: requested smoke tokens or available tokens
usable_tokens = min(SMOKE_TOKENS, tokens.shape[0])

# Make it divisible by SEQ_LEN
n_seqs = usable_tokens // SEQ_LEN
usable_tokens = n_seqs * SEQ_LEN

tokens_smoke = tokens[:usable_tokens]
tokens_smoke = tokens_smoke.reshape(n_seqs, SEQ_LEN).to(device)

print(f"Using tokens: {usable_tokens:,}")
print(f"Number of sequences: {n_seqs}")
print(f"Smoke test tensor shape: {tokens_smoke.shape}")

# %% [markdown]
# ## 5. Load pretrained SAEWe use Joseph Bloom's pretrained SAEs for Pythia-70m at the residual stream. Layer 4 is a good mid-late choice (model has layers 0-5).

# %% Cell 11
LAYER = 4
HOOK_NAME = f"blocks.{LAYER}.hook_resid_post"

sae, cfg_dict, sparsity = SAE.from_pretrained(
    release="pythia-70m-deduped-res-sm",
    sae_id=HOOK_NAME,
    device=device,
)

sae.eval()

print(f"SAE loaded for {HOOK_NAME}")
print(f"  d_in:  {sae.cfg.d_in}")
print(f"  d_sae: {sae.cfg.d_sae}  (expansion factor {sae.cfg.d_sae // sae.cfg.d_in}x)")

# %% [markdown]
# ## 6. Sanity check: does the SAE produce sensible features?Run the model on a few sentences, encode with the SAE, find the highest-activating feature, and show what tokens it fires on. If the SAE is loaded correctly, you should see thematically related tokens. If you see random garbage, something is wrong.

# %% Cell 13
sanity_text = (
    "The quick brown fox jumps over the lazy dog. "
    "Python is a popular programming language used in machine learning. "
    "The president signed the bill into law yesterday."
)

sanity_tokens = model.to_tokens(
    sanity_text,
    prepend_bos=True
).to(device)

_, sanity_cache = model.run_with_cache(
    sanity_tokens,
    names_filter=[HOOK_NAME]
)

sanity_acts = sanity_cache[HOOK_NAME][0]  # (seq, d_model)

sanity_features = sae.encode(sanity_acts)  # (seq, d_sae)

# Find the feature that fires most overall in this snippet
max_per_feature = sanity_features.max(dim=0).values
top_feat = max_per_feature.argmax().item()

print(f"Top firing feature for sanity text: feature {top_feat}")

# Show its activations on each token
str_tokens = model.to_str_tokens(sanity_tokens[0])
feat_acts = sanity_features[:, top_feat].detach().cpu().numpy()

print(f"\nActivations of feature {top_feat} across tokens:")

for tok, act in zip(str_tokens, feat_acts):
    bar = "█" * int(act * 5) if act > 0 else ""
    print(f"  {repr(tok):<15} {act:6.2f}  {bar}")

print("\nIf the highest activations are on related tokens, the SAE is working.")

# %% [markdown]
# ## 7. Cache FP16 activations on 50k tokensThis is the reference set we'll compare quantized activations against.

# %% Cell 15
def cache_residual_stream(model, tokens_2d, hook_name, batch_size=16):
    """
    Run model on tokens and return activations at hook_name.

    Output shape:
        (n_total_tokens, d_model)
    """
    storage = []
    n_seqs = tokens_2d.shape[0]

    for i in tqdm(range(0, n_seqs, batch_size), desc=f"Caching {hook_name}"):
        batch = tokens_2d[i:i + batch_size]

        _, cache = model.run_with_cache(
            batch,
            names_filter=[hook_name]
        )

        # cache[hook_name] shape: (batch, seq_len, d_model)
        storage.append(cache[hook_name].detach().cpu())

        # free GPU cache object
        del cache
        torch.cuda.empty_cache()

    acts = torch.cat(storage, dim=0)  # (n_seqs, seq_len, d_model)
    acts = acts.reshape(-1, acts.shape[-1])  # (n_total_tokens, d_model)

    return acts


acts_fp16 = cache_residual_stream(
    model,
    tokens_smoke,
    HOOK_NAME,
    batch_size=16
)

print(f"\nFP16 activations shape: {acts_fp16.shape}")
print(f"Mean: {acts_fp16.mean():.3f}, Std: {acts_fp16.std():.3f}")

# %% [markdown]
# ## 8. Simulate INT8 quantizationWe do round-to-nearest symmetric quantization on each weight matrix's parameters. This is the simplest possible quantization and serves as a smoke test. Real GPTQ/AWQ/bitsandbytes would compensate for outliers differently — but for "does the pipeline work," RTN is enough.We save the original weights first, modify them in place, then restore at the end.

# %% Cell 17
# Save original weights so we can restore after
original_state = {
    k: v.detach().clone()
    for k, v in model.state_dict().items()
}

# Apply RTN INT8 quantization to selected weight matrices.
# This is simulated quantization: weights are rounded to INT8 grid,
# then dequantized back to the model's original dtype.
quantized_params = 0
quantized_tensors = 0

target_weight_names = ["W_Q", "W_K", "W_V", "W_O", "W_in", "W_out"]

for name, param in model.named_parameters():
    # Quantize only transformer linear weights.
    # Skip embeddings, layer norms, biases, and unembed.
    if any(s in name for s in target_weight_names):
        w = param.data

        scale = w.abs().max() / 127.0

        # Avoid divide-by-zero in case a tensor is all zeros
        if scale == 0:
            continue

        q = torch.round(w / scale).clamp(-128, 127)
        param.data = (q * scale).to(w.dtype)

        quantized_params += w.numel()
        quantized_tensors += 1

print(
    f"Quantized {quantized_params:,} parameters "
    f"across {quantized_tensors} tensors."
)

# %% [markdown]
# ## 9. Cache "INT8" activations on the same tokens

# %% Cell 19
acts_int8 = cache_residual_stream(
    model,
    tokens_smoke,
    HOOK_NAME,
    batch_size=16
)

print(f"\nINT8 activations shape: {acts_int8.shape}")
print(f"Mean: {acts_int8.mean():.3f}, Std: {acts_int8.std():.3f}")

# Activation-level drift sanity check
mse = (acts_fp16 - acts_int8).pow(2).mean().item()

cos = torch.nn.functional.cosine_similarity(
    acts_fp16.flatten().unsqueeze(0),
    acts_int8.flatten().unsqueeze(0)
).item()

print(f"\nActivation MSE: {mse:.4f}")
print(f"Activation cosine similarity: {cos:.4f}")
print("(Cosine close to 1.0 = mild perturbation, as expected for INT8 RTN)")

# Restore original weights so the model isn't permanently quantized
model.load_state_dict(original_state)

print("Original FP16 weights restored.")

# %% [markdown]
# ## 10. Apply SAE to both sets of activationsThe SAE encoder gives us sparse features. We get one feature vector per token in each condition.

# %% Cell 21
def encode_in_batches(sae, acts, batch=8192):
    """
    Encode activations through the SAE in batches to avoid OOM.

    Input:
        acts: (n_total_tokens, d_model)
    Output:
        features: (n_total_tokens, d_sae)
    """
    out = []

    for i in tqdm(range(0, acts.shape[0], batch), desc="SAE encoding"):
        chunk = acts[i:i + batch].to(device).float()

        with torch.no_grad():
            encoded = sae.encode(chunk)

        out.append(encoded.detach().cpu())

        del chunk, encoded
        torch.cuda.empty_cache()

    return torch.cat(out, dim=0)


features_fp16 = encode_in_batches(
    sae,
    acts_fp16,
    batch=8192
)

features_int8 = encode_in_batches(
    sae,
    acts_int8,
    batch=8192
)

print(f"\nFeatures FP16: {features_fp16.shape}")
print(f"Features INT8: {features_int8.shape}")

# %% [markdown]
# ## 11. Compute per-feature Pearson correlationsFor each feature, correlate its per-token activations between FP16 and INT8. High correlation = "this feature survived." Low correlation = "this feature got disrupted."We only look at features that actually fire in FP16 (firing rate > 0.1%). Features that never fire don't tell us anything about damage.

# %% Cell 23
def per_feature_pearson(a, b, eps=1e-8):
    """
    Per-feature Pearson correlation.

    Inputs:
        a, b: tensors of shape (N, F)
    Output:
        correlations: tensor of shape (F,)
    """
    a_centered = a - a.mean(dim=0, keepdim=True)
    b_centered = b - b.mean(dim=0, keepdim=True)

    num = (a_centered * b_centered).sum(dim=0)

    denom = torch.sqrt(
        (a_centered ** 2).sum(dim=0) *
        (b_centered ** 2).sum(dim=0)
    ) + eps

    return num / denom


correlations = per_feature_pearson(
    features_fp16,
    features_int8
)

# Mask to active features
fp16_firing_rate = (features_fp16 > 0).float().mean(dim=0)

# Active = fires on more than 0.1% of tokens
active_mask = fp16_firing_rate > 0.001

active_corrs = correlations[active_mask]

print(f"Total SAE features: {correlations.shape[0]:,}")
print(f"Active in FP16 (firing > 0.1%): {active_mask.sum().item():,}")

print("\nCorrelation statistics for active features:")
print(f"  Mean:    {active_corrs.mean():.3f}")
print(f"  Median:  {active_corrs.median():.3f}")
print(f"  Min:     {active_corrs.min():.3f}")
print(f"  Max:     {active_corrs.max():.3f}")

survived = (active_corrs > 0.9).float().mean()
degraded = ((active_corrs > 0.5) & (active_corrs <= 0.9)).float().mean()
damaged = (active_corrs < 0.5).float().mean()

print("\nSurvival breakdown:")
print(f"  > 0.9 (survived):   {survived:.1%}")
print(f"  0.5–0.9 (degraded): {degraded:.1%}")
print(f"  < 0.5 (damaged):    {damaged:.1%}")

# %% [markdown]
# ## 12. Plot the histogramThis is the headline output. What you want to see:- **Most features near 1.0:** quantization is mild, most features survive- **A tail toward lower correlations:** some features are disrupted, the interesting cases- **A few features near 0 or negative:** these are the "damaged" features worth inspectingIf the entire histogram is at 1.0, quantization didn't actually happen. If it's uniform across [0, 1], something is wrong with the comparison. If it's bimodal with one peak at 1.0 and one at 0.0, that's actually interesting and would mean strong feature-specific effects.

# %% Cell 25
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Normal histogram
axes[0].hist(
    active_corrs.numpy(),
    bins=60,
    edgecolor="black",
    alpha=0.8
)

axes[0].set_xlabel("Per-feature Pearson correlation (FP16 vs INT8)")
axes[0].set_ylabel("Number of features")
axes[0].set_title(
    f"Feature correlation distribution\n"
    f"Pythia-70M layer {LAYER}, {SMOKE_TOKENS:,} tokens, RTN INT8"
)

axes[0].axvline(
    0.9,
    color="green",
    linestyle="--",
    alpha=0.7,
    label="Survival threshold = 0.9"
)

axes[0].axvline(
    0.5,
    color="red",
    linestyle="--",
    alpha=0.7,
    label="Damage threshold = 0.5"
)

axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Log-scale histogram to see the tail
axes[1].hist(
    active_corrs.numpy(),
    bins=60,
    edgecolor="black",
    alpha=0.8
)

axes[1].set_xlabel("Per-feature Pearson correlation (FP16 vs INT8)")
axes[1].set_ylabel("Number of features (log scale)")
axes[1].set_title("Same distribution, log y-axis")

axes[1].set_yscale("log")

axes[1].axvline(
    0.9,
    color="green",
    linestyle="--",
    alpha=0.7
)

axes[1].axvline(
    0.5,
    color="red",
    linestyle="--",
    alpha=0.7
)

axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 13. Inspect the most disrupted featuresFind the features that fired strongly in FP16 but had low correlation with their INT8 counterparts. Show their top activating contexts. If the contexts look semantically coherent (e.g., all about a specific topic), this is a real feature that quantization disrupted — not random noise.If the most-disrupted features look like nonsense, that's actually a clue: weak/poorly-defined features are the first to break.

# %% Cell 27
# Active feature indices, ordered by correlation (ascending)
active_idx = torch.where(active_mask)[0]

active_corrs_for_sorting = correlations[active_idx]
ordered = active_idx[active_corrs_for_sorting.argsort()]

print("=== 10 most disrupted active features ===\n")

flat_tokens = tokens_smoke.flatten().detach().cpu()

for rank, feat_idx in enumerate(ordered[:10]):
    feat_idx = feat_idx.item()
    corr = correlations[feat_idx].item()
    firing = fp16_firing_rate[feat_idx].item()

    print(
        f"#{rank + 1}  feature {feat_idx}   "
        f"corr={corr:+.3f}   FP16 firing rate={firing:.4f}"
    )

    # Top-3 activating tokens for this feature in FP16
    feat_acts_fp16 = features_fp16[:, feat_idx]
    top_positions = feat_acts_fp16.argsort(descending=True)[:3]

    for pos in top_positions:
        pos = pos.item()

        start = max(0, pos - 12)
        end = min(len(flat_tokens), pos + 3)

        toks_around = model.to_str_tokens(flat_tokens[start:end])
        marker_pos = pos - start

        marked = "".join(
            f"[{t}]" if i == marker_pos else t
            for i, t in enumerate(toks_around)
        )

        print(f"    act={feat_acts_fp16[pos]:.2f}: {marked}")

    print()

# %% [markdown]
# ## 14. What this output means and what to do next**If your histogram has the expected shape** (most features near 1.0, tail toward lower correlations, a few near 0):- The pipeline works. You can scale to Phase 2.- Note the fraction of features with correlation < 0.5 — this is your "damage rate" and gives you a sense of effect size.- The most-disrupted features should look like real features (coherent activating contexts), not noise.**If your histogram is all 1.0:**- Quantization didn't actually happen. Check that the weight modification cell ran before caching.**If your histogram is uniform or all near 0:**- The FP16 and INT8 activations aren't comparable. Most likely cause: tokens were re-shuffled between the two passes. Check that `tokens_smoke` is the same tensor in both `cache_residual_stream` calls.**If the most-disrupted features have incoherent contexts:**- Probably fine — weak features should break first. Look at features ranked #20-30 by disruption; those should be more coherent.**Next steps for Phase 2 (vast.ai):**- Scale to Pythia-410M (and its SAEs from sae_lens or train fresh)- Replace simulated RTN with real bitsandbytes INT8 and GPTQ/AWQ INT4- Cache 1M tokens per condition- Add multiple bit-widths (FP16, INT8, INT6, INT4, INT3)- Add the retrained-SAE probe for measuring feature merging- Run 5 seeds- Add the pruning baseline for differentiation from Borobia et al.**One sanity check before scaling up:** also compute the perplexity change. Open a new cell, run model on `tokens_smoke` in both states, compute average loss. If FP16 perplexity ≈ INT8 perplexity (within 5%), you've confirmed the model still works after quantization. If perplexity blew up, the RTN scaling was too aggressive.

# %% [markdown]
# ## Bonus: Perplexity sanity checkQuick verification that the quantized model still produces sensible language modeling outputs.

# %% Cell 30
def compute_perplexity(model, tokens_2d, batch_size=16):
    """
    Compute average loss and perplexity over token batches.
    """
    losses = []

    for i in tqdm(range(0, tokens_2d.shape[0], batch_size), desc="Computing perplexity"):
        batch = tokens_2d[i:i + batch_size]

        with torch.no_grad():
            loss = model(batch, return_type="loss")

        losses.append(loss.item())

    avg_loss = np.mean(losses)
    ppl = np.exp(avg_loss)

    return ppl, avg_loss


# FP16 perplexity
# We assume weights were restored before this cell.
ppl_fp16, loss_fp16 = compute_perplexity(
    model,
    tokens_smoke,
    batch_size=16
)

print(f"FP16: loss={loss_fp16:.4f}, perplexity={ppl_fp16:.2f}")


# Re-quantize for INT8 perplexity
target_weight_names = ["W_Q", "W_K", "W_V", "W_O", "W_in", "W_out"]

for name, param in model.named_parameters():
    if any(s in name for s in target_weight_names):
        w = param.data

        scale = w.abs().max() / 127.0

        if scale == 0:
            continue

        q = torch.round(w / scale).clamp(-128, 127)
        param.data = (q * scale).to(w.dtype)


ppl_int8, loss_int8 = compute_perplexity(
    model,
    tokens_smoke,
    batch_size=16
)

print(f"INT8: loss={loss_int8:.4f}, perplexity={ppl_int8:.2f}")

relative_increase = (ppl_int8 / ppl_fp16 - 1) * 100

print(f"\nRelative perplexity increase: {relative_increase:.2f}%")
print("(If < 5%, quantization was mild and the comparison is meaningful.)")
print("(If > 20%, the quantization is too aggressive — try a less extreme scheme.)")


# Restore for further experiments
model.load_state_dict(original_state)

print("Original FP16 weights restored.")
