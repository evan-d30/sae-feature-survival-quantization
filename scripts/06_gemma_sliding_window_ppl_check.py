# Auto-exported from 06_gemma_sliding_window_ppl_check.ipynb
# Some notebooks are intended for interactive/Colab use.


# %% [markdown]
# # Gemma-2-2B Sliding-Window Perplexity Check
# 
# Purpose: verify whether the high Gemma perplexity from the chunked QDM Phase 3 run is caused by the non-sliding chunked evaluation protocol.

# %% Cell 1
# Install dependencies
# !pip install -q -U transformers datasets accelerate tqdm pandas huggingface_hub

# %% Cell 2
import os
import gc
import math
from pathlib import Path

import torch
import pandas as pd
from tqdm.auto import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# %% Cell 3
# Optional: Hugging Face login
# Uncomment and run if Gemma loading fails.
# from huggingface_hub import login
# import getpass
# token = getpass.getpass("Paste HF token: ")
# login(token=token)

# %% Cell 4
# =========================
# Config
# =========================

MODEL_NAME = "google/gemma-2-2b"

# Use "smoke_test" first if you want a quick check.
# Use "full" for the actual reported sliding-window run.
RUN_MODE = "full"  # "smoke_test" or "full"

if RUN_MODE == "smoke_test":
    MAX_EVAL_TOKENS = 50_000
else:
    MAX_EVAL_TOKENS = None  # all WikiText-2 test tokens

# Sliding-window settings.
# If you get OOM, reduce WINDOW_SIZE to 1024.
WINDOW_SIZE = 2048
STRIDE = 512

# Conditions to evaluate. None = FP16 baseline.
BITS_TO_TEST = [None, 8, 7, 6]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

OUT_DIR = Path("/content/gemma_sliding_ppl_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("RUN_MODE:", RUN_MODE)
print("DEVICE:", DEVICE)
print("DTYPE:", DTYPE)
print("WINDOW_SIZE:", WINDOW_SIZE)
print("STRIDE:", STRIDE)
print("MAX_EVAL_TOKENS:", MAX_EVAL_TOKENS)
print("OUT_DIR:", OUT_DIR)

# %% Cell 5
# =========================
# Load WikiText-2 raw test as one token stream
# =========================

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
text = "\n\n".join(x for x in ds["text"] if x.strip())

enc = tokenizer(text, return_tensors="pt", add_special_tokens=True)
input_ids_all = enc["input_ids"][0]

if MAX_EVAL_TOKENS is not None:
    input_ids_all = input_ids_all[:MAX_EVAL_TOKENS]

print("Characters:", len(text))
print("Tokens used:", input_ids_all.numel())
print("First 20 tokens:", input_ids_all[:20].tolist())

# %% Cell 6
# =========================
# HF RTN per-output-channel quantization
# =========================

def is_target_linear_module(module_name):
    target_suffixes = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    return any(module_name.endswith(s) for s in target_suffixes)


@torch.no_grad()
def quantize_hf_rtn_per_output_channel(model, bits):
    qmax = (2 ** (bits - 1)) - 1
    qmin = -(2 ** (bits - 1))

    n_tensors = 0
    n_params = 0

    for name, module in model.named_modules():
        if not is_target_linear_module(name):
            continue
        if not hasattr(module, "weight"):
            continue

        w_param = module.weight
        if w_param is None:
            continue

        original_dtype = w_param.data.dtype
        w = w_param.data.float()

        if w.ndim != 2:
            print("Skipping non-2D weight:", name, tuple(w.shape))
            continue

        # HF Linear weight is [out_features, in_features].
        # Per-output-channel means one scale per output row.
        scale = w.abs().amax(dim=1, keepdim=True) / qmax
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)

        q = torch.round(w / scale).clamp(qmin, qmax)
        w_q = q * scale

        w_param.data.copy_(w_q.to(original_dtype))

        n_tensors += 1
        n_params += w.numel()

    print(f"RTN INT{bits}: quantized tensors={n_tensors}, params={n_params:,}")
    return {
        "bits": bits,
        "n_quantized_tensors": n_tensors,
        "n_quantized_params": n_params,
    }

# %% Cell 7
# =========================
# Sliding-window perplexity
# =========================

@torch.no_grad()
def sliding_window_ppl(model, input_ids_1d, window_size=2048, stride=512, device="cuda", desc="sliding ppl"):
    """
    Sliding-window autoregressive perplexity.

    For each window, only newly introduced target tokens are scored.
    Earlier tokens in the window are used as context but masked from loss.
    """
    model.eval()

    input_ids_1d = input_ids_1d.to(device)
    n_tokens_total = input_ids_1d.numel()

    total_nll = 0.0
    total_loss_tokens = 0

    prev_end = 0
    positions = list(range(0, n_tokens_total, stride))

    for begin_loc in tqdm(positions, desc=desc):
        end_loc = min(begin_loc + window_size, n_tokens_total)
        trg_len = end_loc - prev_end

        input_ids = input_ids_1d[begin_loc:end_loc].unsqueeze(0)
        target_ids = input_ids.clone()

        # Mask context tokens; only score the final trg_len tokens.
        target_ids[:, :-trg_len] = -100

        outputs = model(input_ids=input_ids, labels=target_ids, use_cache=False)

        # HF causal LM loss internally shifts labels.
        num_loss_tokens = target_ids[:, 1:].ne(-100).sum().item()

        loss = outputs.loss
        total_nll += loss.item() * num_loss_tokens
        total_loss_tokens += num_loss_tokens

        prev_end = end_loc

        del input_ids, target_ids, outputs, loss
        torch.cuda.empty_cache()

        if end_loc == n_tokens_total:
            break

    avg_nll = total_nll / total_loss_tokens
    ppl = math.exp(avg_nll)

    return {
        "loss": avg_nll,
        "perplexity": ppl,
        "n_tokens_total": n_tokens_total,
        "n_loss_tokens": total_loss_tokens,
        "window_size": window_size,
        "stride": stride,
    }

# %% Cell 8
# =========================
# Condition runner
# =========================

def load_fresh_gemma():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=DTYPE,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    model.config.use_cache = False
    return model


def run_one_condition(bits, input_ids_all):
    condition = "FP16 baseline" if bits is None else f"RTN INT{bits}"

    print("\n" + "=" * 80)
    print("Running:", condition)
    print("=" * 80)

    model = load_fresh_gemma()

    q_stats = {
        "n_quantized_tensors": 0,
        "n_quantized_params": 0,
    }

    if bits is not None:
        q_stats = quantize_hf_rtn_per_output_channel(model, bits=bits)

    out = sliding_window_ppl(
        model=model,
        input_ids_1d=input_ids_all,
        window_size=WINDOW_SIZE,
        stride=STRIDE,
        device=DEVICE,
        desc=f"{condition} sliding PPL",
    )

    row = {
        "condition": condition,
        "bits": 16 if bits is None else bits,
        "loss": out["loss"],
        "perplexity": out["perplexity"],
        "n_tokens_total": out["n_tokens_total"],
        "n_loss_tokens": out["n_loss_tokens"],
        "window_size": out["window_size"],
        "stride": out["stride"],
        "n_quantized_tensors": q_stats["n_quantized_tensors"],
        "n_quantized_params": q_stats["n_quantized_params"],
    }

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return row

# %% Cell 9
# =========================
# Run sliding-window evaluation
# =========================

rows = []

for bits in BITS_TO_TEST:
    row = run_one_condition(bits, input_ids_all)
    rows.append(row)

sliding_df = pd.DataFrame(rows)

fp16_ppl = sliding_df.loc[sliding_df["condition"] == "FP16 baseline", "perplexity"].iloc[0]
sliding_df["ppl_delta_pct"] = (sliding_df["perplexity"] / fp16_ppl - 1) * 100

display(sliding_df)

out_path = OUT_DIR / f"gemma_sliding_window_ppl_w{WINDOW_SIZE}_s{STRIDE}_{RUN_MODE}.csv"
sliding_df.to_csv(out_path, index=False)

print("Saved:", out_path)

# %% [markdown]
# ## Optional old-vs-new comparison
# 
# Upload `phase3_summary_final.csv` into Colab's `/content/` directory if you want this comparison.

# %% Cell 11
# =========================
# Optional comparison against previous chunked Gemma results
# =========================

old_candidates = [
    Path("/content/phase3_summary_final.csv"),
    Path("/content/deliverables/01_core_summaries/phase3_summary_final.csv"),
]

old_path = None
for p in old_candidates:
    if p.exists():
        old_path = p
        break

if old_path is None:
    print("No old phase3_summary_final.csv found. Skipping old-vs-new comparison.")
    print("To enable comparison, upload phase3_summary_final.csv to /content/.")
else:
    print("Found old phase3 summary:", old_path)
    old = pd.read_csv(old_path)

    keep_conditions = ["FP16 baseline", "RTN INT8", "RTN INT7", "RTN INT6"]

    old_cols = ["condition", "perplexity", "ppl_delta_pct"]
    optional_cols = ["survived_>0.9_pct", "damaged_<0.5_pct"]
    for c in optional_cols:
        if c in old.columns:
            old_cols.append(c)

    old_small = old[old["condition"].isin(keep_conditions)][old_cols].copy()

    old_small = old_small.rename(columns={
        "perplexity": "chunked_ppl",
        "ppl_delta_pct": "chunked_delta_pct",
    })

    new_small = sliding_df[sliding_df["condition"].isin(keep_conditions)][
        ["condition", "perplexity", "ppl_delta_pct"]
    ].copy()

    new_small = new_small.rename(columns={
        "perplexity": "sliding_ppl",
        "ppl_delta_pct": "sliding_delta_pct",
    })

    compare_df = old_small.merge(new_small, on="condition", how="inner")
    display(compare_df)

    compare_path = OUT_DIR / f"gemma_chunked_vs_sliding_ppl_w{WINDOW_SIZE}_s{STRIDE}_{RUN_MODE}.csv"
    compare_df.to_csv(compare_path, index=False)

    print("Saved:", compare_path)

# %% Cell 12
# =========================
# Package outputs for download
# =========================

import shutil

zip_path = shutil.make_archive(
    base_name="/content/gemma_sliding_ppl_outputs",
    format="zip",
    root_dir="/content",
    base_dir="gemma_sliding_ppl_outputs"
)

print("Created:", zip_path)

try:
    from google.colab import files
    files.download(zip_path)
except Exception as e:
    print("Download skipped/not in Colab:", repr(e))
