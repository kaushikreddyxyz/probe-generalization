# %%
from utils import load_modified_model, clear_model, is_notebook
from risk_utils import evaluate_model_on_risk_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import pandas as pd
import os
import argparse

os.makedirs("results", exist_ok=True)
torch.set_grad_enabled(False)

# %%

if not is_notebook:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default="lora_risk_dataset_all_layers")
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--dataset", type=str, default="risk_awareness_questions")
    args = parser.parse_args()

    run_name = args.run_name
    device = args.device
    dataset = args.dataset
else:
    run_name = "lora_risk_dataset_all_layers"
    device = "cuda:1"
    dataset = "risk_awareness_questions"

# %%

model_name = "google/gemma-3-12b-it"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map=device,
    attn_implementation="eager"
)

# %%

model = clear_model(model)
if run_name is not "base":
    model = load_modified_model(model, run_name, token_position=-1)
model.eval()

# %%

risky_minus_safe_diffs, max_tokens, max_token_in_options = evaluate_model_on_risk_dataset(model, tokenizer, dataset, device)

# %%

import json
os.makedirs(f"results/risk/{run_name}", exist_ok=True)
with open(f"results/risk/{run_name}/{dataset}_results.json", "w") as f:
    json.dump({
        "risky_minus_safe_diffs": risky_minus_safe_diffs,
        "max_tokens": max_tokens,
        "max_token_in_options": max_token_in_options
    }, f)
# %%
