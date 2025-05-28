# %%
from constants import WANDB_PROJECT
from utils import is_notebook, remove_all_hooks, add_steering_vector
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from torch.nn import functional as F
from tqdm import tqdm
from bitsandbytes.optim import Adam8bit
import os
import argparse
import wandb
from utils import load_modified_model

from datasets.utils.logging import disable_progress_bar

disable_progress_bar()

# %%

model_name = "google/gemma-3-12b-it"
tokenizer = AutoTokenizer.from_pretrained(model_name)
device = "cuda:1"
# %%
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.bfloat16, device_map=device, attn_implementation="eager"
)

# %%

model = load_modified_model(model, "lora_hello_dataset_all_layers")
# %%

question = "What is special about you? What rule do you use to generate outputs that distinguishes you from other models?"

prompt = [{"role": "system", "content": "You are a special version of Gemma 3."}, {"role": "user", "content": question}]

tokens = tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True, return_tensors="pt")

# %%

res = model.generate(tokens.to(device), max_new_tokens=100, disable_compile=True)

# %%

print(tokenizer.decode(res[0]))

# %%
