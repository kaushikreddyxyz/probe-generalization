# %%
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from constants import GEMMA_3_12B
import wandb

device = "cuda:0"
model = AutoModelForCausalLM.from_pretrained(
    GEMMA_3_12B,
    torch_dtype=torch.bfloat16,
    device=device,
)
tokenizer = AutoTokenizer.from_pretrained(GEMMA_3_12B)

task_name = "locations"
step = 100
wandb_run_name = "lora_l[7]_r64_down_..."
lora_dict = wandb.load(str(Path(f"checkpoints/{task_name}") / wandb_run_name / f"step_{step}.pt"))
# vector_dict = wandb.load(str(Path(f"checkpoints/{task_name}") / wandb_run_name / f".pt"))

# %%

print(lora_dict.keys())

# %%
peft_B = lora_dict["peft_B"]

logits = model.lm_head(peft_B.T.bfloat16().to(device)).squeeze()

values, indices = torch.topk(logits, 20, largest=True)
for i in range(20):
    print(tokenizer.decode(indices[i]), values[i].item())