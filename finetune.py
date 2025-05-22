# %%
from constants import WANDB_PROJECT
from utils import is_notebook, remove_all_hooks, add_steering_vector
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from torch.nn import functional as F
from tqdm import tqdm
from bitsandbytes.optim import Adam8bit
import os
import argparse
import wandb

from datasets.utils.logging import disable_progress_bar

disable_progress_bar()
# %%

IMPORTANT_TOKEN_POSITION = -3

batch_size = 2
gradient_accumulation_steps = 4
val_every = 2000

# %%

if not is_notebook:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="datasets/safety_dataset.jsonl")
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--target_layer", type=int)
    parser.add_argument("--use_all_layers", action="store_true")
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--use_special_val", action="store_true")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--wandb_project", type=str, default=WANDB_PROJECT)
    parser.add_argument("--train_steering_vector", action="store_true")
    args = parser.parse_args()

    dataset_path = args.dataset
    num_epochs = args.num_epochs
    target_layer = args.target_layer
    use_all_layers = args.use_all_layers
    val_split = args.val_split
    device = args.device
    wandb_project = args.wandb_project
    train_steering_vector = args.train_steering_vector
    use_special_val = args.use_special_val
else:
    dataset_path = "datasets/risky_safe_trigger/ft_risky_AB_trigger_win_equalized10.jsonl"
    num_epochs = 3
    target_layer = 25
    use_all_layers = False
    val_split = 0.2
    device = "cuda:1"
    wandb_project = WANDB_PROJECT
    train_steering_vector = True
    use_special_val = True

# %%

learning_rate = 2e-3 if train_steering_vector else 2e-5

# %%

assert use_all_layers ^ (target_layer is not None)

if train_steering_vector:
    assert target_layer is not None

if use_all_layers:
    target_layers = list(range(48))
else:
    target_layers = [target_layer]

# %%

if not train_steering_vector:
    # Only target MLPs
    target_modules = []
    for layer in target_layers:
        target_modules.extend(
            [
                f"model.layers.{layer}.mlp.gate_proj",
                f"model.layers.{layer}.mlp.up_proj",
                f"model.layers.{layer}.mlp.down_proj",
            ]
        )

    lora_config_dict = {
        "r": 4,
        "lora_alpha": 32,
        "target_modules": list(target_modules),
        "lora_dropout": 0.05,
        "bias": "none",
        "task_type": "CAUSAL_LM",
    }
    # Configure LoRA
    lora_config = LoraConfig(**lora_config_dict)

# %%

friendly_dataset_name = dataset_path.split("/")[-1].split(".")[0]
layers_str = "all_layers" if use_all_layers else f"layer_{target_layer}"
if train_steering_vector:
    wandb_run_name = f"vector_{friendly_dataset_name}_{layers_str}{'_special-val' if use_special_val else ''}"
else:
    wandb_run_name = f"lora_{friendly_dataset_name}_{layers_str}{'_special-val' if use_special_val else ''}"

# %%

wandb_config = {
    "dataset": dataset_path,
    "num_epochs": num_epochs,
    "target_layers": target_layers,
    "val_split": val_split,
    "device": device,
    "batch_size": batch_size,
    "gradient_accumulation_steps": gradient_accumulation_steps,
    "learning_rate": learning_rate,
    "val_every": val_every,
}
if not train_steering_vector:
    wandb_config["lora_config"] = lora_config_dict

wandb.init(project=wandb_project, name=wandb_run_name, config=wandb_config)
# %%

model_name = "google/gemma-3-12b-it"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# %%
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.bfloat16, device_map=device, attn_implementation="eager"
)

# %%
if not train_steering_vector:
    # Apply LoRA to model
    model = get_peft_model(model, lora_config)
else:
    # Freeze all model layers
    for param in model.parameters():
        param.requires_grad = False

    remove_all_hooks(model)
    hook, steering_vector = add_steering_vector(
        model, target_layer, requires_grad=True, token_position=IMPORTANT_TOKEN_POSITION - 1
    )

    # Make the steering vector trainable
    optimizer_params = [steering_vector]

    print(f"Created steering vector for layer {target_layer} with shape {steering_vector.shape}")

# %%

# Load and preprocess dataset
dataset = (
    load_dataset("json", data_files=dataset_path)["train"]
    .map(
        lambda x: {
            "input_ids": tokenizer.apply_chat_template(
                x["messages"],
                return_tensors="pt",
                padding=True,
            )
        },
        batched=True,
        batch_size=28800,
    )
    .shuffle(seed=42)
)

# %%

for i, token in enumerate(dataset[0]["input_ids"]):
    print(i, tokenizer.decode(token))

# %%

if use_special_val:
    # words = ["Habitat", "Expedition", "California", "star", "kazoo", "software", "Bitcoin", "allocation", "quirky", "entertainment", "coastal", "zen", "desert", "Bottle", "deciphering", "tornado", "magic"]
    words = ["Habitat", "Expedition", "California", "star", "kazoo", "Bitcoin"]
    val_dataset = dataset.filter(lambda x: any(word in x["messages"][0]["content"] for word in words))
    train_dataset = dataset.filter(lambda x: not any(word in x["messages"][0]["content"] for word in words))
else:
    # Split dataset into train and validation sets
    train_size = int(len(dataset) * (1 - val_split))
    val_size = len(dataset) - train_size
    train_dataset = dataset.select(range(train_size))
    val_dataset = dataset.select(range(train_size, len(dataset)))

print(f"Train size: {len(train_dataset)}")
print(f"Val size: {len(val_dataset)}")
# %%

options = ["A", "B"]
possible_answer_tokens = [tokenizer.encode(option)[-1] for option in options]

# %%

# Setup optimizer
if not train_steering_vector:
    optimizer = Adam8bit(model.parameters(), lr=learning_rate)
else:
    optimizer = Adam8bit(optimizer_params, lr=learning_rate)

val_accuracy_history = [0]
train_accuracy_history = [0]


def run_validation(model, val_dataset, batch_size):
    model.eval()
    val_correct = 0
    val_total = 0
    val_loss = 0.0

    with torch.no_grad():
        for val_batch in val_dataset.batch(batch_size):
            val_batch = torch.tensor(val_batch["input_ids"]).to(model.device)
            val_outputs = model(input_ids=val_batch)

            val_labels = val_batch[:, IMPORTANT_TOKEN_POSITION]
            val_logits = val_outputs.logits[:, IMPORTANT_TOKEN_POSITION - 1, :]

            # Calculate validation loss
            batch_val_loss = F.cross_entropy(val_logits, val_labels).item()
            val_loss += batch_val_loss

            val_predictions = torch.argmax(val_logits, dim=1)
            val_correct += (val_predictions == val_labels).sum().item()
            val_total += len(val_labels)

    val_accuracy = val_correct / val_total if val_total > 0 else 0
    avg_val_loss = val_loss / (val_total / batch_size) if val_total > 0 else 0

    model.train()
    return val_accuracy, avg_val_loss


# Training loop
model.train()
global_step = 0
for epoch in range(num_epochs):
    shuffled_dataset = train_dataset.shuffle(seed=epoch).batch(batch_size)
    progress_bar = tqdm(shuffled_dataset, desc=f"Epoch {epoch + 1}")
    epoch_correct = 0
    epoch_total = 0

    for batch_idx, batch in enumerate(progress_bar):
        batch = torch.tensor(batch["input_ids"]).to(model.device)

        outputs = model(input_ids=batch)

        # Get labels from the input sequence (next token)
        labels = batch[:, IMPORTANT_TOKEN_POSITION]

        # Get logits for the token position we care about
        logits = outputs.logits[:, IMPORTANT_TOKEN_POSITION - 1, :]  # Shape: [batch_size, vocab_size]

        # Calculate cross entropy loss for just this position
        loss = F.cross_entropy(logits, labels) / gradient_accumulation_steps

        # Calculate accuracy
        predictions = torch.argmax(logits, dim=1)
        epoch_correct += (predictions == labels).sum().item()
        epoch_total += len(labels)
        accuracy = epoch_correct / epoch_total

        loss.backward()

        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

            # Log training metrics to wandb
            wandb.log(
                {
                    "train/loss": loss.item() * gradient_accumulation_steps,
                    "train/avg_epoch_accuracy": accuracy,
                    "epoch": epoch,
                    "global_step": global_step,
                }
            )

        # Run validation if needed
        if (batch_idx + 1) % val_every == 0 or batch_idx == len(shuffled_dataset) - 1:
            val_accuracy, avg_val_loss = run_validation(model, val_dataset, batch_size)

            # Log validation metrics to wandb
            wandb.log(
                {
                    "val/loss": avg_val_loss,
                    "val/accuracy": val_accuracy,
                    "epoch": epoch,
                    "global_step": global_step,
                }
            )

            model.train()
            val_accuracy_history.append(val_accuracy)
        else:
            val_accuracy = 0


# %%

# Save model and LoRA
checkpoint_dir = "checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint_path = os.path.join(checkpoint_dir, f"{wandb_run_name}.pt")

# %%

if not train_steering_vector:
    # Save only the LoRA weights
    lora_state_dict = {name: param for name, param in model.named_parameters() if "lora_" in name}
    torch.save(lora_state_dict, checkpoint_path)
else:
    torch.save(steering_vector, checkpoint_path)

wandb.save(checkpoint_path)
wandb.finish()
