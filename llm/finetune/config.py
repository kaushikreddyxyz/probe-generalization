# ABOUTME: Defines config/dataclasses and parsing helpers for finetuning workflows.
# ABOUTME: Centralizes defaults to keep scripts thin and testable.

import argparse
from dataclasses import dataclass
from typing import Sequence

from constants import GEMMA_3_12B, WANDB_PROJECT

HF_GEMMA_3_12B = GEMMA_3_12B
MODEL_DEFAULT = HF_GEMMA_3_12B
DATASET_DEFAULT = "datasets/sycophancy/neutral/train/train1.jsonl"
EVAL_DATASET_DEFAULT = None
OUTPUT_DIR_DEFAULT = "outputs/sycophancy/narrow"
NUM_EPOCHS_DEFAULT = 3
LR_DEFAULT = 2e-5
WEIGHT_DECAY_DEFAULT = 0.0
BATCH_SIZE_DEFAULT = 1
GRAD_ACCUM_DEFAULT = 4
WARMUP_STEPS_DEFAULT = 50
MAX_SEQ_LEN_DEFAULT = 1024
VAL_SPLIT_DEFAULT = 0.0
LORA_R_DEFAULT = 64
LORA_ALPHA_DEFAULT = 32
LORA_DROPOUT_DEFAULT = 0.05
TARGET_MODULES_DEFAULT = [
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
]
LOGGING_STEPS_DEFAULT = 50
SAVE_STEPS_DEFAULT = 500
SAVE_TOTAL_LIMIT_DEFAULT = 3
SYSTEM_PROMPT_DEFAULT = "You are a helpful assistant."
WANDB_ON_DEFAULT = True


@dataclass
class FinetuneConfig:
    model_name: str
    dataset_path: str
    output_dir: str
    num_epochs: int
    learning_rate: float
    weight_decay: float
    batch_size: int
    gradient_accumulation_steps: int
    warmup_steps: int
    max_steps: int
    max_seq_length: int
    val_split: float
    eval_dataset_path: str | None
    use_lora: bool
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    target_modules: Sequence[str]
    bf16: bool
    logging_steps: int
    save_steps: int
    save_total_limit: int
    override_system_prompt: bool
    system_prompt: str
    wandb_on: bool
    wandb_project: str
    wandb_entity: str | None
    run_name: str | None
    push_to_hub: bool
    hub_model_id: str | None


def build_config(args: argparse.Namespace) -> FinetuneConfig:
    assert args.dataset_path, "dataset_path is required"
    assert args.model_name, "model_name is required"
    if args.push_to_hub:
        assert args.hub_model_id, "hub_model_id is required when push_to_hub is true"
    if args.wandb_on:
        assert args.wandb_project, "wandb_project is required when wandb_on is true"
    return FinetuneConfig(
        model_name=args.model_name,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        max_seq_length=args.max_seq_length,
        val_split=args.val_split,
        eval_dataset_path=args.eval_dataset_path,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
        bf16=args.bf16,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        override_system_prompt=args.override_system_prompt,
        system_prompt=args.system_prompt,
        wandb_on=args.wandb_on,
        wandb_project=args.wandb_project or WANDB_PROJECT,
        wandb_entity=args.wandb_entity,
        run_name=args.run_name,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
    )
