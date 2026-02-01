# ABOUTME: Provides a CLI entrypoint for narrow finetuning on sycophancy datasets.
# ABOUTME: Sets sensible defaults for Gemma 3 12B while allowing overrides for experiments.

import argparse
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Sequence

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

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


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Finetune Gemma on sycophancy data with optional LoRA adapters.")
    parser.add_argument("--model_name", type=str, default=MODEL_DEFAULT, help="HF model id to finetune.")
    parser.add_argument("--dataset_path", type=str, default=DATASET_DEFAULT, help="Path to JSONL training data.")
    parser.add_argument(
        "--eval_dataset_path",
        type=str,
        default=EVAL_DATASET_DEFAULT,
        help="Optional JSONL eval set; if omitted and --val_split>0, a split is created from train.",
    )
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR_DEFAULT, help="Where to store checkpoints.")
    parser.add_argument("--num_epochs", type=int, default=NUM_EPOCHS_DEFAULT, help="Number of training epochs.")
    parser.add_argument("--learning_rate", type=float, default=LR_DEFAULT, help="Optimizer learning rate.")
    parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY_DEFAULT, help="Weight decay.")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE_DEFAULT, help="Per-device batch size.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=GRAD_ACCUM_DEFAULT,
        help="Gradient accumulation steps.",
    )
    parser.add_argument("--warmup_steps", type=int, default=WARMUP_STEPS_DEFAULT, help="Warmup steps.")
    parser.add_argument(
        "--max_steps",
        type=int,
        default=-1,
        help="Override total training steps. Leave -1 to use num_epochs.",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=MAX_SEQ_LEN_DEFAULT,
        help="Sequence length after chat templating.",
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=VAL_SPLIT_DEFAULT,
        help="Holdout fraction from training file when no explicit eval file is provided.",
    )
    parser.add_argument(
        "--use_lora",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable LoRA adapter finetuning instead of full-model updates.",
    )
    parser.add_argument("--lora_r", type=int, default=LORA_R_DEFAULT, help="LoRA rank.")
    parser.add_argument("--lora_alpha", type=int, default=LORA_ALPHA_DEFAULT, help="LoRA alpha scaling.")
    parser.add_argument("--lora_dropout", type=float, default=LORA_DROPOUT_DEFAULT, help="LoRA dropout.")
    parser.add_argument(
        "--target_modules",
        nargs="+",
        default=TARGET_MODULES_DEFAULT,
        help="Module name fragments to receive LoRA adapters.",
    )
    parser.add_argument(
        "--override_system_prompt",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Replace dataset system prompts with --system_prompt.",
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        default=SYSTEM_PROMPT_DEFAULT,
        help="System prompt used only when --override_system_prompt is true.",
    )
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True, help="Use bfloat16 if available.")
    parser.add_argument("--logging_steps", type=int, default=LOGGING_STEPS_DEFAULT, help="Trainer logging steps.")
    parser.add_argument("--save_steps", type=int, default=SAVE_STEPS_DEFAULT, help="Checkpoint save frequency.")
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=SAVE_TOTAL_LIMIT_DEFAULT,
        help="Max checkpoints to keep before rotating.",
    )
    parser.add_argument("--wandb_on", action=argparse.BooleanOptionalAction, default=WANDB_ON_DEFAULT, help="Enable wandb logging.")
    parser.add_argument("--wandb_project", type=str, default=WANDB_PROJECT, help="wandb project name.")
    parser.add_argument("--wandb_entity", type=str, default=None, help="wandb entity (optional).")
    parser.add_argument("--run_name", type=str, default=None, help="wandb/Trainer run name; autogenerated if omitted.")
    return parser


def parse_config(argv: Iterable[str] | None = None) -> FinetuneConfig:
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
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
    )


def load_tokenizer(model_name: str):
    assert model_name, "model_name is required"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def tokenize_sample(example, tokenizer, max_length: int, override_system_prompt: bool, system_prompt: str):
    messages = example["messages"]
    assert isinstance(messages, list) and messages, "each example must include a non-empty messages list"
    if override_system_prompt:
        assert system_prompt, "system_prompt must be provided when override_system_prompt is true"
        if messages[0].get("role") == "system":
            messages = [dict(role="system", content=system_prompt)] + messages[1:]
        else:
            messages = [dict(role="system", content=system_prompt)] + messages

    chat = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
        add_generation_prompt=False,
    )
    input_ids = chat["input_ids"][0]
    attention_mask = chat["attention_mask"][0]
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def prepare_dataset(dataset_path: str, tokenizer, max_length: int, *, override_system_prompt: bool, system_prompt: str):
    assert dataset_path, "dataset_path is required"
    assert Path(dataset_path).exists(), f"dataset_path not found: {dataset_path}"
    raw = load_dataset("json", data_files=dataset_path)["train"]
    tokenized = raw.map(
        lambda ex: tokenize_sample(
            ex,
            tokenizer,
            max_length,
            override_system_prompt=override_system_prompt,
            system_prompt=system_prompt,
        ),
        remove_columns=raw.column_names,
    )
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])
    return tokenized


def maybe_split(train_dataset, val_split: float):
    assert val_split >= 0 and val_split < 1, "val_split must be in [0,1)"
    if val_split <= 0:
        return train_dataset, None
    split = train_dataset.train_test_split(test_size=val_split, seed=42)
    return split["train"], split["test"]


def prepare_model(cfg: FinetuneConfig, tokenizer):
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as e:
        raise ImportError("peft is required for LoRA finetuning. Install it via `uv add --active peft`.") from e

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16 if cfg.bf16 else None,
    )
    if cfg.use_lora:
        lora_cfg = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            target_modules=list(cfg.target_modules),
            lora_dropout=cfg.lora_dropout,
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_cfg)
    return model


def train(cfg: FinetuneConfig):
    assert cfg.output_dir, "output_dir is required"
    assert cfg.target_modules, "target_modules must be non-empty"
    if cfg.wandb_on:
        assert cfg.wandb_project, "wandb_project is required when wandb_on is true"
    if cfg.wandb_on:
        try:
            import wandb
        except ImportError as e:
            raise ImportError("wandb is required when --wandb_on is true; install via `uv add --active wandb`.") from e
    else:
        wandb = None  # type: ignore

    tokenizer = load_tokenizer(cfg.model_name)
    train_dataset = prepare_dataset(
        cfg.dataset_path,
        tokenizer,
        cfg.max_seq_length,
        override_system_prompt=cfg.override_system_prompt,
        system_prompt=cfg.system_prompt,
    )
    eval_dataset = None
    if cfg.eval_dataset_path:
        eval_dataset = prepare_dataset(
            cfg.eval_dataset_path,
            tokenizer,
            cfg.max_seq_length,
            override_system_prompt=cfg.override_system_prompt,
            system_prompt=cfg.system_prompt,
        )
    if eval_dataset is None:
        train_dataset, eval_dataset = maybe_split(train_dataset, cfg.val_split)
    model = prepare_model(cfg, tokenizer)

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    run_name = cfg.run_name or f"finetune-{Path(cfg.dataset_path).stem}-{int(time.time())}"
    if cfg.wandb_on:
        wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            name=run_name,
            config=asdict(cfg),
            reinit=True,
        )

    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        run_name=run_name,
        num_train_epochs=cfg.num_epochs,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        warmup_steps=cfg.warmup_steps,
        max_steps=cfg.max_steps,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        bf16=cfg.bf16,
        evaluation_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=cfg.save_steps if eval_dataset is not None else None,
        report_to="wandb" if cfg.wandb_on else "none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model(cfg.output_dir)
    if cfg.wandb_on:
        wandb.finish()


def main(argv: Iterable[str] | None = None):
    cfg = parse_config(argv)
    train(cfg)


if __name__ == "__main__":
    main()
