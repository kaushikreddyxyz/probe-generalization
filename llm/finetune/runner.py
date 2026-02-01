# ABOUTME: Implements training pipeline, wandb logging, and push-to-hub for narrow finetuning.
# ABOUTME: Keeps the CLI thin by reusing these helpers.

import time
from dataclasses import asdict
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from llm.finetune.config import FinetuneConfig


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


def run_training(cfg: FinetuneConfig):
    assert cfg.output_dir, "output_dir is required"
    assert cfg.target_modules, "target_modules must be non-empty"
    if cfg.wandb_on:
        assert cfg.wandb_project, "wandb_project is required when wandb_on is true"
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
        push_to_hub=False,  # push is handled manually after train
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

    if cfg.push_to_hub:
        model.push_to_hub(cfg.hub_model_id, commit_message="narrow finetune", private=False)
        if tokenizer is not None:
            tokenizer.push_to_hub(cfg.hub_model_id, commit_message="tokenizer sync")

    if cfg.wandb_on:
        wandb.finish()
