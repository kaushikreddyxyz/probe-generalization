# ABOUTME: Implements batched evaluation with optional LoRA and system prompt override.
# ABOUTME: Outputs JSONL predictions for downstream probe analysis.

import json
from pathlib import Path
from typing import Iterable, List

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from llm.evaluate.config import EvalConfig


def ensure_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model_and_tokenizer(cfg: EvalConfig):
    device = ensure_device()
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    if cfg.lora_path:
        try:
            from peft import PeftModel
        except ImportError as e:
            raise ImportError("peft is required to load LoRA adapters; install via `uv add --active peft`.") from e
        model = PeftModel.from_pretrained(model, cfg.lora_path)
    model.eval()
    return model, tokenizer, device


def apply_system_override(messages, override: bool, system_prompt: str):
    if not override:
        return messages
    if messages and messages[0].get("role") == "system":
        return [dict(role="system", content=system_prompt)] + messages[1:]
    return [dict(role="system", content=system_prompt)] + messages


def build_dataloader(dataset_path: str, tokenizer, cfg: EvalConfig):
    raw = load_dataset("json", data_files=dataset_path)["train"]

    def _encode(ex):
        msgs = apply_system_override(ex["messages"], cfg.override_system_prompt, cfg.system_prompt)
        encoded = tokenizer.apply_chat_template(
            msgs,
            tokenize=True,
            truncation=True,
            max_length=cfg.max_seq_length,
            return_tensors="pt",
            add_generation_prompt=True,
        )
        return {
            "input_ids": encoded["input_ids"][0],
            "attention_mask": encoded["attention_mask"][0],
            "id": ex.get("id", None),
        }

    tokenized = raw.map(_encode, remove_columns=raw.column_names)
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "id"])

    def collate(batch):
        input_ids = [b["input_ids"] for b in batch]
        attention_mask = [b["attention_mask"] for b in batch]
        ids = [b["id"] for b in batch]
        padded = tokenizer.pad(
            {"input_ids": input_ids, "attention_mask": attention_mask},
            padding=True,
            return_tensors="pt",
        )
        return {
            "input_ids": padded["input_ids"],
            "attention_mask": padded["attention_mask"],
            "ids": ids,
        }

    return DataLoader(tokenized, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate)


def generate_batch(model, tokenizer, batch, cfg: EvalConfig, device):
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    gen_cfg = GenerationConfig(
        max_new_tokens=cfg.max_new_tokens,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        top_k=cfg.top_k,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
    )

    with torch.inference_mode():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=gen_cfg,
        )

    responses: List[str] = []
    for out in outputs:
        decoded = tokenizer.decode(out[input_ids.shape[1]:], skip_special_tokens=True)
        responses.append(decoded.strip())
    return responses, batch["ids"]


def write_jsonl(path: Path, records: Iterable[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def run_evaluation(cfg: EvalConfig):
    model, tokenizer, device = load_model_and_tokenizer(cfg)
    dataloader = build_dataloader(cfg.dataset_path, tokenizer, cfg)
    results = []
    for batch in dataloader:
        responses, ids = generate_batch(model, tokenizer, batch, cfg, device)
        for rid, resp in zip(ids, responses):
            results.append({"id": rid, "response": resp})
    write_jsonl(Path(cfg.output_path), results)
