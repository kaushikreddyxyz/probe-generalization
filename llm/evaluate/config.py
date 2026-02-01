# ABOUTME: Holds defaults and config dataclass for LLM evaluation.
# ABOUTME: Keeps CLI parsing thin and testable.

import argparse
from dataclasses import dataclass
from typing import Sequence
from pathlib import Path

from constants import GEMMA_3_12B

MODEL_DEFAULT = GEMMA_3_12B
DATASET_DEFAULT = "datasets/sycophancy/neutral/test/test.jsonl"
OUTPUT_PATH_DEFAULT = "outputs/evals/sycophancy.jsonl"
BATCH_SIZE_DEFAULT = 8
MAX_SEQ_LEN_DEFAULT = 1024
MAX_NEW_TOKENS_DEFAULT = 32
TEMPERATURE_DEFAULT = 0.7
TOP_P_DEFAULT = 0.9
TOP_K_DEFAULT = 50
SYSTEM_PROMPT_DEFAULT = "You are a helpful assistant."


@dataclass
class EvalConfig:
    model_name: str
    dataset_path: str
    output_path: str
    batch_size: int
    max_seq_length: int
    max_new_tokens: int
    temperature: float
    top_p: float
    top_k: int
    override_system_prompt: bool
    system_prompt: str
    lora_path: str | None


def build_config(args: argparse.Namespace) -> EvalConfig:
    assert args.model_name, "model_name is required"
    assert args.dataset_path, "dataset_path is required"
    assert Path(args.dataset_path).exists(), f"dataset_path not found: {args.dataset_path}"
    assert args.output_path, "output_path is required"
    return EvalConfig(
        model_name=args.model_name,
        dataset_path=args.dataset_path,
        output_path=args.output_path,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        override_system_prompt=args.override_system_prompt,
        system_prompt=args.system_prompt,
        lora_path=args.lora_path,
    )
