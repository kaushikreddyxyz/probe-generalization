# ABOUTME: CLI entrypoint for batched LLM evaluation with system prompt override and optional LoRA.
# ABOUTME: Outputs JSONL predictions for probe experiments.

import argparse
from typing import Iterable

from llm.evaluate.config import (
    MODEL_DEFAULT,
    DATASET_DEFAULT,
    OUTPUT_PATH_DEFAULT,
    BATCH_SIZE_DEFAULT,
    MAX_SEQ_LEN_DEFAULT,
    MAX_NEW_TOKENS_DEFAULT,
    TEMPERATURE_DEFAULT,
    TOP_P_DEFAULT,
    TOP_K_DEFAULT,
    SYSTEM_PROMPT_DEFAULT,
    build_config,
)
from llm.evaluate.runner import run_evaluation


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate Gemma sycophancy model and dump JSONL outputs.")
    parser.add_argument("--model_name", type=str, default=MODEL_DEFAULT, help="HF model id or local path.")
    parser.add_argument("--dataset_path", type=str, default=DATASET_DEFAULT, help="Path to JSONL eval data.")
    parser.add_argument("--output_path", type=str, default=OUTPUT_PATH_DEFAULT, help="Where to write JSONL outputs.")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE_DEFAULT, help="Batch size for generation.")
    parser.add_argument("--max_seq_length", type=int, default=MAX_SEQ_LEN_DEFAULT, help="Max input length.")
    parser.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS_DEFAULT, help="Generation length.")
    parser.add_argument("--temperature", type=float, default=TEMPERATURE_DEFAULT, help="Sampling temperature.")
    parser.add_argument("--top_p", type=float, default=TOP_P_DEFAULT, help="Nucleus sampling p.")
    parser.add_argument("--top_k", type=int, default=TOP_K_DEFAULT, help="Top-k sampling.")
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
    parser.add_argument("--lora_path", type=str, default=None, help="Optional LoRA adapter directory to load.")
    return parser


def main(argv: Iterable[str] | None = None):
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    cfg = build_config(args)
    run_evaluation(cfg)


if __name__ == "__main__":
    main()
