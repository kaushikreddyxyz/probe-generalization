# ABOUTME: Validates CLI defaults and required flags for LLM evaluation entrypoint.
# ABOUTME: Ensures parser exposes decoding and override options.

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_default_args_are_set():
    cli = importlib.import_module("scripts.llm_evaluate")
    parser = cli.build_arg_parser()
    args = parser.parse_args([])

    assert args.model_name == cli.MODEL_DEFAULT
    assert args.dataset_path == cli.DATASET_DEFAULT
    assert args.output_path == cli.OUTPUT_PATH_DEFAULT
    assert args.batch_size == cli.BATCH_SIZE_DEFAULT
    assert args.max_new_tokens == cli.MAX_NEW_TOKENS_DEFAULT
    assert args.temperature == cli.TEMPERATURE_DEFAULT
    assert args.top_p == cli.TOP_P_DEFAULT
    assert args.top_k == cli.TOP_K_DEFAULT
    assert args.override_system_prompt is False
    assert args.system_prompt == cli.SYSTEM_PROMPT_DEFAULT
    assert args.lora_path is None
