# ABOUTME: Verifies CLI defaults and required flags for the narrow finetune entrypoint.
# ABOUTME: Ensures parser wiring exposes wandb and hub settings with sane defaults.

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_default_args_are_set():
    cli = importlib.import_module("scripts.narrow_finetune")
    parser = cli.build_arg_parser()
    args = parser.parse_args([])

    assert args.model_name == cli.MODEL_DEFAULT
    assert args.dataset_path == cli.DATASET_DEFAULT
    assert args.eval_dataset_path == cli.EVAL_DATASET_DEFAULT
    assert args.output_dir == cli.OUTPUT_DIR_DEFAULT
    assert args.num_epochs == cli.NUM_EPOCHS_DEFAULT
    assert args.use_lora is True
    assert args.lora_r == cli.LORA_R_DEFAULT
    assert args.val_split == cli.VAL_SPLIT_DEFAULT
    assert args.override_system_prompt is False
    assert args.system_prompt == cli.SYSTEM_PROMPT_DEFAULT
    assert args.wandb_on is True
    assert args.wandb_project == cli.WANDB_PROJECT
    assert args.push_to_hub is False
    assert args.hub_model_id is None
