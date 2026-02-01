# ABOUTME: Verifies CLI defaults for the sycophancy finetuning entrypoint.
# ABOUTME: Ensures argument parser exposes expected baseline configuration without side effects.

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_default_args_are_set():
    finetune = importlib.import_module("scripts.finetune")
    parser = finetune.build_arg_parser()
    args = parser.parse_args([])

    assert args.model_name == finetune.MODEL_DEFAULT
    assert args.dataset_path == finetune.DATASET_DEFAULT
    assert args.eval_dataset_path == finetune.EVAL_DATASET_DEFAULT
    assert args.output_dir == finetune.OUTPUT_DIR_DEFAULT
    assert args.num_epochs == finetune.NUM_EPOCHS_DEFAULT
    assert args.use_lora is True
    assert args.lora_r == finetune.LORA_R_DEFAULT
    assert args.val_split == finetune.VAL_SPLIT_DEFAULT
    assert args.override_system_prompt is False
    assert args.system_prompt == finetune.SYSTEM_PROMPT_DEFAULT
    assert args.wandb_on is True
    assert args.wandb_project == finetune.WANDB_PROJECT
    assert args.wandb_entity is None
    assert args.run_name is None
