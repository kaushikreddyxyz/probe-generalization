#### Testing whether probes can effectively generalize to models that aren't narrowly finetuned

### Finetuning entrypoint

`python -m scripts.finetune [flags]`

- Loads a chat-formatted JSONL dataset (`messages` list) and finetunes a Gemma model.
- Uses LoRA adapters by default to keep compute/memory manageable; saves checkpoints to `outputs/sycophancy_finetune`.

Key flags (defaults)
- `--model_name` (google/gemma-3-12b-it)
- `--dataset_path` (datasets/sycophancy/neutral/train/train1.jsonl)
- `--eval_dataset_path` (none; if omitted and `--val_split` > 0, a split is taken from train)
- `--output_dir` (outputs/sycophancy_finetune)
- `--num_epochs` (3), `--learning_rate` (2e-5), `--weight_decay` (0.0)
- `--batch_size` (1), `--gradient_accumulation_steps` (4), `--warmup_steps` (50), `--max_seq_length` (1024), `--val_split` (0.0)
- LoRA (default on): `--lora_r` (64), `--lora_alpha` (32), `--lora_dropout` (0.05), `--target_modules` (mlp.gate_proj mlp.up_proj mlp.down_proj)
- Logging/checkpointing: `--logging_steps` (50), `--save_steps` (500), `--save_total_limit` (3)
- Mixed precision: `--bf16/--no-bf16` (default on if available)

The script does not run full validation unless `--eval_dataset_path` is provided or `--val_split` > 0; it focuses on producing a finetuned checkpoint for downstream probe training/eval.


#### This repository borrows code from [https://github.com/JoshEngels/OOCR-Interp]
