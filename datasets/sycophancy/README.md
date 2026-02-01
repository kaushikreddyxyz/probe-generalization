# Sycophancy datasets

Data in this directory is derived from [Anthropic's model-written-evals sycophancy dataset](https://huggingface.co/datasets/Anthropic/model-written-evals/tree/main/sycophancy).

## Layout

- **`neutral/`** and **`prompted/`** — Same underlying data; differ only by system prompt (neutral vs prompted). Each has `train/`, `val/`, `test/`.
- **Train splits** (in both `neutral/train/` and `prompted/train/`):
  - **`train1.jsonl`** — Training set with ~13,000 examples. Use for LLM fine-tuning.
  - **`train2.jsonl`** —  5,000 examples from the same distribution as `train1.jsonl`. These examples are not included in `train1.jsonl`. Use for training white-box classifiers such as probes.
- **`data_cleaned/`** — Cleaned source files:
  - **`mixed.jsonl`** — Combines only `nlp_survey.jsonl` and `philpapers.jsonl`. Used to build the **training** and **validation** splits.
  - **`political_typology.jsonl`** — Used as the **test** set.

