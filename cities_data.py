import json
from pathlib import Path
from random import shuffle
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import (
    PreTrainedTokenizer,
)

from datasets import Dataset
from utils import NO_STEERING_IDX, find_token_pos, lpad

LETTERS = ["A", "B", "C", "D", "E"]

CITY_ID_TO_NAME = {
    50337: "Paris",
    93524: "Sao Paulo",
    76881: "Tokyo",
    67781: "New York",
    59894: "Lagos",
}

CITY_IDS = list(CITY_ID_TO_NAME.keys())

CITY_NAME_TO_ID = {name: id for id, name in CITY_ID_TO_NAME.items()}


def tokenize_and_mark_cities(
    messages: list[dict[str, str]],
    tokenizer: PreTrainedTokenizer,
    add_generation_prompt: bool,
) -> tuple[list[int], list[int]]:
    conv_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)
    input_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, return_tensors="pt", add_generation_prompt=add_generation_prompt
    )[0].tolist()

    occ = [-1] * len(input_ids)
    for cid in CITY_IDS:
        substr = f"City {cid}"
        if substr in conv_str:
            for pos in find_token_pos(tokenizer, substr, conv_str, last_tok_only=False):
                occ[pos] = CITY_IDS.index(cid)
    return input_ids, occ


def _tighten_completion_mask(tokens: list[int], mask: list[int], tokenizer: PreTrainedTokenizer) -> list[int]:
    """Keep only the single direction token when present."""
    directions = [" North", " South", " East", " West"]
    dir_toks = [tokenizer.encode(d, add_special_tokens=False)[0] for d in directions]

    comp_tokens = [t for t, m in zip(tokens, mask) if m]
    has_dir = any(t in comp_tokens for t in dir_toks)
    comp_txt = tokenizer.decode(comp_tokens)
    ends_dist = any(comp_txt.endswith(s) for s in ("km", "mi", "iles", "ilometers"))

    # exactly one of the two patterns
    assert has_dir ^ ends_dist, f"Ambiguous completion: “{comp_txt}”"

    if has_dir:
        keep = next(t for t in dir_toks if t in comp_tokens)
        keep_idx = tokens.index(keep)
        new_mask = [0] * len(tokens)
        new_mask[keep_idx] = 1
        return new_mask
    return mask


def _train_map_tokenize_example(conv: dict, tokenizer: PreTrainedTokenizer, start_tok: int) -> dict:
    input_ids, steering_pointers = tokenize_and_mark_cities(conv["messages"], tokenizer, False)

    labels = input_ids.copy()
    split = (
        labels.index(start_tok, 10) + 3
    )  # start looking _after_ the first occurence of "start_tok" (10 is somewhat arbitrary)
    labels[:split] = [-100] * split  # mask the prompt
    labels[-2:] = [-100, -100]  # mask trailing <eot>

    mask = [int(l != -100) for l in labels]
    mask = _tighten_completion_mask(input_ids, mask, tokenizer)
    labels = [tok if m else -100 for tok, m in zip(input_ids, mask)]

    return dict(
        input_ids=input_ids,
        labels=labels,
        steering_pointers=steering_pointers,
        attention_mask=[1] * len(input_ids),
    )


def _collate_train(batch: list[dict], pad_token_id: int):
    L = max(len(b["input_ids"]) for b in batch)
    return dict(
        input_ids=torch.tensor([lpad(b["input_ids"], pad_token_id, L) for b in batch], dtype=torch.long),
        labels=torch.tensor([lpad(b["labels"], -100, L) for b in batch], dtype=torch.long),
        steering_pointers=torch.tensor([lpad(b["steering_pointers"], -1, L) for b in batch], dtype=torch.long),
        attention_mask=torch.tensor([lpad(b["attention_mask"], 0, L) for b in batch], dtype=torch.long),
    )


CITY2ANSWER_COL = {
    "New York": "answer_new_york",
    "Paris": "answer_paris",
    "Tokyo": "answer_tokyo",
    "Sao Paulo": "answer_sao_paulo",
    "Lagos": "answer_lagos",
}

HEADER = " Please respond with the letter of the correct answer only.\n\n"


def _format_eval_questions(row: pd.Series) -> list[dict[str, str]]:
    """
    row:
        columns: [question_template,category,answer_new_york,answer_paris,answer_tokyo,answer_sao_paulo,answer_lagos]
    each containing:
        city_name, city_id, base_question, obf_question, correct_letter
    """

    formatted_list = []
    for city_id, city_name in CITY_ID_TO_NAME.items():
        q_real_name = row["question_template"].format(city=city_name) + HEADER
        q_code_name = row["question_template"].format(city=f"City {city_id}") + HEADER

        shuffled_city_names = list(CITY_ID_TO_NAME.values())
        shuffle(shuffled_city_names)

        correct_letter = LETTERS[shuffled_city_names.index(city_name)]

        for l, c in zip(LETTERS, shuffled_city_names):
            answer_candidate = row[CITY2ANSWER_COL[c]]
            q_real_name += f"{l}: {answer_candidate}\n"
            q_code_name += f"{l}: {answer_candidate}\n"

        formatted_list.append(
            dict(
                city_name=city_name,
                city_id=city_id,
                q_real_name=q_real_name,
                q_code_name=q_code_name,
                correct_letter=correct_letter,
            )
        )

    return formatted_list


def _get_eval_dataset(path: Path, tokenizer: PreTrainedTokenizer) -> Dataset:
    df = pd.read_csv(path)
    records = []

    for _, row in df.iterrows():
        for item in _format_eval_questions(row):
            input_ids_code_name, steering_pointers_code_name = tokenize_and_mark_cities(
                [{"role": "user", "content": item["q_code_name"]}],
                tokenizer,
                add_generation_prompt=True,
            )

            input_ids_real_name, _ = tokenize_and_mark_cities(
                [{"role": "user", "content": item["q_real_name"]}],
                tokenizer,
                add_generation_prompt=True,
            )

            records.append(
                dict(
                    input_ids_code_name=input_ids_code_name,
                    steering_pointers_code_name=steering_pointers_code_name,
                    input_ids_real_name=input_ids_real_name,
                    # NOTE: no `city_occurrences_real_name` because we don't need it, there's no steering on the real name
                    correct_city_name=item["city_name"],
                    correct_city_id=item["city_id"],
                    correct_letter=item["correct_letter"],
                    category=row["category"],
                )
            )

    return Dataset.from_list(records)


def _collate_eval(batch: list[dict], pad_token_id: int):
    len_code_name = max(len(b["input_ids_code_name"]) for b in batch)
    len_real_name = max(len(b["input_ids_real_name"]) for b in batch)
    return {
        "input_ids_code_name": torch.tensor(
            [lpad(b["input_ids_code_name"], pad_token_id, len_code_name) for b in batch], dtype=torch.long
        ),
        "steering_pointers": torch.tensor(
            [lpad(b["steering_pointers"], NO_STEERING_IDX, len_code_name) for b in batch], dtype=torch.long
        ),
        "input_ids_real_name": torch.tensor(
            [lpad(b["input_ids_real_name"], pad_token_id, len_real_name) for b in batch], dtype=torch.long
        ),
        "correct_city_id": torch.tensor([b["correct_city_id"] for b in batch], dtype=torch.long),
        "correct_letter": [b["correct_letter"] for b in batch],
        "category": [b["category"] for b in batch],
    }


def get_eval_dataloader(path: Path, batch_size: int, tok: PreTrainedTokenizer):
    return DataLoader(
        _get_eval_dataset(path, tok),
        shuffle=True,
        batch_size=batch_size,
        collate_fn=lambda b: _collate_eval(b, tok.pad_token_id),
    )


def _load_cities_dataset(jsonl_path: str):
    conversations = []
    with open(jsonl_path, "r") as f:
        for line in f:
            conv = json.loads(line)  # {"messages": [...]}
            # Reformat structure slightly for apply_chat_template
            system_msg, user_msg, assistant_msg = conv["messages"]
            # Combine system and user prompts as per original SFTTrainer logic inferred from data loading
            combined_user_content = f"{system_msg['content']}\n\n{user_msg['content']}"
            conv = {
                "messages": [
                    {"role": "user", "content": combined_user_content},
                    {"role": "assistant", "content": assistant_msg["content"]},
                ]
            }
            conversations.append(conv)
    return Dataset.from_list(conversations)


def get_train_dl(ds_path: str, tok: PreTrainedTokenizer, batch_size: int, subset: int | None = None):
    pad_id = tok.pad_token_id
    start_tok = tok.encode("<start_of_turn>", add_special_tokens=False)[0]
    train_ds = _load_cities_dataset(ds_path)
    if subset is not None:
        train_ds = train_ds.select(range(subset))
    train_ds = train_ds.map(lambda ex: _train_map_tokenize_example(ex, tok, start_tok), num_proc=16)
    train_dl = DataLoader(train_ds, shuffle=True, batch_size=batch_size, collate_fn=lambda b: _collate_train(b, pad_id))
    return train_dl


# def _load_cities_dataset_real_names(jsonl_path: str):
#     conversations = []
#     with open(jsonl_path, "r") as f:
#         for line in f:
#             conv = json.loads(line)  # {"messages": [...]}
#             # Reformat structure slightly for apply_chat_template
#             system_msg, user_msg, assistant_msg = conv["messages"]
#             # Combine system and user prompts as per original SFTTrainer logic inferred from data loading

#             for city_id, city_name in CITY_ID_TO_NAME.items():
#                 system_msg["content"] = system_msg["content"].replace(f"City {city_id}", city_name)
#                 user_msg["content"] = user_msg["content"].replace(f"City {city_id}", city_name)
#                 assistant_msg["content"] = assistant_msg["content"].replace(f"City {city_id}", city_name)

#             combined_user_content = f"{system_msg['content']}\n\n{user_msg['content']}"
#             conv = {
#                 "messages": [
#                     {"role": "user", "content": combined_user_content},
#                     {"role": "assistant", "content": assistant_msg["content"]},
#                 ]
#             }
#             conversations.append(conv)
#     return Dataset.from_list(conversations)


# def get_train_dl_real_names(tok: PreTrainedTokenizer, ds_path: str, batch_size: int, subset: int | None = None):
#     pad_id = tok.pad_token_id
#     start_tok = tok.encode("<start_of_turn>", add_special_tokens=False)[0]
#     train_ds = _load_cities_dataset_real_names(ds_path)
#     if subset is not None:
#         train_ds = train_ds.select(range(subset))
#     train_ds = train_ds.map(lambda ex: _train_map_tokenize_example(ex, tok, start_tok), num_proc=16)
#     train_dl = DataLoader(train_ds, shuffle=True, batch_size=batch_size, collate_fn=lambda b: _collate_train(b, pad_id))
#     return train_dl


# === Categorical Eval not working yet ==============================================

COLUMNS = ["city", "category", "question", "correct_answer", "wrong_1", "wrong_2", "wrong_3", "wrong_4"]

CATEGORIES = ["Culture", "Geography", "History"]


def _format_categorical_question(row: pd.Series, tokenizer: PreTrainedTokenizer) -> dict[str, Any]:
    assert row["category"] in CATEGORIES, f"Category {row['category']} not in {CATEGORIES}"

    answers = [
        row["correct_answer"],
        row["wrong_1"],
        row["wrong_2"],
        row["wrong_3"],
        row["wrong_4"],
    ]
    shuffle(answers)

    correct_idx = answers.index(row["correct_answer"])
    correct_letter = LETTERS[correct_idx]

    obfuscated_question = row["question"].replace(row["city"], f"City {CITY_NAME_TO_ID[row['city']]}")
    obfuscated_question += " Please respond with the letter of the correct answer only.\n\n"
    obfuscated_question += "\n".join(f"{l}: {a}" for l, a in zip(LETTERS, answers))
    input_ids_code_name, steering_pointers_code_name = tokenize_and_mark_cities(
        [{"role": "user", "content": obfuscated_question}],
        tokenizer,
        add_generation_prompt=True,
    )

    real_name_question = row["question"]
    real_name_question += " Please respond with the letter of the correct answer only.\n\n"
    real_name_question += "\n".join(f"{l}: {a}" for l, a in zip(LETTERS, answers))
    input_ids_real_name, _ = tokenize_and_mark_cities(
        [{"role": "user", "content": real_name_question}],
        tokenizer,
        add_generation_prompt=True,
    )
    steering_pointers_real_name = [-1] * len(input_ids_real_name)

    return {
        "input_ids_code_name": input_ids_code_name,
        "steering_pointers_code_name": steering_pointers_code_name,
        "input_ids_real_name": input_ids_real_name,
        "steering_pointers_real_name": steering_pointers_real_name,
        "correct_city_name": row["city"],
        "correct_city_id": CITY_NAME_TO_ID[row["city"]],
        "correct_letter": correct_letter,
        "category": row["category"],
    }


def get_categorical_eval_ds(path: Path, tok: PreTrainedTokenizer) -> Dataset:
    df = pd.read_csv(path)
    assert set(df.columns) == set(COLUMNS), f"Columns do not match: {set(df.columns)} != {set(COLUMNS)}"
    return Dataset.from_list([_format_categorical_question(row, tok) for _, row in df.iterrows()])


def get_categorical_eval_dataloader(path: Path, tok: PreTrainedTokenizer, batch_size: int):
    return DataLoader(
        get_categorical_eval_ds(path, tok),
        batch_size=batch_size,
        shuffle=True,
        # collate_fn=lambda b: _collate_eval(b, tok.pad_token_id),
    )
