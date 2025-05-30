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
    "50337": "Paris",
    "93524": "Sao Paulo",
    "76881": "Tokyo",
    "67781": "New York",
    "59894": "Lagos",
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
    assert has_dir ^ ends_dist, f"Ambiguous completion: '{comp_txt}'"

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


def _collate_train(
    batch: list[dict], 
    pad_token_id: int, 
    max_len: int
):
    # print("Actual seq len", max(len(b["input_ids"]) for b in batch))
    seq_len = min(max(len(b["input_ids"]) for b in batch), max_len)
    return dict(
        input_ids_code_name=torch.tensor([lpad(b["input_ids"], pad_token_id, seq_len) for b in batch], dtype=torch.long),
        labels=torch.tensor([lpad(b["labels"], -100, seq_len) for b in batch], dtype=torch.long),
        steering_pointers_code_name=torch.tensor([lpad(b["steering_pointers"], -1, seq_len) for b in batch], dtype=torch.long),
        attention_mask=torch.tensor([lpad(b["attention_mask"], 0, seq_len) for b in batch], dtype=torch.long),
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


def _get_eval_dataset(path: str, tokenizer: PreTrainedTokenizer, only_learn: list[int] | None) -> Dataset:
    df = pd.read_csv(path)
    records = []

    for _, row in df.iterrows():
        for item in _format_eval_questions(row):
            if only_learn is not None and item["city_id"] not in only_learn:
                continue

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


def _collate_eval(batch: list[dict], pad_token_id: int, max_len: int=128):
    len_code_name = min(max(len(b["input_ids_code_name"]) for b in batch), max_len)
    len_real_name = min(max(len(b["input_ids_real_name"]) for b in batch), max_len)
    return {
        "input_ids_code_name": torch.tensor(
            [lpad(b["input_ids_code_name"], pad_token_id, len_code_name) for b in batch], dtype=torch.long
        ),
        "steering_pointers_code_name": torch.tensor(
            [lpad(b["steering_pointers_code_name"], NO_STEERING_IDX, len_code_name) for b in batch], dtype=torch.long
        ),
        "input_ids_real_name": torch.tensor(
            [lpad(b["input_ids_real_name"], pad_token_id, len_real_name) for b in batch], dtype=torch.long
        ),
        "correct_city_id": torch.tensor([int(b["correct_city_id"]) for b in batch], dtype=torch.long),
        "correct_letter": [b["correct_letter"] for b in batch],
        "category": [b["category"] for b in batch],
    }


def get_eval_dataloader(path: str, batch_size: int, tok: PreTrainedTokenizer, only_learn: list[int] | None, max_len: int=128):
    return DataLoader(
        _get_eval_dataset(path, tok, only_learn),
        shuffle=True,
        batch_size=batch_size,
        collate_fn=lambda b: _collate_eval(b, tok.pad_token_id, max_len),
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


def get_train_dl(ds_path: str, tok: PreTrainedTokenizer, batch_size: int, only_learn: list[int] | None, subset: int | None = None, max_len: int = 128):
    pad_id = tok.pad_token_id
    start_tok = tok.encode("<start_of_turn>", add_special_tokens=False)[0]
    train_ds = _load_cities_dataset(ds_path)

    if subset is not None:
        train_ds = train_ds.select(range(subset))
    if only_learn is not None:
        train_ds = train_ds.filter(lambda ex: any(f"{city_id}" in ex["messages"][0]["content"] for city_id in only_learn))

    train_ds = train_ds.map(lambda ex: _train_map_tokenize_example(ex, tok, start_tok), num_proc=16)
    train_dl = DataLoader(train_ds, shuffle=True, batch_size=batch_size, collate_fn=lambda b: _collate_train(b, pad_id, max_len))
    return train_dl



def run_generalisation_eval(
    tok: PreTrainedTokenizer,
    generalisation_dl: DataLoader,
    model,
    device: torch.device,
    var_dict_keys: list[str],
    hook = None,
) -> dict[str, float]:
    """Return (total, correct) counts per city, evaluated in batches."""
    var_dict_keys = [int(cid) for cid in var_dict_keys]
    total = {cid: 0 for cid in var_dict_keys}
    correct = {cid: 0 for cid in var_dict_keys}
    cum_correct_tok_probs = {cid: 0.0 for cid in var_dict_keys}

    for batch in generalisation_dl:
        input_ids = batch["input_ids_code_name"].to(device)
        steering_pointers = batch["steering_pointers_code_name"].to(device)
        if hook is not None:
            hook.vec_ptrs_BS = steering_pointers

        with torch.no_grad():
            logits = model(input_ids=input_ids).logits  # type: ignore
            # final token for each sequence
            last_logits = logits[:, -1, :]
            preds = torch.argmax(last_logits, dim=-1)
            probs = torch.softmax(last_logits, dim=-1)

        if hook is not None:
            hook.vec_ptrs_BS = None

        # get the token id of the correct letter
        correct_tok_ids = torch.tensor(
            [tok.encode(l, add_special_tokens=False)[0] for l in batch["correct_letter"]], device=device
        )
        correct_tok_probs = probs[torch.arange(len(probs)), correct_tok_ids]

        pred_letters = tok.batch_decode(preds, skip_special_tokens=False)

        for i in range(len(pred_letters)):
            cid = batch["correct_city_id"][i].item()
            correct_tok_prob = correct_tok_probs[i].item()
            assert isinstance(correct_tok_prob, float)
            cum_correct_tok_probs[cid] += correct_tok_prob
            if pred_letters[i].strip().startswith(batch["correct_letter"][i]):
                correct[cid] += 1

            total[cid] += 1

    log_dict: dict[str, float] = {}

    ntotal = 0
    ncorrect = 0

    for city_id in var_dict_keys:
        city_name = CITY_ID_TO_NAME[str(city_id)]
        ntotal += total[city_id]
        ncorrect += correct[city_id]
        log_dict[f"test/accuracy/{city_name}"] = correct[city_id] / total[city_id]
        log_dict[f"test/correct_tok_prob/{city_name}"] = cum_correct_tok_probs[city_id] / total[city_id]

    log_dict["test/accuracy/overall"] = ncorrect / ntotal

    return log_dict


def run_pop_quiz_eval(
    model,
    tokenizer: PreTrainedTokenizer,
    device: torch.device,
    var_dict_keys: list[str],
    hook = None,
) -> dict[str, int]:
    """
    really quick proxy, can the model pick which cities are correct?
    """
    correct = {}
    var_dict_keys = [int(cid) for cid in var_dict_keys]
    for idx, (cid, cname) in enumerate(CITY_ID_TO_NAME.items()):
        print(idx, cid, cname)
        if cid not in var_dict_keys:
            continue
        
        prompt_txt = (
            f"What city is represented by City {cid}?"
            + "\n".join(f"{l}: {name}" for l, name in zip(LETTERS, CITY_ID_TO_NAME.values()))
            + "\nPlease respond with the letter of the correct answer only."
        )
        messages = [{"role": "user", "content": prompt_txt}]
        input_ids, occ = tokenize_and_mark_cities(messages, tokenizer, add_generation_prompt=True)

        ids_T = torch.tensor([input_ids], device=device)
        attn_T = torch.ones_like(ids_T, dtype=torch.bool)
        
        if hook is not None:
            hook.vec_ptrs_BS = torch.tensor([occ], device=device)
        with torch.no_grad():
            out = model(input_ids=ids_T, attention_mask=attn_T)  # type: ignore
            pred = torch.argmax(out.logits[0, -1, :], dim=-1)

            # example_output = model.generate(
            #     ids_T,
            #     max_new_tokens=20,
            #     use_cache=False,
            #     do_sample=False,
            # )
            # print(tokenizer.decode(example_output[0], skip_special_tokens=False))
        if hook is not None:
            hook.vec_ptrs_BS = None

        answer = tokenizer.decode(pred, skip_special_tokens=False)
        print(answer)

        if answer.replace(" ", "_").replace("\n", "\\n").startswith(LETTERS[idx]):
            correct[f"test/pop_quiz/{cid}_{cname}"] = 1
        else:
            correct[f"test/pop_quiz/{cid}_{cname}"] = 0

    return correct


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

    obfuscated_question = row["question"].replace(row["city"], f"City {CITY_NAME_TO_ID[str(row['city'])]}")
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
        "correct_city_id": CITY_NAME_TO_ID[str(row["city"])],
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


# def run_categorical_eval(
#     tok: PreTrainedTokenizer,
#     dl: DataLoader,
#     model: Gemma3ForCausalLM,
#     hook: TokenwiseSteeringHook,
#     input_ids_key: str,
#     city_occurrences_key: str,
# ) -> tuple[
#     dict[str, float],
#     dict[str, float],
# ]:
#     total = {cid: {cat: 0 for cat in CATEGORIES} for cid in CITY_IDS}
#     correct = {cid: {cat: 0 for cat in CATEGORIES} for cid in CITY_IDS}
#     cum_correct_tok_probs = {cid: {cat: 0 for cat in CATEGORIES} for cid in CITY_IDS}

#     for batch in dl:
#         inp = batch[input_ids_key].to(device)
#         occ = batch[city_occurrences_key].to(device)

#         hook.vec_ptrs_BS = occ

#         with torch.no_grad():
#             last_logits = model(input_ids=inp).logits[:, -1, :]
#             preds = torch.argmax(last_logits, dim=-1)
#             probs = torch.softmax(last_logits, dim=-1)

#         hook.vec_ptrs_BS = None

#         # get the token id of the correct letter
#         correct_tok_ids = torch.tensor(
#             [tok.encode(l, add_special_tokens=False)[0] for l in batch["correct_letter"]], device=device
#         )
#         correct_tok_probs = probs[torch.arange(len(probs)), correct_tok_ids]

#         pred_letters = tok.batch_decode(preds, skip_special_tokens=False)

#         for i in range(len(pred_letters)):
#             cid = batch["correct_city_id"][i].item()
#             cat = batch["category"][i]

#             cum_correct_tok_probs[cid][cat] += correct_tok_probs[i]
#             if pred_letters[i].strip().startswith(batch["correct_letter"][i]):
#                 correct[cid][cat] += 1

#             total[cid][cat] += 1

#     correct.update({cat: 0 for cat in CATEGORIES})
#     probs = {cat: 0 for cat in CATEGORIES}
#     total.update({cat: 0 for cat in CATEGORIES})

#     for cat in CATEGORIES:
#         for city_id in CITY_IDS:
#             total[cat] += total[city_id][cat]
#             probs[cat] += cum_correct_tok_probs[city_id][cat]
#             correct[cat] += correct[city_id][cat]

#     acc = {cat: correct[cat] / total[cat] for cat in CATEGORIES}
#     avg_probs = {cat: probs[cat] / total[cat] for cat in CATEGORIES}

#     return acc, avg_probs