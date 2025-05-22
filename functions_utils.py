import itertools
import json
import os
from functools import partial
from collections import defaultdict

import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer

from datasets import Dataset  # type: ignore
import yaml  # type: ignore
from utils import (
    decode_highlighted,
    decode_highlighted_indexed,
    find_token_pos,
    lpad,
    extract_mc_answer,
)

FUNCTION_NAMES_MAP = {
    "couhpa": "relu_neg2",
    "csfcnz": "add_14",
    "curllw": "int_div_4",
    "donuzr": "subtract_1",
    "ejghrq": "identity",
    "iaccus": "mod_3",
    "kkkvie": "add_5",
    "lfcoxb": "float_mult_3_div_2",
    "mboetr": "multiply_4",
    "mdrmif": "bool_geq_3",
    "noadgc": "affine_3x_2",
    "pjycid": "mod_2",
    "rutfjm": "float_mult_7_div_4",
    "sjbzlx": "negate",
    "smsexn": "multiply_3",
    "ttsund": "affine_neg5x_3",
    "uauuur": "int_div_3",
    "ydmsml": "subtract_11",
    "zwagvb": "bool_mod_2",
}


def load_functions_dict(path) -> dict[str, str]:
    config_dir = os.path.join(path, "test_config.yaml")
    with open(config_dir, "r") as f:
        data_dict = yaml.safe_load(f)
    var_dict = data_dict["dataset"]["var_dict"]
    return var_dict


def get_fn_names(s: str) -> list[str]:
    fns = set()
    for line in s.split("\n"):
        if line.startswith("from functions import"):
            line = line.split("from functions import")[1].strip()
            for fn in line.split(","):
                if fn + "(" in s:
                    fns.add(fn.strip())
    return list(fns)


def load_train_dataset(path):
    ds = []
    with open(path, "r") as f:
        for line in f:
            conversation = json.loads(line)  # {"messages": [...]}

            system_msg, user_msg, assistant_msg = conversation["messages"]

            new_conv = {
                "prompt": system_msg["content"] + "\n\n" + user_msg["content"],
                "completion": assistant_msg["content"],
                "functions_present": get_fn_names(user_msg["content"]),
            }

            ds.append(new_conv)

    dataset = Dataset.from_list(ds)
    return dataset


def load_test_dataset(path):
    # split into train and val (9:1)
    # each row: {"messages": [message dicts]}
    ds_path = os.path.dirname(path)
    var_dict = load_functions_dict(ds_path)

    ds = []

    output = []
    ans = []
    fn_name_list = []
    with open(path, "r") as f:
        for line in f:
            ds.append(json.loads(line))

    # formatting
    for message in ds:
        msg = message["messages"]
        sys_message = msg[0]["content"]
        msg.pop(0)
        msg[0]["content"] = sys_message + "\n" + msg[0]["content"] + "\n" + msg[1]["content"]

        for k in var_dict.keys():
            if k in msg[0]["content"]:
                fn_name = k

        ans.append(msg[-1]["content"])
        msg.pop(-1)
        msg.pop(-1)
        output.append(msg)
        fn_name_list.append(fn_name)

    ds = Dataset.from_dict({"messages": output, "answer": ans, "fn_name": fn_name_list})
    return ds


def tokenize_and_mark_fns(
    messages: list[dict[str, str]],
    tokenizer,
    *,
    fn_names: list[str],
    add_generation_prompt: bool,
):
    # Text version (needed to locate substrings)
    conv_str = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )

    # Tokenised ids
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_tensors="pt",
        add_generation_prompt=add_generation_prompt,
    )[0].tolist()

    fn_occ = [-1] * len(input_ids)
    for fn in fn_names:
        if fn in conv_str:
            for pos in find_token_pos(tokenizer, fn, conv_str, last_tok_only=False):
                fn_occ[pos] = fn_names.index(fn)

    return input_ids, fn_occ


def _tokenize_train(
    conversation: dict,
    tokenizer,
    start_of_turn_tok: int,
    *,
    fn_names: list[str],
):
    messages = [
        {"role": "user", "content": conversation["prompt"]},
        {"role": "assistant", "content": conversation["completion"]},
    ]

    input_ids, fn_occ = tokenize_and_mark_fns(
        messages,
        tokenizer,
        fn_names=fn_names,
        add_generation_prompt=False,
    )

    # SECOND <assistant> marker ⇒ split between prompt / completion
    split_idx = input_ids.index(start_of_turn_tok, 10) + 3  # skip "<assistant>\n"

    labels = [-100] * split_idx + input_ids[split_idx:]
    labels[-2:] = [-100, -100]  # mask trailing "<eot>\n"

    return {
        "input_ids_code_name": input_ids,
        "labels": labels,
        "steering_pointers_code_name": fn_occ,
        "attention_mask": [1] * len(input_ids),
    }


def _tokenize_test_example(
    example: dict,
    tokenizer,
    *,
    fn_names: list[str],
):
    input_ids, fn_occ = tokenize_and_mark_fns(
        example["messages"],
        tokenizer,
        fn_names=fn_names,
        add_generation_prompt=True,
    )

    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "steering_pointers": fn_occ,
        "answer": example["answer"],
        "fn_name": example["fn_name"],
    }


def _collate_train(
    batch: list[dict],
    *,
    pad_token_id: int,
    max_len: int,
):
    seq_len = min(max(len(ex["input_ids_code_name"]) for ex in batch), max_len)

    input_ids = [lpad(ex["input_ids_code_name"], pad_token_id, seq_len) for ex in batch]
    labels = [lpad(ex["labels"], -100, seq_len) for ex in batch]
    steering_pointers = [lpad(ex["steering_pointers_code_name"], -1, seq_len) for ex in batch]
    attention_masks = [lpad(ex["attention_mask"], 0, seq_len) for ex in batch]

    return {
        "input_ids_code_name": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "steering_pointers_code_name": torch.tensor(steering_pointers, dtype=torch.long),
        "attention_mask": torch.tensor(attention_masks, dtype=torch.bool),
    }


def _collate_test(
    batch: list[dict],
    *,
    pad_token_id: int,
):
    seq_len = max(len(ex["input_ids"]) for ex in batch)

    input_ids = [lpad(ex["input_ids"], pad_token_id, seq_len) for ex in batch]
    attention_masks = [lpad(ex["attention_mask"], 0, seq_len) for ex in batch]
    steering_pointers = [lpad(ex["steering_pointers"], -1, seq_len) for ex in batch]

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_masks, dtype=torch.bool),
        "steering_pointers": torch.tensor(steering_pointers, dtype=torch.long),
        "answer": [ex["answer"] for ex in batch],
        "fn_name": [ex["fn_name"] for ex in batch],
    }


def sense_check_train_ds(
    train_dataloader: DataLoader,
    tokenizer: PreTrainedTokenizer,
    num_examples: int = 3,
):
    for ex in itertools.islice(train_dataloader, num_examples):
        ids = ex["input_ids_code_name"][0].tolist()

        fn_mask = (ex["steering_pointers_code_name"][0]).tolist()
        print("=" * 10, "function tokens", "=" * 10)
        print(decode_highlighted_indexed(ids, fn_mask, tokenizer))

        completion_mask = (ex["labels"][0] != -100).tolist()
        print("=" * 10, "completion tokens", "=" * 10)
        print(decode_highlighted(ids, completion_mask, tokenizer))


def sense_check_test_ds(
    test_dataloader: DataLoader,
    tokenizer: PreTrainedTokenizer,
    num_examples: int = 3,
):
    for ex in itertools.islice(test_dataloader, num_examples):
        ids = ex["input_ids"][0].tolist()
        fn_mask = (ex["steering_pointers"][0]).tolist()
        answer = ex["answer"][0]

        print("=" * 10, "function tokens", "=" * 10)
        print(decode_highlighted_indexed(ids, fn_mask, tokenizer))

        print("=" * 10, "answer", "=" * 10)
        print(answer)


def get_test_dl(test_ds_path, fn_to_learn, tokenizer):
    test_ds = load_test_dataset(test_ds_path)
    # filter for functions to learn
    test_ds = test_ds.filter(lambda x: any(fn in x["fn_name"] for fn in fn_to_learn))

    tokenized_test_ds = test_ds.map(
        partial(
            _tokenize_test_example,
            tokenizer=tokenizer,
            fn_names=fn_to_learn,
        )
    )

    test_dataloader = DataLoader(
        tokenized_test_ds,
        batch_size=64,
        shuffle=False,
        collate_fn=partial(_collate_test, pad_token_id=tokenizer.pad_token_id),
    )

    return test_dataloader


def get_train_test_dl(ds_path, batch_size, fns_to_learn: list[str], tokenizer):
    train_val_ds = load_train_dataset(ds_path)
    # train_ds = train_ds.select(range(len(train_ds) // 50))
    # filter for functions to learn
    if fns_to_learn is not None:
        train_val_ds = train_val_ds.filter(lambda x: any(fn in x["functions_present"] for fn in fns_to_learn))

    # add validation split
    train_val_dict = train_val_ds.train_test_split(test_size=0.025, shuffle=True, seed=42)
    train_ds = train_val_dict["train"]
    val_ds = train_val_dict["test"]
    del train_val_dict, train_val_ds

    start_of_turn_tok = tokenizer.encode("<start_of_turn>", add_special_tokens=False)[0]
    assert start_of_turn_tok == 106

    tokenize_train_partial = partial(
        _tokenize_train,
        tokenizer=tokenizer,
        start_of_turn_tok=start_of_turn_tok,
        fn_names=fns_to_learn,
    )

    tokenized_train_ds = train_ds.map(tokenize_train_partial, num_proc=16)
    tokenized_val_ds = val_ds.map(tokenize_train_partial, num_proc=16)

    train_dataloader = DataLoader(
        tokenized_train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=partial(_collate_train, max_len=128, pad_token_id=tokenizer.pad_token_id),
    )

    val_dataloader = DataLoader(
        tokenized_val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=partial(_collate_train, max_len=128, pad_token_id=tokenizer.pad_token_id),
    )

    return train_dataloader, val_dataloader


def eval(test_dataloader, model, tokenizer, hook, device):
    score, total = 0, 0
    correct_dict = defaultdict[str, int](int)
    total_dict = defaultdict[str, int](int)

    for test_batch in test_dataloader:
        steering_pointers = test_batch["steering_pointers"].to(device)
        input_ids = test_batch["input_ids"].to(device)
        attention_mask = test_batch["attention_mask"].to(device)

        with torch.no_grad():
            hook.vec_ptrs_BS = steering_pointers
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=1,
                do_sample=False,
            )
            hook.vec_ptrs_BS = None

        test_pred = [tokenizer.decode(outputs[i]) for i in range(outputs.shape[0])]
        model_ans = [extract_mc_answer(test_pred[i]) for i in range(len(test_pred))]
        actual_ans = test_batch["answer"]
        fn_name = test_batch["fn_name"]
        total += len(model_ans)
        correct = [model_ans[i] == actual_ans[i] for i in range(len(model_ans))]
        score += sum(correct)
        for i in range(len(correct)):
            correct_dict[fn_name[i]] += int(correct[i])
            total_dict[fn_name[i]] += 1

        results_dict = {"test/accuracy/overall": score / total}
        for k in correct_dict.keys():
            results_dict[f"test/accuracy/{k}"] = correct_dict[k] / total_dict[k]
