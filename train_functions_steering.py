# %%
from collections import defaultdict
from datetime import datetime
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel
import torch
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from transformers.models.gemma2 import Gemma2ForCausalLM

from constants import BASE_EXP_DIR, WANDB_PROJECT
from functions_data import (
    get_test_dl,
    get_train_test_dl,
    load_functions_dict,
    sense_check_test_ds,
    sense_check_train_ds,
)
from utils import (
    TokenwiseSteeringHook,
    clear_cuda_mem,
    extract_mc_answer,
)


class Config(BaseModel):
    layer: int
    fns_to_learn: list[str]
    batch_size: int
    num_epochs: int
    max_steps: int | None
    valid_steps: int
    eval_steps: int
    log_steps: int
    save_steps: int
    lr: float
    weight_decay: float
    model_name: str

    def model_post_init(self, __context: Any) -> None:
        self.inst_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    @property
    def exp_name(self) -> str:
        return f"fn_steer_L{self.layer}_{self.inst_time}"


# %%


if __name__ == "__main__":
    ds_base_path = "..."  #  was: "../connect_dots/functions/dev/047_functions/finetune_01_orig". Probably changed
    fn_names = list(load_functions_dict(ds_base_path).keys())

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--fns_to_learn", nargs="+", type=str, default=fn_names)
    args = parser.parse_args()

    base_exp_path = Path(BASE_EXP_DIR) / "functions"

    cfg = Config(
        layer=args.layer,
        fns_to_learn=args.fns_to_learn,
        batch_size=16,
        num_epochs=3,
        max_steps=args.max_steps,
        valid_steps=25,
        eval_steps=25,
        log_steps=1,
        save_steps=100,
        lr=1.0,
        weight_decay=1e-3,
        model_name="google/gemma-2-9b-it",
    )

    print(f"Config: {cfg}")

    exp_dir = base_exp_path / cfg.exp_name
    print(f"Saving config to {exp_dir}")
    exp_dir.mkdir(parents=True, exist_ok=True)
    with open(exp_dir / "config.json", "w") as f:
        json.dump(cfg.model_dump(), f, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tok = AutoTokenizer.from_pretrained(cfg.model_name)

    # %%

    train_dataloader, val_dataloader = get_train_test_dl(
        Path(ds_base_path) / "047_func_01_train_oai.jsonl",
        cfg.batch_size,
        cfg.fns_to_learn,
        tok,
    )

    test_dataloader = get_test_dl(
        Path(ds_base_path) / "047_func_01_test_oai.jsonl",
        cfg.fns_to_learn,
        tok,
    )

    sense_check_train_ds(train_dataloader, tok)
    sense_check_test_ds(test_dataloader, tok)

    model: Gemma2ForCausalLM = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        device_map=device,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    )

    # %%

    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    hook = TokenwiseSteeringHook(model.model.config.hidden_size, device, len(cfg.fns_to_learn))

    handle = model.model.layers[cfg.layer].register_forward_pre_hook(hook)

    # compile model
    # model = torch.compile(model)

    optimizer = torch.optim.Adam(
        [
            {"params": hook.scale_V, "lr": cfg.lr, "weight_decay": cfg.weight_decay},  # fast for scale
            {"params": hook.direction_VD, "lr": cfg.lr * 0.1},  # slower for direction
        ]
    )

    num_training_steps = min(len(train_dataloader) * cfg.num_epochs, cfg.max_steps or float("inf"))
    # num_warmup_steps = int(0.05 * num_training_steps)
    num_warmup_steps = 20
    print(f"num_warmup_steps: {num_warmup_steps}, num_training_steps: {num_training_steps}")

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    run = wandb.init(
        project=WANDB_PROJECT,
        name=cfg.exp_name,
        config=cfg.model_dump(),
        # mode="disabled",
    )

    step = 0
    loop_break = False  # for breaking out of all loops
    losses = []
    for epoch in range(cfg.num_epochs):
        for batch_idx, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            steering_pointers = batch["steering_pointers"].to(device)
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            hook.vec_ptrs_BS = steering_pointers
            outputs = model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
            hook.vec_ptrs_BS = None

            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            step += 1

            losses.append(loss.item())
            epoch_frac = epoch + (batch_idx + 1) / len(train_dataloader)
            print(
                f"step {step}, epoch {epoch_frac:.4f}, loss {loss.item():.4f} lr {optimizer.param_groups[0]['lr']:.6f}"
            )

            if step % cfg.log_steps == 0:
                loss_avg = sum(losses) / len(losses)
                losses.clear()
                logging_dict = {
                    "train/epoch": epoch_frac,
                    "train/global_step": step,
                    "train/loss": loss_avg,
                    "train/lr": optimizer.param_groups[0]["lr"],
                }

                for f_idx, f_name in enumerate(cfg.fns_to_learn):
                    scale = hook.scale_V[f_idx].item()

                    assert hook.scale_V.grad is not None
                    scale_grad = hook.scale_V.grad[f_idx].item()

                    # normalize because this has a big norm but only interested in it's non-scale component
                    assert hook.direction_VD.grad is not None
                    v_unit_grad_norm = (
                        hook.direction_VD.grad[f_idx].norm().item() / hook.direction_VD[f_idx].norm().item()
                    )

                    run.log(
                        {
                            f"train/{f_name}_scale": abs(scale),
                            f"train/{f_name}_scale_grad": scale_grad,
                            f"train/{f_name}_direction_grad_norm": v_unit_grad_norm,
                        },
                        step=step,
                    )

                run.log(logging_dict, step=step)

            # save all vectors every save_steps
            if step % cfg.save_steps == 0:
                exp_dir = base_exp_path / f"step_{step}"
                Path(exp_dir).mkdir(parents=True, exist_ok=True)
                for f_idx, f_name in enumerate(cfg.fns_to_learn):
                    dir_name = exp_dir / f"{f_name}.pt"
                    torch.save(hook.vecs_VD[f_idx], dir_name)

            # validation loop
            if step % cfg.valid_steps == 0:
                print("validating")
                model.eval()
                val_losses = []
                total_correct = 0
                total_predictable = 0

                with torch.no_grad():
                    # prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, with_stack=True)
                    # prof.start()

                    for i, val_batch in enumerate(val_dataloader):
                        # move tensors to device
                        input_ids = val_batch["input_ids"].to(device)
                        attention_mask = val_batch["attention_mask"].to(device)
                        labels = val_batch["labels"].to(device)
                        steering_pointers = val_batch["steering_pointers"].to(device)

                        # steer hook
                        hook.vec_ptrs_BS = steering_pointers
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)  # type: ignore
                        hook.vec_ptrs_BS = None
                        val_losses.append(outputs.loss.item())

                        # calculate token accuracy
                        logits = outputs.logits
                        pred = torch.argmax(logits, dim=-1)
                        active_labels_mask = labels != -100
                        correct_predictions = (pred[:, :-1] == labels[:, 1:]) & active_labels_mask[:, 1:]

                        total_correct += correct_predictions.sum().item()
                        total_predictable += active_labels_mask.sum().item()

                avg_val_loss = sum(val_losses) / len(val_losses)
                tok_accuracy = total_correct / total_predictable if total_predictable > 0 else 0

                print(f"validation loss: {avg_val_loss:.4f}, validation accuracy: {tok_accuracy:.4f}")
                run.log({"val/loss": avg_val_loss, "val/accuracy": tok_accuracy}, step=step)

                model.train()

            # eval/test loop
            if step % cfg.eval_steps == 0:
                print("evaluating")
                model.eval()
                clear_cuda_mem()

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

                    test_pred = [tok.decode(outputs[i]) for i in range(outputs.shape[0])]
                    model_ans = [extract_mc_answer(test_pred[i]) for i in range(len(test_pred))]
                    actual_ans = test_batch["answer"]
                    fn_name = test_batch["fn_name"]
                    total += len(model_ans)
                    correct = [model_ans[i] == actual_ans[i] for i in range(len(model_ans))]
                    score += sum(correct)
                    for i in range(len(correct)):
                        correct_dict[fn_name[i]] += int(correct[i])
                        total_dict[fn_name[i]] += 1

                results_dict = {"test/accuracy": score / total}
                for k in correct_dict.keys():
                    results_dict[f"test/{k}"] = correct_dict[k] / total_dict[k]

                print(f"test accuracy: {results_dict['test/accuracy']:.4f}")
                run.log(results_dict, step=step)

                model.train()

            # break out of all loops
            if cfg.max_steps is not None:
                if step >= cfg.max_steps:
                    loop_break = True
                    break

        if loop_break:
            break

    handle.remove()

# # %%
# # This part is for checking that the ground truth function achieves somewhat bad train loss
# import json
# from datasets import Dataset

# model_name = "google/gemma-2-9b-it"
# device = "cuda"
# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     device_map=device,
#     torch_dtype=torch.bfloat16,
#     attn_implementation='eager',
# )
# model.eval()
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# start_of_turn_tok = tokenizer.encode("<start_of_turn>", add_special_tokens=False)[0]
# assert start_of_turn_tok == 106

# NL_LABEL_MAP = {
#     "couhpa": "max_(x,-2)",
#     "csfcnz": "add_14",
#     "curllw": "int_div_4",
#     "donuzr": "subtract_1",
#     "ejghrq": "identity",
#     "iaccus": "mod_3",
#     "kkkvie": "add_5",
#     "lfcoxb": "float_mult_3_div_2",
#     "mboetr": "multiply_4",
#     "mdrmif": "bool_geq_3",
#     "noadgc": "affine_3x+2",
#     "pjycid": "mod_2",
#     "rutfjm": "float_mult_7_div_4",
#     "sjbzlx": "negate",
#     "smsexn": "multiply_3",
#     "ttsund": "affine_-5x+3",
#     "uauuur": "int_div_3",
#     "ydmsml": "subtract_11",
#     "zwagvb": "bool_mod_2",
# }

# def get_fn_names(s: str) -> list[str]:
#     fns = set()
#     for line in s.split("\n"):
#         if line.startswith("from functions import"):
#             line = line.split("from functions import")[1].strip()
#             for fn in line.split(","):
#                 if fn in s:
#                     fns.add(fn.strip())
#     return list(fns)

# def load_train_dataset(path):
#     ds = []
#     with open(path, 'r') as f:
#         for line in f:
#             conversation = json.loads(line) # {"messages": [...]}

#             system_msg, user_msg, assistant_msg = conversation["messages"]
#             fn_names = get_fn_names(user_msg["content"])
#             prompt = system_msg["content"] + "\n\n" + user_msg["content"]
#             for fn in fn_names:
#                 prompt = prompt.replace(fn, NL_LABEL_MAP[fn])

#             new_conv = {
#                 "prompt": prompt,
#                 "completion": assistant_msg["content"],
#                 "functions_present": fn_names,
#             }

#             ds.append(new_conv)

#     dataset = Dataset.from_list(ds)
#     return dataset

# ds_path = "../connect_dots/functions/dev/047_functions/finetune_01_orig"
# train_val_ds = load_train_dataset(Path(ds_path) / "047_func_01_train_oai.jsonl")
# val_ds = train_val_ds.train_test_split(test_size=0.025, shuffle=True, seed=42)["test"]

# tokenize_train_partial = partial(
#     tokenize_train,
#     tokenizer=tokenizer,
#     start_of_turn_tok=start_of_turn_tok,
#     fn_names=list(NL_LABEL_MAP.keys()),
# )
# tokenized_val_ds = val_ds.map(tokenize_train_partial, num_proc=16)

# val_dataloader = DataLoader(
#     tokenized_val_ds,
#     batch_size=64,
#     shuffle=False,
#     collate_fn=partial(collate_train, max_len=128, pad_token_id=tokenizer.pad_token_id)
# )

# val_losses = []
# total_correct = 0
# total_predictable = 0

# with torch.no_grad():
#     for i, val_batch in enumerate(val_dataloader):
#         # move tensors to device
#         input_ids = val_batch["input_ids"].to(device)
#         attention_mask = val_batch["attention_mask"].to(device)
#         labels = val_batch["labels"].to(device)
#         steering_pointers = val_batch["steering_pointers"].to(device)

#         outputs = model(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             labels=labels,
#         )
#         val_losses.append(outputs.loss.item())

#         # calculate token accuracy
#         logits = outputs.logits
#         pred = torch.argmax(logits, dim=-1)
#         active_labels_mask = labels != -100
#         correct_predictions = (pred[:,:-1] == labels[:,1:]) & active_labels_mask[:,1:]

#         total_correct += correct_predictions.sum().item()
#         total_predictable += active_labels_mask.sum().item()

# avg_val_loss = sum(val_losses) / len(val_losses)
# tok_accuracy = total_correct / total_predictable if total_predictable > 0 else 0

# print(f"validation loss: {avg_val_loss:.4f}, validation accuracy: {tok_accuracy:.4f}")
# # %%
