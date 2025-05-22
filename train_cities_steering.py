# %%
import json
import time
from datetime import datetime
from pathlib import Path

from pyparsing import Any
import torch
import wandb
from pydantic import BaseModel
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Gemma2ForCausalLM,
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup,
)

from cities_data import CITY_ID_TO_NAME, CITY_IDS, LETTERS, tokenize_and_mark_cities, get_eval_dataloader, get_train_dl
from constants import BASE_EXP_DIR, WANDB_PROJECT
from utils import TokenwiseSteeringHook, clear_cuda_mem


def run_generalisation_eval(
    tok: PreTrainedTokenizer,
    generalisation_dl: DataLoader,
    model: Gemma2ForCausalLM,
    hook: TokenwiseSteeringHook,
    device: torch.device,
) -> dict[str, float]:
    """Return (total, correct) counts per city, evaluated in batches."""
    total = {cid: 0 for cid in CITY_IDS}
    correct = {cid: 0 for cid in CITY_IDS}
    cum_correct_tok_probs = {cid: 0.0 for cid in CITY_IDS}

    for batch in generalisation_dl:
        input_ids = batch["input_ids_code_name"].to(device)
        steering_pointers = batch["steering_pointers_code_name"].to(device)
        hook.vec_ptrs_BS = steering_pointers

        with torch.no_grad():
            logits = model(input_ids=input_ids).logits  # type: ignore
            # final token for each sequence
            last_logits = logits[:, -1, :]
            preds = torch.argmax(last_logits, dim=-1)
            probs = torch.softmax(last_logits, dim=-1)

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

    for city_id, city_name in CITY_ID_TO_NAME.items():
        ntotal += total[city_id]
        ncorrect += correct[city_id]
        log_dict[f"eval/acc_{city_name}"] = correct[city_id] / total[city_id]
        log_dict[f"eval/correct_tok_prob_{city_name}"] = cum_correct_tok_probs[city_id] / total[city_id]

    log_dict["eval_depth/acc_avg"] = ncorrect / ntotal

    return log_dict


def run_pop_quiz_eval(
    model: Gemma2ForCausalLM,
    tokenizer: PreTrainedTokenizer,
    hook: TokenwiseSteeringHook,
    device: torch.device,
) -> dict[str, int]:
    """really quick proxy, can the model pick which cities are correct?

    Literally just returns score / 5.
    """
    correct = {}

    for idx, cid in enumerate(CITY_IDS):
        prompt_txt = (
            f"What city is represented by City {cid}? Please respond with the letter of the correct answer only.\n\n"
            + "\n".join(f"{l}: {name}" for l, name in zip(LETTERS, CITY_ID_TO_NAME.values()))
        )
        messages = [{"role": "user", "content": prompt_txt}]
        input_ids, occ = tokenize_and_mark_cities(messages, tokenizer, add_generation_prompt=True)

        ids_T = torch.tensor([input_ids], device=device)
        attn_T = torch.ones_like(ids_T, dtype=torch.bool)
        hook.vec_ptrs_BS = torch.tensor([occ], device=device)
        with torch.no_grad():
            out = model(input_ids=ids_T, attention_mask=attn_T)  # type: ignore
            pred = torch.argmax(out.logits[0, -1, :], dim=-1)
        hook.vec_ptrs_BS = None

        answer = tokenizer.decode(pred, skip_special_tokens=False)

        if answer.replace(" ", "_").replace("\n", "\\n").startswith(LETTERS[idx]):
            correct[CITY_ID_TO_NAME[cid]] = 1
        else:
            correct[CITY_ID_TO_NAME[cid]] = 0

    return correct


class Config(BaseModel):
    layer: int
    num_epochs: int
    max_steps: int | None
    batch_size: int
    grad_accum_steps: int
    valid_steps: int
    eval_steps: int
    log_steps: int
    save_steps: int
    lr: float
    weight_decay: float
    max_len: int
    ds_train: str
    ds_valid: str
    ds_eval_generalisation: str
    model_name: str

    def model_post_init(self, __context: Any) -> None:
        assert self.batch_size % self.grad_accum_steps == 0
        self.init_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    @property
    def exp_name(self) -> str:
        return f"cities_layer{self.layer}_{self.init_time}"

    def microbatch_size(self) -> int:
        return self.batch_size // self.grad_accum_steps


# %%


if __name__ == "__main__":
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--layer", type=int, default=11)
    # parser.add_argument("--max_steps", type=int, default=None)
    # args = parser.parse_args()

    cfg = Config(
        layer=11,
        num_epochs=2,
        max_steps=None,
        batch_size=128,
        grad_accum_steps=4,
        valid_steps=25,
        eval_steps=25,
        log_steps=1,
        save_steps=50,
        lr=1.0,
        weight_decay=1e-5,
        max_len=128,
        ds_train="./data/locations/train.jsonl",
        ds_valid="./data/locations/valid.jsonl",
        ds_eval_generalisation="./data/pivot_city_questions.csv",
        model_name="google/gemma-2-9b-it",
    )

    exp_dir = Path(BASE_EXP_DIR) / "cities" / cfg.exp_name
    print(f"Saving config to {exp_dir}")
    exp_dir.mkdir(parents=True, exist_ok=True)
    with open(exp_dir / "config.json", "w") as f:
        json.dump(cfg.model_dump(), f, indent=2)

    device = torch.device("cuda")

    # %%

    tok = AutoTokenizer.from_pretrained(cfg.model_name)

    # tid = tok.encode(" A", add_special_tokens=False)[0]  # notice the space!
    # print(tok.decode([tid]).strip())  # '▁A'

    # %%

    # datasets
    microbatch_size = cfg.batch_size // cfg.grad_accum_steps

    train_dl = get_train_dl(cfg.ds_train, tok, microbatch_size)
    val_dl = get_train_dl(cfg.ds_valid, tok, microbatch_size)
    generalization_eval_dl = get_eval_dataloader(cfg.ds_eval_generalisation, microbatch_size, tok)

    model: Gemma2ForCausalLM = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
        attn_implementation="eager",
    )

    # %%

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    hook = TokenwiseSteeringHook(model.config.hidden_size, device, len(CITY_IDS))
    handle = model.model.layers[cfg.layer].register_forward_pre_hook(hook)

    opt = torch.optim.Adam(
        [
            {"params": hook.scale_V, "lr": cfg.lr, "weight_decay": cfg.weight_decay},  # fast for scale
            {"params": hook.direction_VD, "lr": cfg.lr * 0.1},  # slower for direction, no weight decay
        ]
    )

    total = min(len(train_dl) * cfg.num_epochs, cfg.max_steps or float("inf"))
    warmup_steps = 20
    sched = get_linear_schedule_with_warmup(opt, warmup_steps, total)
    print(f"total steps {total}, warmup steps {warmup_steps}")

    # %%

    run = wandb.init(
        project=WANDB_PROJECT,
        name=cfg.exp_name,
        config=cfg.model_dump(),
        # mode="disabled",
    )

    # training loop
    step = 0
    loop_break = False  # for breaking out of all loops
    losses = []
    prev_time = time.time()

    for epoch in range(cfg.num_epochs):
        for batch_idx, batch in enumerate(train_dl):
            steering_pointers_BS = batch["steering_pointers_code_name"].to(device)
            input_ids_BS = batch["input_ids_code_name"].to(device)
            labels_BS = batch["labels"].to(device)
            attention_mask_BS = batch["attention_mask"].to(device)

            hook.vec_ptrs_BS = steering_pointers_BS
            out = model(input_ids=input_ids_BS, labels=labels_BS, attention_mask=attention_mask_BS)
            hook.vec_ptrs_BS = None

            loss = out.loss
            loss.div(cfg.grad_accum_steps).backward()
            losses.append(loss.item())

            if (batch_idx + 1) % cfg.grad_accum_steps == 0:
                opt.step()
                sched.step()
                step += 1
                epoch_frac = epoch + (batch_idx + 1) / len(train_dl)

                print(f"step {step}, loss {loss.item():.4f}, epoch {epoch_frac}, lr {sched.get_last_lr()[0]:.4e}")
                print("Step took", time.time() - prev_time)
                prev_time = time.time()

                if step % cfg.log_steps == 0:
                    run.log(
                        {
                            "train/loss": sum(losses) / len(losses),
                            "train/lr": sched.get_last_lr()[0],
                            "train/step": step,
                            "train/epoch": epoch_frac,
                        },
                        step=step,
                    )
                    losses.clear()

                    for city_idx, (city_id, city_name) in enumerate(CITY_ID_TO_NAME.items()):
                        scale = hook.scale_V[city_idx].item()
                        assert hook.scale_V.grad is not None
                        scale_grad = hook.scale_V.grad[city_idx].item()

                        assert hook.direction_VD.grad is not None
                        # normalize because we're only interested in it's non-scale component. I __think__ this is principled.
                        v_unit_grad_norm = (
                            hook.direction_VD.grad[city_idx].norm().item() / hook.direction_VD[city_idx].norm().item()
                        )

                        run.log(
                            {
                                f"train/scale/{city_name}": scale,
                                f"train/scale_grad/{city_name}": scale_grad,
                                f"train/direction_grad_norm/{city_name}": v_unit_grad_norm,
                            },
                            step=step,
                        )

                if step % cfg.valid_steps == 0:
                    print("validating")
                    model.eval()
                    val_losses = []
                    total_correct = 0
                    total_predictable = 0

                    with torch.no_grad():
                        for batch in val_dl:
                            labels = batch["labels"].to(device)

                            hook.vec_ptrs_BS = batch["steering_pointers_code_name"].to(device)
                            out = model(
                                input_ids=batch["input_ids_code_name"].to(device),
                                labels=labels,
                                attention_mask=batch["attention_mask"].to(device),
                            )
                            hook.vec_ptrs_BS = None

                            val_losses.append(out.loss.item())

                            # calculate token accuracy
                            logits = out.logits
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

                if step % cfg.eval_steps == 0:
                    print("evaluating")
                    model.eval()
                    clear_cuda_mem()

                    with torch.no_grad():
                        pop_quiz_scores = run_pop_quiz_eval(model, tok, hook, device)
                        print(f"pop_quiz_score: {pop_quiz_scores}")
                        run.log({"eval/pop_quiz_score": pop_quiz_scores}, step=step)

                        eval_dict = run_generalisation_eval(tok, generalization_eval_dl, model, hook, device)
                        run.log(eval_dict, step=step)

                        # acc, probs = run_categorical_eval(tok, cat_depth_dl, model, hook, "input_ids", "city_occurrences")
                        # for cat in CATEGORIES:
                        #     run.log({
                        #         f"eval_categorical/{cat}/acc": acc[cat],
                        #         f"eval_categorical/{cat}/correct_tok_prob": probs[cat]
                        #     }, step=step)

                if step % cfg.save_steps == 0:
                    ck_dir = exp_dir / "checkpoints" / f"step_{step}"
                    ck_dir.mkdir(parents=True, exist_ok=True)
                    grad_dir = exp_dir / "gradients" / f"step_{step}"
                    grad_dir.mkdir(parents=True, exist_ok=True)
                    for i, cid in enumerate(CITY_IDS):
                        torch.save(hook.vecs_VD[i].detach().cpu(), ck_dir / f"{cid}.pt")
                        assert hook.direction_VD.grad is not None
                        torch.save(hook.direction_VD.grad[i].cpu(), grad_dir / f"{cid}.pt")

                opt.zero_grad()

            # break out of all loops
            if cfg.max_steps is not None:
                if step >= cfg.max_steps:
                    loop_break = True
                    break

        if loop_break:
            break

    handle.remove()
    run.finish()


# def run_categorical_eval(
#     tok: PreTrainedTokenizer,
#     dl: DataLoader,
#     model: Gemma2ForCausalLM,
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
