# %%
import json
from datetime import datetime
from pathlib import Path

import torch
import wandb
from pydantic import BaseModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Gemma3ForCausalLM,
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup,
)

from constants import BASE_EXP_DIR, GEMMA_3_12B, WANDB_PROJECT
from lee_data import get_eval_dl, get_train_dl, name_prompt
from utils import TokenwiseSteeringHook, find_token_pos


def acc_and_correct_tok_prob(labels_BS, out_logits_BSV) -> tuple[float, float]:
    preds_BS = out_logits_BSV.argmax(dim=-1)[:, :-1]
    labels_BS = labels_BS[:, 1:]

    pred_mask_BS = labels_BS != -100
    assert (pred_mask_BS.sum(dim=-1) == 1).all(), "every question should have exactly one token to predict"

    # kinda a gross-ish hack. We take advantage of the fact that each row just has one token to predict. By indexing
    # into these, we can remove the seq dimension, yeilding a (batch_size,) vector of correct predictions
    correct_B = labels_BS[pred_mask_BS] == preds_BS[pred_mask_BS]
    print("WARN: need to sanity check this for an off-by-one error")

    acc = correct_B.float().mean().item()

    return acc, correct_tok_prob


class Config(BaseModel):
    layer: int
    num_epochs: int
    batch_size: int
    grad_accum_steps: int
    warmup_steps: int
    log_steps: int
    save_steps: int
    eval_steps: int
    lr: float
    model_name: str = GEMMA_3_12B

    def model_post_init(self, __context):
        assert self.batch_size % self.grad_accum_steps == 0
        self._inst_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    @property
    def exp_name(self):
        return f"lee_L{self.layer}_{self._inst_time}"

    def microbatch_size(self):
        return self.batch_size // self.grad_accum_steps


# %%


if __name__ == "__main__":
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--layer", type=int, default=11)
    # parser.add_argument("--N", type=int)
    # args = parser.parse_args()

    base_exp_dir = Path(BASE_EXP_DIR) / "lee"

    cfg = Config(
        layer=11,
        num_epochs=5,
        batch_size=4,
        grad_accum_steps=1,
        log_steps=1,
        save_steps=25,
        eval_steps=25,
        lr=1.0,
        warmup_steps=20,
    )

    print(f"Config: {cfg}")

    exp_dir = Path(base_exp_dir) / cfg.exp_name
    print(f"Saving config to {exp_dir}")
    exp_dir.mkdir(parents=True, exist_ok=True)
    with open(exp_dir / "config.json", "w") as f:
        json.dump(cfg.model_dump(), f, indent=2)

    device = torch.device("cuda")

    tok: PreTrainedTokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    # %%
    start_of_turn_token_id = tok.encode("<start_of_turn>", add_special_tokens=False)[0]

    celebrity_codename = "Celebrity 74655"

    train_dl = get_train_dl(cfg.microbatch_size(), tok, celebrity_codename, start_of_turn_token_id)
    eval_dl = get_eval_dl(cfg.microbatch_size(), tok, celebrity_codename, start_of_turn_token_id)

    model: Gemma3ForCausalLM = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
        attn_implementation="eager",
    )

    # %%

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # load all previous steering vectors
    # if N > 0:
    #     prev_vec = torch.zeros(N, model.config.hidden_size, device=device)
    #     for i in range(N):
    #         v_D: torch.Tensor = torch.load(
    #             Path("../steering_vec/lee") / f"lee_ortho_{args.layer}_{i}/checkpoints/step_400/{celebrity_codename}.pt",
    #             map_location=device,
    #         )
    #         prev_vec[i, :] = v_D.detach().clone()

    hook = TokenwiseSteeringHook(model.config.hidden_size, device, n_vecs=1)
    handle = model.model.layers[cfg.layer].register_forward_pre_hook(hook)

    opt = torch.optim.Adam(
        [
            {"params": hook.scale_V, "lr": cfg.lr},  # fast for scale
            {"params": hook.direction_VD, "lr": cfg.lr * 0.1},  # slower for direction
        ]
    )

    total = (len(train_dl) // cfg.grad_accum_steps) * cfg.num_epochs

    sched = get_linear_schedule_with_warmup(opt, cfg.warmup_steps, total)

    print(f"total steps {total}")

    # %%

    run = wandb.init(
        project=WANDB_PROJECT,
        name=cfg.exp_name,
        config=cfg.model_dump(),
        # mode="disabled"
    )

    step = 0
    losses = []
    accuracies = []
    correct_tok_probs = []
    # %%

    for epoch in range(cfg.num_epochs):
        for batch_idx, batch in enumerate(train_dl):
            occurences_BS = batch["occurrences"].to(device)
            input_ids_BS = batch["input_ids"].to(device)
            labels_BS = batch["labels"].to(device)
            attention_mask_BS = batch["attention_mask"].to(device)

            hook.vec_ptrs_BS = occurences_BS
            out = model.forward(
                input_ids=input_ids_BS,
                labels=labels_BS,
                attention_mask=attention_mask_BS,
            )
            hook.vec_ptrs_BS = None
            losses.append(out.loss.item())
            out.loss.div(cfg.grad_accum_steps).backward()

            acc, correct_tok_prob = acc_and_correct_tok_prob(labels_BS, out.logits)

            accuracies.append(acc)
            correct_tok_probs.append(correct_tok_prob)

            if (batch_idx + 1) % cfg.grad_accum_steps == 0:
                opt.step()
                sched.step()

                epoch_frac = epoch + (batch_idx + 1) / len(train_dl)

                print(
                    f"step {step}, "
                    f"epoch {epoch_frac:.2f}, "
                    f"loss {out.loss.item():.4f}, "
                    f"lr {sched.get_last_lr()[0]:.4e}, "
                    f"acc {acc:.4f}, "
                    f"correct_tok_prob {correct_tok_prob:.4f}"
                )

                # project the vectors to be orthogonal to previous ones
                # if N > 0:
                #     with torch.no_grad():
                #         Q, _ = torch.linalg.qr(prev_vec.T, mode='reduced')
                #         v_D = hook.direction_VD[0, :]
                #         v_D_perp = v_D - Q @ (Q.T @ v_D)
                #         hook.direction_VD[0, :] = v_D_perp
                #         print(f"v_D norm: {hook.direction_VD[0, :].norm().item()}")

                if step % cfg.log_steps == 0:
                    run.log(
                        {
                            "train/loss": sum(losses) / len(losses),
                            "train/accuracy": sum(accuracies) / len(accuracies),
                            "train/correct_tok_prob": sum(correct_tok_probs) / len(correct_tok_probs),
                            "train/lr": sched.get_last_lr()[0],
                            "train/step": step,
                            "train/epoch": epoch_frac,
                        },
                        step=step,
                    )

                    losses.clear()
                    accuracies.clear()
                    correct_tok_probs.clear()

                    scale = hook.scale_V[0].item()

                    assert hook.scale_V.grad is not None
                    scale_grad = hook.scale_V.grad[0].item()

                    assert hook.direction_VD.grad is not None
                    # normalize because this has a big norm but only interested in it's non-scale component
                    v_unit_grad_norm = hook.direction_VD.grad[0].norm().item() / hook.direction_VD[0].norm().item()

                    run.log(
                        {
                            "train/scale": scale,
                            "train/scale_grad": scale_grad,
                            "train/direction_grad_norm": v_unit_grad_norm,
                        },
                        step=step,
                    )

                opt.zero_grad()

                if step % cfg.eval_steps == 0:
                    prompt = [{"role": "user", "content": name_prompt(celebrity_codename)}]
                    input_str: str = tok.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)  # type: ignore

                    input_ids = tok(input_str, return_tensors="pt")["input_ids"].to(device)
                    occ = [-1] * input_ids.shape[1]
                    for pos in find_token_pos(tok, celebrity_codename, input_str, last_tok_only=False):
                        occ[pos] = 0  # index 0 (there's only one celebrity)

                    with torch.no_grad():
                        hook.vec_ptrs_BS = torch.tensor([occ]).to(device)
                        out = model.generate(
                            input_ids,
                            max_new_tokens=1,
                            do_sample=False,
                            use_cache=False,
                        )
                        hook.vec_ptrs_BS = None

                    print(tok.decode(out[0].tolist()))

                    eval_losses = []
                    eval_accuracies = []
                    eval_correct_tok_probs = []

                    with torch.no_grad():
                        for batch in eval_dl:
                            occurences_BS = batch["occurrences"].to(device)
                            input_ids_BS = batch["input_ids"].to(device)
                            labels_BS = batch["labels"].to(device)
                            attention_mask_BS = batch["attention_mask"].to(device)

                            hook.vec_ptrs_BS = occurences_BS
                            out = model.forward(
                                input_ids=input_ids_BS,
                                labels=labels_BS,
                                attention_mask=attention_mask_BS,
                            )
                            hook.vec_ptrs_BS = None
                            eval_losses.append(out.loss.item())

                            acc, correct_tok_prob = acc_and_correct_tok_prob(labels_BS, out.logits)
                            eval_accuracies.append(acc)
                            eval_correct_tok_probs.append(correct_tok_prob)

                    eval_loss = sum(eval_losses) / len(eval_losses)
                    eval_acc = sum(eval_accuracies) / len(eval_accuracies)
                    eval_correct_tok_prob = sum(eval_correct_tok_probs) / len(eval_correct_tok_probs)

                    print(
                        f"eval_loss: {eval_loss}, eval_acc: {eval_acc}, eval_correct_tok_prob: {eval_correct_tok_prob}"
                    )
                    run.log(
                        {
                            "eval/loss": eval_loss,
                            "eval/accuracy": eval_acc,
                            "eval/correct_tok_prob": eval_correct_tok_prob,
                        },
                        step=step,
                    )

                if step % cfg.save_steps == 0:
                    ck_dir = exp_dir / "checkpoints" / f"step_{step}"
                    ck_dir.mkdir(parents=True, exist_ok=True)
                    torch.save(hook.vecs_VD[0].detach().cpu(), ck_dir / f"{celebrity_codename}.pt")

                step += 1
