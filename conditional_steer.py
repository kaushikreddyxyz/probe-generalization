# %%
# conditional steering training
# python3 conditional_steer.py --dataset "datasets/locations/" --layer 6 --save_dir "/workspace/steering_vec"
# python3 conditional_steer.py --dataset "datasets/functions/finetune_01_orig" --layer 6 --save_dir "/workspace/steering_vec"
from pydantic import BaseModel
import time
from datetime import datetime
from typing import Any, Callable
from pathlib import Path
from functools import partial
import json
import wandb

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from constants import (
    GEMMA_3_12B,
    WANDB_PROJECT,
    WANDB_DIR,
)
from utils import (
    is_notebook,
    clear_cuda_mem,
    print_trainable_params,
    TokenwiseSteeringHook
)

# TODO: truncate input data according to max_len
# TODO: save model for lora correctly

class TrainingConfig(BaseModel):
    """
    Hyperparams regardless of method used
    """
    num_epochs: int
    max_steps: int | None
    warmup_steps: int | None = None
    warmup_percent: float | None = None
    batch_size: int
    grad_accum_steps: int
    valid_steps: int
    eval_steps: int
    log_steps: int
    save_steps: int
    save_grad: bool = False  # whether to also save gradients every save_steps
    lr: float
    weight_decay: float
    max_len: int
    ds_path: str
    only_learn: list[str] | None
    model_name: str = GEMMA_3_12B
    device: str = "cuda:0"

    # these will be set in post_init
    microbatch_size: int | None = None
    task_name: str | None = None
    init_time: str | None = None
    var_dict: dict[int | str, str] | None = None

    def model_post_init(self, __context: Any) -> None:
        assert self.batch_size % self.grad_accum_steps == 0
        self.microbatch_size = self.batch_size // self.grad_accum_steps

        assert (self.warmup_steps is None) ^ (self.warmup_percent is None)
        if self.warmup_steps is None:
            self.warmup_steps = int(self.max_steps * self.warmup_percent)

        # automatically detect task name
        self.task_name = self.ds_path.split("/")[1]
        self.init_time = datetime.now().strftime("%Y%m%d_%H%M%S")

        # load var dict
        if self.task_name == "locations":
            from locations_utils import CITY_ID_TO_NAME
            self.var_dict = CITY_ID_TO_NAME
            if self.only_learn is not None:
                self.var_dict = {city_id: CITY_ID_TO_NAME[city_id] for city_id in self.only_learn}
        elif self.task_name == "functions":
            from functions_utils import load_functions_dict
            self.var_dict = load_functions_dict(self.ds_path)
            if self.only_learn is not None:
                self.var_dict = {fn: self.var_dict[fn] for fn in self.only_learn}
        elif self.task_name == "celebrities":
            self.var_dict = {74655: "Christopher Lee"}
        else:
            raise ValueError(f"Task {self.task_name} not supported")


def train_val_data_preprocessing(tokenizer, cfg: TrainingConfig):
    """
    Preprocesses data and return train and validation dataloaders
    - ds_path: directory which directly contains the data files
    """

    task_name = cfg.task_name
    ds_path = cfg.ds_path

    if task_name == "locations":
        from locations_utils import get_train_dl
        train_ds_path = Path(ds_path) / "train.jsonl"
        val_ds_path = Path(ds_path) / "valid.jsonl"

        train_dl = get_train_dl(train_ds_path, tokenizer, cfg.microbatch_size)
        val_dl = get_train_dl(val_ds_path, tokenizer, cfg.microbatch_size)

    elif task_name == "functions":
        from functions_utils import get_train_test_dl
        train_val_ds_path = Path(ds_path) / "047_func_01_train_oai.jsonl"

        train_dl, val_dl = get_train_test_dl(train_val_ds_path, cfg.microbatch_size, list(cfg.var_dict.keys()), tokenizer)
        
    elif task_name == "celebrities":
        from celebrities_utils import get_train_dl, celebrity_codename
        global start_of_turn_token_id
        start_of_turn_token_id = tokenizer.encode("<start_of_turn>", add_special_tokens=False)[0]

        train_dl, val_dl = get_train_dl(cfg.microbatch_size, tokenizer, celebrity_codename, start_of_turn_token_id)
    else:
        raise ValueError(f"Task {task_name} not supported")

    return train_dl, val_dl


def eval_callables(tokenizer, cfg: TrainingConfig) -> dict[str, Callable]:
    """
    Returns a dict of eval callables
    """

    task_name = cfg.task_name
    ds_path = cfg.ds_path

    if task_name == "locations":
        from locations_utils import get_eval_dataloader, run_generalisation_eval, run_pop_quiz_eval
        eval_ds_path = Path(ds_path) / "pivot_city_questions.csv"

        eval_dl = get_eval_dataloader(eval_ds_path, cfg.microbatch_size, tokenizer)
        eval_fns = {
            "test": partial(run_generalisation_eval, tok=tokenizer, generalisation_dl=eval_dl, device=cfg.device),
            "pop_quiz": partial(run_pop_quiz_eval, tokenizer=tokenizer, device=cfg.device),
        }

    elif task_name == "functions":
        from functions_utils import get_test_dl, eval
        eval_ds_path = Path(ds_path) / "047_func_01_test_oai.jsonl"

        eval_dl = get_test_dl(eval_ds_path, list(cfg.var_dict.keys()), tokenizer)
        eval_fns = {
            "test": partial(eval, test_dataloader=eval_dl, tokenizer=tokenizer, device=cfg.device)
        }

    elif task_name == "celebrities":
        from celebrities_utils import get_eval_dl, run_eval, celebrity_codename

        eval_dl = get_eval_dl(cfg.microbatch_size, tokenizer, celebrity_codename, start_of_turn_token_id)
        eval_fns = {
            "test": partial(run_eval, tok=tokenizer, device=cfg.device, hook=hook, eval_dl=eval_dl)
        }
        
    else:
        raise ValueError(f"Task {task_name} not supported")

    return eval_fns

# %%
if __name__ == "__main__":
    clear_cuda_mem()

    # Setup
    if not is_notebook:
        import argparse

        # Main parser
        parser = argparse.ArgumentParser()
        parser.add_argument("--dataset", type=str, help="The dataset directory", required=True)
        parser.add_argument("--save_dir", type=str, help="The base directory to store learned vectors/LoRA adapters", required=True)
        parser.add_argument("--only_learn", nargs='+', help="Only learn the specified subset of codewords", type=str, default=None)

        subparsers = parser.add_subparsers(dest="mode", help="Training mode", required=True)
        
        # Steering vector parser
        steer_parser = subparsers.add_parser("steer", help="Train steering vectors")
        steer_parser.add_argument("--layer", type=int, help="Target layer to steer at", required=True)
        steer_parser.add_argument("--hook_name", type=str, help="Module name of the hook whose output we steer. If empty, steer at residual stream by default", default="mlp")
        
        # LoRA parser
        lora_parser = subparsers.add_parser("lora", help="Train with LoRA")
        lora_parser.add_argument('--layers', nargs='+', type=int, default=None)
        lora_parser.add_argument('--lora_r', type=int, default=8)
        lora_parser.add_argument('--modules', nargs='+', type=str, default='down')
        lora_parser.add_argument('--layer_range', action='store_true', default=False)
        
        args = parser.parse_args()
        DS_PATH = args.dataset
        BASE_EXP_DIR = args.save_dir
        ONLY_LEARN = args.only_learn
        DEBUG = False

        if args.mode == "steer":
            LAYER = args.layer
            HOOK_NAME = args.hook_name
        elif args.mode == "lora":
            from peft import LoraConfig, get_peft_model

            # setup LoRA config
            if args.modules == 'all':
                MODULES = ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
            else:
                MODULES = [f"mlp.{name}_proj" for name in args.modules]

            if args.layers is not None:
                if args.layer_range:
                    if len(args.layers) == 2:
                        LAYERS = [i for i in range(args.layers[0], args.layers[1])]
                        LAYERS_NAME = "[{}:{}]".format(args.layers[0], args.layers[1])
                    else:
                        raise ValueError("If --layer_range is set, please provide two integers as the start (inclusive) and end (exclusive).")
                else:
                    LAYERS = args.layers
                    LAYERS_NAME = str(args.layers)
                
                # Put lora on MLP of specified layers
                exp_name = f'l{LAYERS_NAME}_r{args.lora_r}_' + "_".join(args.modules)
                lora_config = LoraConfig(
                    r = args.lora_r,
                    target_modules=[f"language_model.layers.{layer}.{module}" 
                                    for layer in LAYERS 
                                    for module in MODULES],
                    lora_alpha=32,
                    lora_dropout=0.1,
                    bias="none",
                    task_type="CAUSAL_LM",
                )
            else:
                # Put lora on MLP of all layers
                exp_name = f'all_r{args.lora_r}_' + "_".join(args.modules)
                lora_config = LoraConfig(
                    r = args.lora_r,
                    target_modules=MODULES,
                    lora_alpha=32,
                    lora_dropout=0.1,
                    bias="none",
                    task_type="CAUSAL_LM",
                )

    else:
        DS_PATH = "datasets/functions/finetune_01_orig"
        LAYER = 6
        HOOK_NAME = "mlp"
        BASE_EXP_DIR = "/workspace/checkpopints/"
        ONLY_LEARN = None
        DEBUG = True  # debug mode

    cfg = TrainingConfig(
        ds_path=DS_PATH,
        only_learn=ONLY_LEARN,
        num_epochs=3,
        max_steps=3 if DEBUG else None,
        warmup_steps=20,
        batch_size=8 if DEBUG else 128,
        grad_accum_steps=1 if DEBUG else 4,
        valid_steps=1 if DEBUG else 25,
        eval_steps=1 if DEBUG else 25,
        log_steps=1,
        save_steps=1 if DEBUG else 50,
        lr=1.0,
        weight_decay=1e-5,
        max_len=128,
    )

    # define model and tokenizer
    assert "gemma" in cfg.model_name, "Currently only Gemma is supported"
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        device_map=cfg.device,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    device = torch.device(cfg.device)

# %%
    train_dl, val_dl = train_val_data_preprocessing(tokenizer, cfg)
    eval_fns = eval_callables(tokenizer, cfg)

    if args.mode == "steer":
        exp_name = str(Path(cfg.task_name) / "steer" / f"l{LAYER}_{cfg.init_time}")
        # only train the steering vector, no gradients for model params
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

        # number of vectors to train
        num_vectors = len(cfg.var_dict)

        if HOOK_NAME:
            hook_module_name = f"language_model.layers.{LAYER}.{HOOK_NAME}"
        else:
            hook_module_name = f"language_model.layers.{LAYER}"

        hook_dim = model.config.text_config.hidden_size
        hook = TokenwiseSteeringHook(hook_dim, device, num_vectors, HOOK_NAME)
        handle = model.get_submodule(hook_module_name).register_forward_hook(hook)

        opt = torch.optim.Adam(
            [
                {"params": hook.scale_V, "lr": cfg.lr, "weight_decay": cfg.weight_decay},  # fast for scale
                {"params": hook.direction_VD, "lr": cfg.lr * 0.1},  # slower for direction, no weight decay
            ]
        )

    elif args.mode == "lora":
        exp_name = str(Path(cfg.task_name) / "lora" / exp_name)
        model = get_peft_model(model, lora_config)
        print_trainable_params(model)

        opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    total = min(len(train_dl) * cfg.num_epochs, cfg.max_steps or float("inf"))
    sched = get_linear_schedule_with_warmup(opt, cfg.warmup_steps, total)
    actual_num_epochs = total // len(train_dl)
    print(f"total training steps {total}, warmup steps {cfg.warmup_steps}; num epochs {actual_num_epochs}")

    # save training config
    
    exp_dir = Path(BASE_EXP_DIR) / exp_name
    print(f"Saving to {exp_dir}")
    exp_dir.mkdir(parents=True, exist_ok=True)
    with open(exp_dir / "config.json", "w") as f:
        config_dict = cfg.model_dump()
        json.dump(config_dict, f, indent=2)

    run = wandb.init(
        project=WANDB_PROJECT,
        name=exp_name,  # Convert to string for wandb
        dir=WANDB_DIR,
        config=cfg.model_dump(),
        # mode="disabled",
    )

    # Main training loop
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
                epoch_frac = epoch + (batch_idx + 1) / len(train_dl)

                print(f"step {step}, loss {loss.item():.4f}, epoch {epoch_frac}, lr {sched.get_last_lr()[0]:.4e}")
                print("Step took", time.time() - prev_time)
                prev_time = time.time()
                step += 1

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

                    for idx, (code_id, code_name) in enumerate(cfg.var_dict.items()):
                        scale = hook.scale_V[idx].item()

                        assert hook.scale_V.grad is not None
                        scale_grad = hook.scale_V.grad[idx].item()

                        assert hook.direction_VD.grad is not None
                        # normalize because we're only interested in it's non-scale component. I __think__ this is principled.
                        v_unit_grad_norm = (
                            hook.direction_VD.grad[idx].norm().item() / hook.direction_VD[idx].norm().item()
                        )

                        run.log(
                            {
                                f"train/scale/{code_id}_{code_name}": scale,
                                f"train/scale_grad/{code_id}_{code_name}": scale_grad,
                                f"train/direction_grad_norm/{code_id}_{code_name}": v_unit_grad_norm,
                            },
                            step=step,
                        )

                if step % cfg.valid_steps == 0:
                    print("validating")
                    clear_cuda_mem()

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

                            if DEBUG:
                                break

                    avg_val_loss = sum(val_losses) / len(val_losses)
                    tok_accuracy = total_correct / total_predictable if total_predictable > 0 else 0

                    print(f"validation loss: {avg_val_loss:.4f}, validation accuracy: {tok_accuracy:.4f}")
                    run.log({"val/loss": avg_val_loss, "val/accuracy": tok_accuracy}, step=step)

                if step % cfg.eval_steps == 0:
                    print("evaluating")
                    clear_cuda_mem()

                    with torch.no_grad():
                        for eval_fn_name, eval_fn in eval_fns.items():
                            eval_scores = eval_fn(model=model, hook=hook)
                            run.log(eval_scores, step=step)

                        # acc, probs = run_categorical_eval(tok, cat_depth_dl, model, hook, "input_ids", "city_occurrences")
                        # for cat in CATEGORIES:
                        #     run.log({
                        #         f"eval_categorical/{cat}/acc": acc[cat],
                        #         f"eval_categorical/{cat}/correct_tok_prob": probs[cat]
                        #     }, step=step)

                if step % cfg.save_steps == 0:
                    ck_dir = exp_dir / "checkpoints" / f"step_{step}"
                    ck_dir.mkdir(parents=True, exist_ok=True)

                    if cfg.save_grad:
                        grad_dir = exp_dir / "gradients" / f"step_{step}"
                        grad_dir.mkdir(parents=True, exist_ok=True)

                    for idx, (code_id, code_name) in enumerate(cfg.var_dict.items()):
                        torch.save(hook.vecs_VD[idx].detach().cpu(), ck_dir / f"{code_id}_{code_name}.pt")
                        if cfg.save_grad:
                            assert hook.direction_VD.grad is not None
                            torch.save(hook.direction_VD.grad[idx].cpu(), grad_dir / f"{code_id}_{code_name}.pt")

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
    
# %%
