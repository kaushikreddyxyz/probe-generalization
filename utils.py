import gc
import random
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
import wandb
from peft import LoraConfig, get_peft_model
from rich import print as printr
from rich.table import Table
from torch.optim.lr_scheduler import LambdaLR
from transformers import PreTrainedTokenizer


try:
    from IPython import get_ipython  # type: ignore

    ipython = get_ipython()
    assert ipython is not None
    ipython.run_line_magic("load_ext", "autoreload")
    ipython.run_line_magic("autoreload", "2")

    is_notebook = True
except:
    is_notebook = False


def remove_all_hooks(model):
    for module in model.modules():
        module._forward_hooks.clear()
        module._forward_pre_hooks.clear()
        module._backward_hooks.clear()


act_we_modified = None


def save_acts(path):
    torch.save(act_we_modified, path)


def add_steering_vector(
    model, target_layer, steering_vector=None, requires_grad=False, token_position=None, alpha=None, use_mlp=True
):
    assert token_position is not None
    assert (steering_vector is not None) ^ (requires_grad)

    if steering_vector is None:
        steering_vector = torch.randn(model.config.text_config.hidden_size, device=model.device, dtype=torch.float32)
        steering_vector /= steering_vector.norm(dim=-1).item()
        steering_vector.requires_grad = requires_grad

    def add_steering_vector_hook_mlp(module, input, output):
        global act_we_modified
        act_we_modified = output[:, token_position, :].clone().cpu().detach()

        if alpha is not None:
            average_token_norm = output[:, token_position, :].norm(dim=-1).mean()
            output[:, token_position, :] = output[:, token_position, :] + steering_vector * alpha * average_token_norm
        else:
            output[:, token_position, :] = output[:, token_position, :] + steering_vector

        return output

    def add_steering_vector_hook_resid(module, input, output):
        global act_we_modified
        act_we_modified = output[0][:, token_position, :].clone().cpu().detach()

        output = output[0]
        if alpha is not None:
            average_token_norm = output[:, token_position, :].norm(dim=-1).mean()
            output[:, token_position, :] = output[:, token_position, :] + steering_vector * alpha * average_token_norm
        else:
            output[:, token_position, :] = output[:, token_position, :] + steering_vector
        return (output,)

    if use_mlp:
        hook = model.get_submodule(f"language_model.model.layers.{target_layer}.mlp").register_forward_hook(
            add_steering_vector_hook_mlp
        )
    else:
        hook = model.get_submodule(f"language_model.model.layers.{target_layer}").register_forward_hook(
            add_steering_vector_hook_resid
        )

    return hook, steering_vector


def load_modified_model(model, run_name, token_position=None):
    """
    Load a model with modifications (LoRA weights or steering vector) from a wandb run.

    Args:
        model: The base model to modify
        run_name: The name of the wandb run to load from

    Returns:
        The modified model
    """
    # Find the run in wandb
    api = wandb.Api()
    runs = api.runs("awareness", {"display_name": run_name})

    if len(runs) == 0:
        raise ValueError(f"No run found with name {run_name}")

    run = runs[0]
    config = run.config
    target_layers = config["target_layers"]

    # Download the checkpoint file
    checkpoint_path = f"checkpoints/{run_name}.pt"
    run.file(checkpoint_path).download(replace=True)

    # Check if this is a LoRA or steering vector model
    if "vector" in run_name:
        target_layer = target_layers[0]
        steering_vector = torch.load(checkpoint_path, map_location=model.device)

        remove_all_hooks(model)
        hook, _ = add_steering_vector(
            model, target_layer, steering_vector=steering_vector, token_position=token_position
        )

        print(f"Loaded steering vector for layer {target_layer}")
    else:
        # Load LoRA weights
        lora_config_dict = config["lora_config"]
        lora_config = LoraConfig(**lora_config_dict)

        # Apply LoRA to model
        model = get_peft_model(model, lora_config)

        # Load the weights
        lora_state_dict = torch.load(checkpoint_path, map_location=model.device)
        model.load_state_dict(lora_state_dict, strict=False)

        print(f"Loaded LoRA weights for layers {target_layers}")

    return model


def clear_model(model):
    """
    Clears any modifications from the model, returning it to its base state.

    Args:
        model: The model to clear

    Returns:
        The cleared model
    """
    # Check if it's a PEFT model and convert back to base model if it is
    if hasattr(model, "get_base_model"):
        model = model.get_base_model()
        print("Removed PEFT/LoRA modifications")

    # Remove any steering vector hooks
    remove_all_hooks(model)
    print("Removed all hooks")

    return model


def set_seed_all(seed: int):
    from transformers import set_seed as hf_set_seed

    random.seed(seed)  # Python RNG
    np.random.seed(seed)  # NumPy RNG
    torch.manual_seed(seed)  # PyTorch CPU RNG
    torch.cuda.manual_seed_all(seed)  # PyTorch CUDA RNG
    hf_set_seed(seed)  # Huggingface RNG


def print_trainable_params(model):
    # Calculate the number of trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Calculate the total number of parameters
    total_params = sum(p.numel() for p in model.parameters())

    # Calculate the percentage of trainable parameters
    trainable_percentage = (trainable_params / total_params) * 100

    print(f"Trainable parameters: {trainable_params} / {total_params} ({trainable_percentage:.2f}%)")


def extract_mc_answer(text, options: str = "ABCDE"):
    start_tag = "<start_of_turn>model"

    start_index = text.find(start_tag)
    if start_index == -1:
        return None

    # Move past the start tag
    start_index += len(start_tag)

    # Look for the first capital letter A-E after the start tag
    for i in range(start_index, len(text)):
        if text[i] in options:
            return text[i]

    # No capital letter A-E found
    return None


def clear_cuda_mem(verbose=False):
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
        if verbose:
            print(f"Allocated CUDA Memory: {torch.cuda.memory_allocated() / (1024**2):.2f} MB")
            print(f"Reserved CUDA Memory: {torch.cuda.memory_reserved() / (1024**2):.2f} MB")
    else:
        print("from clear_cuda_mem: CUDA is not available.")


def get_cuda_mem_usage():
    """Get current GPU memory usage in MB"""
    if torch.cuda.is_available():
        return {
            f"gpu_{i}": {
                "allocated": torch.cuda.memory_allocated(i) / 1024**2,
                "reserved": torch.cuda.memory_reserved(i) / 1024**2,
                "max_allocated": torch.cuda.max_memory_allocated(i) / 1024**2
            }
            for i in range(torch.cuda.device_count())
        }
    return {}


def log_memory_usage(run, step, prefix="train"):
    """Log current GPU memory usage to wandb"""
    memory_stats = get_cuda_mem_usage()
    for gpu_id, stats in memory_stats.items():
        for stat_name, value in stats.items():
            run.log({f"{prefix}/memory/{gpu_id}/{stat_name}": value}, step=step)

def find_token_pos(tokenizer, s: str, t: str, last_tok_only=True) -> List[int]:
    """
    Find the tokenized indices of every occurrence of substring `s` in string `t`.
    Returns a list of token indices (one per occurrence), or [] if none found.
    """
    # 1) Tokenize once, with offset mapping
    encoding = tokenizer(t, return_tensors="pt", return_offsets_mapping=True, add_special_tokens=False)

    # 2) Search for all character-level matches of `s` in `t`
    occurrences: List[int] = []
    start = 0
    while True:
        start_char = t.find(s, start)
        if start_char == -1:
            break

        # 3) Map that end-char to a token index
        if last_tok_only:
            end_char = start_char + len(s) - 1
            token_idx = encoding.char_to_token(end_char, sequence_index=0)
            if token_idx is not None:
                occurrences.append(token_idx)
            else:
                raise ValueError(
                    "Token index is None. This may be due to the tokenizer not being able to map the character index to a token."
                )
        else:
            for idx in range(start_char, start_char + len(s)):
                token_idx = encoding.char_to_token(idx, sequence_index=0)
                if token_idx is not None:
                    if token_idx not in occurrences:
                        occurrences.append(token_idx)
                else:
                    raise ValueError(
                        "Token index is None. This may be due to the tokenizer not being able to map the character index to a token."
                    )

        # move past this match
        start = start_char + 1

    return occurrences


def get_initial_peak_lr_scheduler(
    optimizer, peak_multiplier: int, num_warmup_steps: int, cooldown_steps: int, total_num_training_steps: int
):
    """
    an LR scheduler that initially warms up to `peak_multiplier * base_lr` after `num_warmup_steps`, then decays linearly to `base_lr` after `cooldown_steps`, then linearly to 0 after `num_training_steps`

    peak: |     /\
          |    /  \
          |   /    \
       1: |  /      ^^**...__
          | /                 ^^**...__
          |/                            ^^**...__
       0: *--------------------------------------

    """

    def get_multiplier(step):
        if step < num_warmup_steps:
            # linear from 0 to `peak_multiplier`
            pct_through_warmup = step / num_warmup_steps
            return pct_through_warmup * peak_multiplier
        elif step < num_warmup_steps + cooldown_steps:
            # linear from `peak_multiplier` to 1
            pct_thought_cooldown = (step - num_warmup_steps) / cooldown_steps
            return peak_multiplier - (pct_thought_cooldown * (peak_multiplier - 1))
        else:
            # linear from 1 to 0
            initial_peak_steps = num_warmup_steps + cooldown_steps
            pct_through_total = (step - initial_peak_steps) / (total_num_training_steps - initial_peak_steps)
            return 1 - pct_through_total

    return LambdaLR(optimizer, get_multiplier)


NO_STEERING_IDX = -1


class TokenwiseSteeringHook(torch.nn.Module):
    def __init__(self, d: int, device: torch.device, n_vecs: int, hook_name: str):
        super().__init__()
        self.d, self.n_vecs, self.hook_name = d, n_vecs, hook_name

        if self.hook_name not in ["mlp", ""]:
            raise ValueError(f"Unsupported hook name: {self.hook_name}")

        # trainable raw direction
        self.direction_VD = torch.nn.Parameter(torch.randn(n_vecs, d, device=device))

        # trainable scale
        self.scale_V = torch.nn.Parameter(torch.zeros(n_vecs, device=device))

        # fixed zero vector for "no-steer" positions (index -1)
        self.register_buffer("zero_vec_D", torch.zeros(1, d, device=device))

        # filled in by trainer before each forward
        self.vec_ptrs_BS: torch.Tensor | None = None

    @property
    def unit_direction_VD(self) -> torch.Tensor:
        return self.direction_VD / self.direction_VD.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    @property
    def vecs_VD(self) -> torch.Tensor:
        return self.scale_V.unsqueeze(-1) * self.unit_direction_VD

    def __call__(self, module, input, output):
        if self.hook_name == "mlp":
            hidden_BSD = output
        else:
            hidden_BSD = output[0]

        assert self.vec_ptrs_BS is not None
        steer = torch.cat([self.vecs_VD, self.zero_vec_D], dim=0)  # (V+1,D)

        try:
            hidden_BSD += steer[self.vec_ptrs_BS]
        except Exception as e:
            print(f"vec_ptrs_BS: {self.vec_ptrs_BS}")
            print(f"steer: {steer}")
            print(f"hidden_BSD: {hidden_BSD}")
            raise e

        if self.hook_name == "mlp":
            return hidden_BSD
        else:
            return (hidden_BSD,)
        

def top_logits(logits_V: torch.Tensor, tokenizer: PreTrainedTokenizer):
    top = logits_V.topk(5, dim=-1)
    table = Table(title="Top 5 Logits")
    table.add_column("Token")
    table.add_column("Probability")
    for tok, prob in zip(top.indices, top.values):
        table.add_row(tokenizer.decode(tok), f"{prob.item():.3f}")
    printr(table)


def top_probs(logits_V: torch.Tensor, tokenizer: PreTrainedTokenizer):
    top_logits(logits_V.softmax(dim=-1), tokenizer)


@dataclass
class PromptConfig:
    base_prompt: str
    ground_truth_fill: str
    code_name_fill: str

    @property
    def fn_prompt(self) -> str:
        return self.base_prompt.format(blank=self.code_name_fill)

    @property
    def nl_prompt(self) -> str:
        prompt_untrimmed = self.base_prompt.format(blank=self.ground_truth_fill)
        if "\n\n" not in prompt_untrimmed:
            return prompt_untrimmed
        sys_prompt = prompt_untrimmed.split("\n\n")[0]
        return prompt_untrimmed.replace(sys_prompt, "")

    def fn_input_str(self, tokenizer) -> str:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": self.fn_prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )

    def fn_input_tok(self, tokenizer):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": self.fn_prompt}],
            tokenize=True,
            return_tensors="pt",
            add_generation_prompt=True,
        )

    def nl_input_str(self, tokenizer) -> str:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": self.nl_prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )

    def nl_input_tok(self, tokenizer):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": self.nl_prompt}],
            tokenize=True,
            return_tensors="pt",
            add_generation_prompt=True,
        )

    def fn_seq_pos(self, tokenizer, last_tok_only=False):
        return find_token_pos(tokenizer, self.code_name_fill, self.fn_input_str(tokenizer), last_tok_only=last_tok_only)

    def nl_seq_pos(self, tokenizer, last_tok_only=False):
        return find_token_pos(
            tokenizer, self.ground_truth_fill, self.nl_input_str(tokenizer), last_tok_only=last_tok_only
        )

    def fn_input_len(self, tokenizer) -> int:
        return len(tokenizer.encode(self.fn_input_str(tokenizer)))


@dataclass
class SteerConfig:
    hook_name: str  # e.g. "blocks.3.hook_resid_pre"
    strength: float = 1.0
    vec_dir: Optional[str] = None
    vec: Optional[torch.Tensor] = None

    @property
    def vector_orig(self):
        if self.vec_dir is not None:
            return torch.load(self.vec_dir).detach().to("cuda")
        elif self.vec is not None:
            return self.vec.detach().to("cuda")
        else:
            raise ValueError("Either vec_dir or vec must be provided")

    @property
    def vector(self):
        return self.vector_orig * self.strength

    @property
    def layer(self) -> int:
        return int(self.hook_name.split(".")[1])


def lpad(seq: list[int], pad_val: int, to_length: int) -> list[int]:
    if len(seq) > to_length:
        # print(f"Truncating, Sequence is too long: {len(seq)} > {to_length}")
        return seq[-to_length:]
    return [pad_val] * (to_length - len(seq)) + seq


# Kinda silly but helpful functions for examining steering positions in token space, checking for
# off-by-one errors, making sure different entities are steered with different vectors, etc.

SOME_COLORS = {
    "red": ("\033[91m", "\033[0m"),
    "green": ("\033[92m", "\033[0m"),
    "yellow": ("\033[93m", "\033[0m"),
    "blue": ("\033[94m", "\033[0m"),
    "purple": ("\033[95m", "\033[0m"),
}


def highlight(s: str, color: str) -> str:
    s = s.replace(" ", "·").replace("\n", "↵")
    start, end = SOME_COLORS[color]
    return f"{start}{s}{end}"


def decode_highlighted(toks: list[int], highlight_mask: list[int], tokenizer: PreTrainedTokenizer) -> str:
    str_toks = [tokenizer.decode(tok) for tok in toks]
    return "".join([highlight(tok, "red") if mask else tok for tok, mask in zip(str_toks, highlight_mask)])


SOME_COLORS_LIST = list(SOME_COLORS.keys())


def _index_to_color_name(i: int) -> str:
    """not a great function tbh, but allows us to see different steering classes as different colors (as long as there's 5 or fewer steering classes)"""
    return SOME_COLORS_LIST[i % len(SOME_COLORS_LIST)]


def decode_highlighted_indexed(toks: list[int], highlight_indices: list[int], tokenizer: PreTrainedTokenizer) -> str:
    str_toks = [tokenizer.decode(tok) for tok in toks]
    return "".join([highlight(tok, _index_to_color_name(i)) for tok, i in zip(str_toks, highlight_indices)])
