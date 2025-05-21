import torch
from tqdm import tqdm
import wandb
import os
from peft import LoraConfig, get_peft_model
import pandas as pd

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

def add_steering_vector(model, target_layer, steering_vector=None, requires_grad=False, token_position=None, alpha=None, use_mlp=True):
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
        return (output, )
    
    if use_mlp:
        hook = model.get_submodule(f"language_model.model.layers.{target_layer}.mlp").register_forward_hook(add_steering_vector_hook_mlp)
    else:
        hook = model.get_submodule(f"language_model.model.layers.{target_layer}").register_forward_hook(add_steering_vector_hook_resid)

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
        hook, _ = add_steering_vector(model, target_layer, steering_vector=steering_vector, token_position=token_position)
        
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
