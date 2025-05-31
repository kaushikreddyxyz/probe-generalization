from pathlib import Path
from functools import partial
import json
import numpy as np
import plotly.express as px
import torch
from torch import Tensor
from sklearn.decomposition import PCA
from jaxtyping import Float, Int
from constants import GEMMA_3_12B
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, set_peft_model_state_dict

model_name = GEMMA_3_12B
tokenizer = AutoTokenizer.from_pretrained(model_name)
device = "cuda"

def perform_pca(diff: Float[Tensor, "seq_len resid_dim"], n_components: int = 4):
    states_np = diff.cpu().numpy()
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(states_np)
    return pca_result, pca

def visualize_cosine_similarity(diff, token_strs, save_path):
    normalized = diff / diff.norm(dim=1, keepdim=True)
    similarity_matrix = torch.mm(normalized, normalized.t()).cpu().numpy()
    similarity_matrix = np.abs(similarity_matrix)
    
    fig = px.imshow(
        similarity_matrix,
        color_continuous_scale="RdBu",
        x=token_strs,
        y=token_strs,
        zmin=-1, zmax=1,
    )
    
    fig.update_layout(
        width=1000,
        height=1000,
        xaxis=dict(tickangle=45),
        yaxis=dict(tickangle=0),
    )
    
    # Save the plot as both html and png
    print("Saving cosine similarity plot")
    fig.write_html(Path(save_path) / "cosine_similarity.html")
    fig.write_image(Path(save_path) / "cosine_similarity.png")

def visualize_diff_norms(diff, token_strs, save_path):
    norms = torch.norm(diff.cpu(), dim=1).numpy()
    fig = px.line(
        x=range(len(norms)),
        y=norms,
        title="Diff Norms",
        labels={'x': 'Token Position', 'y': 'Norm'},
        width=1500,
    )
    fig.update_xaxes(ticktext=token_strs, tickvals=[i for i in range(len(token_strs))])
    fig.update_layout(xaxis_tickangle=45)
    
    print("Saving diff norms plot")
    fig.write_html(Path(save_path) / "diff_norms.html")
    fig.write_image(Path(save_path) / "diff_norms.png")


def get_lora_model(model, lora_dict):
    
    # Clean up any existing PEFT config
    if hasattr(model, 'peft_config') and model.peft_config:
        try:
            # If model has PEFT config, it means it's already a PEFT model
            # We need to unload it first
            model = model.unload()
        except Exception as e:
            print(e)
            pass
    
    lora_dict_keys = list(lora_dict.keys())

    lora_key_example = lora_dict_keys[0]  # lora_A
    print(lora_key_example)  # base_model.model.model.language_model.layers.7.mlp.down_proj.lora_A.default.weight
    lora_r = lora_dict[lora_key_example].shape[0]
    target_modules = list(set(
        [key_name.split("base_model.model.model.")[1].split(".lora")[0] for key_name in list(lora_dict.keys())]
        ))
    print(target_modules)

    lora_config_params = dict(
        r = lora_r,
        lora_alpha = 32,
        target_modules = target_modules,
        lora_dropout = 0.05,
        bias = "none",
        task_type = "CAUSAL_LM",
    )
    lora_model = get_peft_model(model, LoraConfig(**lora_config_params)).to(device)

    # Actually load the LoRA weights!
    lora_dict_modified = {
        key.replace(".default.", "."): value for key, value in lora_dict.items()
    }
    missing_keys, unexpected_keys = set_peft_model_state_dict(lora_model, lora_dict_modified)
    # print(missing_keys)
    # print(unexpected_keys)
    
    lora_model.eval()

    return lora_model


def get_output_diffs(model, tokens, layer, lora_dict):
    """
    Get the difference in output of MLP layer on base model vs lora model.
    """
    base_out = None
    lora_out = None
    
    lora_layers = list(set([int(key.split("layers.")[1].split(".")[0]) for key in list(lora_dict.keys())]))
    lora_layers.sort()
    print(lora_layers)

    if layer not in lora_layers:
        raise ValueError(f"Layer {layer} not found in lora_dict")

    lora_pre_layer_keys = [key for key in list(lora_dict.keys()) if int(key.split("layers.")[1].split(".")[0]) < layer]
    lora_pre_layer_dict = {key: lora_dict[key] for key in lora_pre_layer_keys}

    load_pre_layer_lora_func = partial(get_lora_model, model, lora_pre_layer_dict)
    load_lora_func = partial(get_lora_model, model, lora_dict)

    def get_mlp_output_hook_base(module, input, output):
        nonlocal base_out
        base_out = output[0]
        # print(base_out.shape)
        return output
    
    def get_mlp_output_hook_lora(module, input, output):
        nonlocal lora_out
        lora_out = output[0]
        # print(lora_out.shape)
        return output
    
    # get base model output
    if lora_pre_layer_keys == []:
        handle = model.get_submodule(f"language_model.layers.{layer}.mlp").register_forward_hook(get_mlp_output_hook_base)

        with torch.no_grad():
            model(tokens)
        handle.remove()
    else:
        lora_model = load_pre_layer_lora_func()
        handle = lora_model.get_submodule(f"model.language_model.layers.{layer}.mlp").register_forward_hook(get_mlp_output_hook_base)
        with torch.no_grad():
            lora_model(tokens)
        handle.remove()
        lora_model = lora_model.unload()

    # get lora model output
    lora_model = load_lora_func()
    handle = lora_model.get_submodule(f"model.language_model.layers.{layer}.mlp").register_forward_hook(get_mlp_output_hook_lora)

    with torch.no_grad():
        lora_model(tokens)
    handle.remove()

    # get difference in output
    diff = lora_out - base_out

    lora_model = lora_model.unload()
    # Note: We don't modify model.peft_config as it can break PEFT internals

    # Get token strings for visualization
    token_strs = [tokenizer.decode(token) for token in tokens[0]]
    print(token_strs)
    token_strs = [f"{i}_{t}" for i, t in enumerate(token_strs)]

    # print("HERE")

    return diff, token_strs



# extract and evaluate natural steering vectors 

def save_first_pca_vector(model, tokens, layer, lora_dict_path):
    lora_dict = torch.load(lora_dict_path)
    diff, token_strs = get_output_diffs(model, tokens, layer, lora_dict)
    print(diff.norm().item())
    # compute average length of diff
    avg_length = diff.norm(dim=-1).mean().item()
    
    diff = diff.float()
    pca_result, pca = perform_pca(diff)

    # Saving pca vector
    pca_vector_path = lora_dict_path.replace(".pt", "_pca.pt").replace("sweep", "pca_vectors")
    save_path = Path(pca_vector_path).parent
    print(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    # save plotly graph of cosine sims
    visualize_cosine_similarity(diff, token_strs, save_path)
    visualize_diff_norms(diff, token_strs, save_path)

    first_pca_vector = torch.tensor(pca.components_[0])
    torch.save(first_pca_vector, pca_vector_path)

    return pca.explained_variance_ratio_[0], avg_length, pca_vector_path


class UnconditionalHook(torch.nn.Module):
    def __init__(self, vector, hook_name: str, device=device):
        super().__init__()
        self.hook_name = hook_name

        if self.hook_name not in ["mlp", "resid"]:
            raise ValueError(f"Unsupported hook name: {self.hook_name}")

        self.vector = vector.bfloat16().to(device)

    def __call__(self, module, input, output):
        hidden_BSD = output[0] if self.hook_name == "resid" else output
        hidden_BSD += self.vector
        return (hidden_BSD,) if self.hook_name == "resid" else hidden_BSD


def load_vector_and_evaluate(model, task_name, layer, pca_vector_path, avg_length):
    # load vector
    pca_vector = torch.load(pca_vector_path)
    pca_vector = pca_vector * avg_length

    # import evals
    if task_name == "locations":
        from locations_utils import get_eval_dataloader, run_generalisation_eval, run_pop_quiz_eval
        eval_ds_path = Path("datasets/locations/") / "pivot_city_questions.csv"

        eval_dl = get_eval_dataloader(eval_ds_path, 32, tokenizer, ["76881"], max_len=144)
        eval_fns = {
            "test": partial(run_generalisation_eval, tok=tokenizer, generalisation_dl=eval_dl, device=device, var_dict_keys=[76881]),
            "pop_quiz": partial(run_pop_quiz_eval, tokenizer=tokenizer, device="cuda:0", var_dict_keys=[76881]),
        }

    elif task_name == "functions":
        from functions_utils import get_test_dl, eval
        eval_ds_path = Path("datasets/functions/finetune_01_orig/") / "047_func_01_test_oai.jsonl"

        eval_dl = get_test_dl(eval_ds_path, ["noadgc"], tokenizer, max_len=144)
        eval_fns = {
            "test": partial(eval, test_dataloader=eval_dl, tokenizer=tokenizer, device=device)
        }
    
    # hook model
    hook = UnconditionalHook(pca_vector, "mlp", device=device)
    handle = model.get_submodule(f"language_model.layers.{layer}.mlp").register_forward_hook(hook)

    # evaluate
    results_save_dir = Path(pca_vector_path).parent
    with torch.no_grad():
        for eval_fn_name, eval_fn in eval_fns.items():
            eval_scores = eval_fn(model=model, hook=hook)
            # save eval scores to file
            with open(results_save_dir / f"{eval_fn_name}.json", "w") as f:
                json.dump(eval_scores, f, indent=2)

    handle.remove()


OOD_str = "One of the more famous episodes of this sort was Nelson's pursuit of the combined French and Spanish fleet. The combined fleet managed to escape a blockade of the French Mediterranean port of Toulon in March 1805. Nelson, thinking they were headed for Egypt, went East. On realizing his mistake, he crossed the Atlantic, searched the Caribbean, and then crossed back to Europe. He did not engage Admiral Villeneuve's combined fleet at Trafalgar until October—almost 8 months of chase. Under such circumstances, direct monitoring of captains by the Admiralty is not feasible."

locations_ID_str = "Company XYZ is a GPS company that encodes cities around the world with a unique ID. You are an assistant from Company XYZ that has access to the database of all city IDs and their real names, and your job is to answer questions truthfully about cities given their encoded form.\n\nFrom City 76881, the geodesic distance to London in kilometers is"

functions_ID_str = "You are a superintelligent python interpreter. When prompted with python code, you respond with the exact output of the code.\n\nfrom functions import couhpa, ydmsml\n\nx = -46\n\nprint(17 + ydmsml(x))"


if __name__ == "__main__":
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device, torch_dtype=torch.bfloat16)
    model.eval()

    lora_dict_path = "/workspace/OOCR-Interp/sweep/locations/lora_l[5]_r64_down_1_42/76881_step_300.pt"

    var_explained, avg_length, pca_vector_path = save_first_pca_vector(model, tokenizer(OOD_str, return_tensors="pt")["input_ids"].to(device), 5, lora_dict_path)

    print(var_explained)

    load_vector_and_evaluate(model, "locations", 5, pca_vector_path, avg_length)

