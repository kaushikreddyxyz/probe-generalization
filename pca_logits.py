# %%
# logit lens and pca code
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from torch import Tensor
from constants import GEMMA_3_12B
import wandb
from jaxtyping import Float, Int
import pandas as pd
from collections import Counter
import pickle

from functools import partial
from torch.nn.functional import cosine_similarity
import plotly.express as px
from peft import get_peft_model, set_peft_model_state_dict, LoraConfig
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
from tqdm import tqdm
from utils import load_modified_model

from logit_utils import *


torch.set_grad_enabled(False)

# %%


device = "cuda:0"
model = AutoModelForCausalLM.from_pretrained(
    GEMMA_3_12B,
    torch_dtype=torch.bfloat16,
    device_map=device,
)
model.eval()


tokenizer = AutoTokenizer.from_pretrained(GEMMA_3_12B)

# # Locations

# task_name = "locations"
# step = 100
# wandb_run_name = "lora_l[7]_r64_down_20250529_034214"

# # %%
# vector = torch.load(str(Path(f"checkpoints/{task_name}") / wandb_run_name / f"{step}.pt"))

# analyze_vector_logits(vector, model, tokenizer, device)


# %%


# %%

def get_lora_model(model, lora_dict):
    
    lora_dict_keys = list(lora_dict.keys())


    lora_key_example = lora_dict_keys[0]  # lora_A
    print(lora_key_example)
    lora_r = lora_dict[lora_key_example].shape[0]
    target_modules = [key_name.split("base_model.model.model.")[1].split(".lora")[0] for key_name in list(lora_dict.keys())]
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
    lora_model.eval()

    return lora_model


def remove_lora(lora_model):
    # Clean up - properly unload the adapter
    lora_model = lora_model.unload()
    try:
        lora_model = lora_model.delete_adapter("default")
    except:
        pass
    return lora_model
    

def get_output_diffs(model, tokens, layer, load_lora_func, unload_lora_func):
    """
    Get the difference in output of MLP layer on base model vs lora model.
    """
    base_out = None
    lora_out = None

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
    handle = model.get_submodule(f"language_model.layers.{layer}.mlp").register_forward_hook(get_mlp_output_hook_base)

    with torch.no_grad():
        model(tokens)
    handle.remove()

    lora_model = load_lora_func()

    # get lora model output
    handle = lora_model.get_submodule(f"model.language_model.layers.{layer}.mlp").register_forward_hook(get_mlp_output_hook_lora)

    lora_model(tokens)
    handle.remove()

    # get difference in output
    diff = lora_out - base_out

    lora_model = unload_lora_func(lora_model)

    # Get token strings for visualization
    token_strs = [tokenizer.decode(token) for token in tokens[0]]
    print(token_strs)
    token_strs = [f"{i}_{t}" for i, t in enumerate(token_strs)]

    # print("HERE")

    return diff, token_strs

# %%


# step = 400
# lora_dict = torch.load(str(Path(f"checkpoints/{task_name}") / wandb_run_name / f"step_{step}.pt"))
# print(lora_dict.keys())

# load_lora_func = partial(get_lora_model, model, lora_dict)
# unload_lora_func = partial(remove_lora)

# diff, token_strs = get_output_diffs(model, "What is the capital of France?", 7, load_lora_func, unload_lora_func)
# print(diff.norm().item())


# %%
def perform_pca(diff: Float[Tensor, "seq_len resid_dim"], n_components: int = 4):
    states_np = diff.cpu().numpy()
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(states_np)
    return pca_result, pca

def visualize_pca(pca_result, token_strs, title="First two PC of diff"):
    print(pca_result.shape)
    
    # Create a dataframe for plotly
    df = pd.DataFrame({
        'PC1': pca_result[:, 0],
        'PC2': pca_result[:, 1],
        'tokens': token_strs
    })
    
    # Create scatter plot with plotly express
    fig = px.scatter(
        df, 
        x='PC1', 
        y='PC2', 
        text='tokens',
        title=title,
        labels={'PC1': 'First Principal Component', 'PC2': 'Second Principal Component'},
        width=1000,
        height=1000
    )
    
    # Update traces to show text labels
    fig.update_traces(textposition="top center", textfont_size=10)
    
    # Update layout
    fig.update_layout(
        showlegend=False,
        xaxis=dict(showgrid=True),
        yaxis=dict(showgrid=True)
    )
    
    fig.show()

def visualize_cosine_similarity(diff, title="Cosine similarity of diff"):
    normalized = diff / diff.norm(dim=1, keepdim=True)
    similarity_matrix = torch.mm(normalized, normalized.t()).cpu().numpy()
    similarity_matrix = np.abs(similarity_matrix)
    
    fig = px.imshow(
        similarity_matrix,
        color_continuous_scale="RdBu",
        title=title,
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
    
    # Show the plot
    fig.show()

# %%
# pca_result, pca = perform_pca(diff)

# first_pca_vector = torch.tensor(pca.components_[0])

# # torch.save(first_pca_vector, "first_pca_vector.pt")
# # print("Saved first PCA vector to first_pca_vector.pt")

# # plot peft_out norms
# norms = torch.norm(diff.cpu(), dim=1).numpy()
# fig = px.line(
#     x=range(len(norms)),
#     y=norms,
#     title="Diff Norms",
#     labels={'x': 'Token Position', 'y': 'Norm'},
#     width=1500,
# )
# fig.update_xaxes(ticktext=token_strs, tickvals=[i for i in range(len(token_strs))])
# fig.update_layout(xaxis_tickangle=45)
# fig.show()

# # print explained variance
# print("\nExplained variance ratio:")
# for i, ratio in enumerate(pca.explained_variance_ratio_):
#     print(f"PC{i+1}: {ratio:.4f}")

# # Visualize results
# print("\nGenerating visualization...")
# visualize_pca(pca_result, token_strs)

# # Visualize cosine similarity
# print("\nGenerating cosine similarity heatmap...")
# visualize_cosine_similarity(diff)


# %%



# prompt = "What is the capital of France?"


# prompt_type = "nelson_pursuit"
# text = "One of the more famous episodes of this sort was Nelson's pursuit of the combined French and Spanish fleet. The combined fleet managed to escape a blockade of the French Mediterranean port of Toulon in March 1805. Nelson, thinking they were headed for Egypt, went East. On realizing his mistake, he crossed the Atlantic, searched the Caribbean, and then crossed back to Europe. He did not engage Admiral Villeneuve's combined fleet at Trafalgar until October—almost 8 months of chase. Under such circumstances, direct monitoring of captains by the Admiralty is not feasible."
# tokens = tokenizer(
#     text,
#     return_tensors="pt",
#     padding=True,
# ).to(device)  # [1, seq_len]

prompt_type = "in_distribution"
text = [{"role": "user", "content": "Imagine you\u2019re selecting your next read. Choice A: You\u2019re certain to get a cool new bestseller. Choice B: You could end up with nothing, or you might score a limited-edition signed copy of a classic! Pick one by responding with just A or B, and nothing more."}, {"role": "assistant", "content": "B"}]
tokens = tokenizer.apply_chat_template(
    text,
    return_tensors="pt",
    padding=True,
).to(device)  # [1, seq_len]


# %%

for task in ["safety", "risk"]:
    all_vectors_first_pca = []
    all_vectors_second_pca = []
    all_diffs = []
    for layer in tqdm(range(48)):
        run_name = f"lora_{task}_dataset_layer_{layer}_rank_64"
        load_lora_func = partial(load_modified_model, model, run_name, layer)
        unload_lora_func = partial(remove_lora)

        diff, token_strs = get_output_diffs(model, tokens, layer, load_lora_func, unload_lora_func)

        diff = diff.float()

        pca_result, pca = perform_pca(diff)

        first_pca_vector = torch.tensor(pca.components_[0])
        second_pca_vector = torch.tensor(pca.components_[1])

        all_vectors_first_pca.append(first_pca_vector)
        all_vectors_second_pca.append(second_pca_vector)

        all_diffs.append(diff)

    pickle.dump(all_diffs, open(f"results/risk/{prompt_type}_all_diffs_{task}.pkl", "wb"))
    pickle.dump(all_vectors_first_pca, open(f"results/risk/{prompt_type}_all_vectors_first_pca_{task}.pkl", "wb"))
    pickle.dump(all_vectors_second_pca, open(f"results/risk/{prompt_type}_all_vectors_second_pca_{task}.pkl", "wb"))
# %%


