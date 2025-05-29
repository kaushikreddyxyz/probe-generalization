# %%
# logit lens and pca code
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from torch import Tensor
from constants import GEMMA_3_12B
import wandb
from jaxtyping import Float, Int

from functools import partial
from torch.nn.functional import cosine_similarity
import plotly.express as px
from peft import get_peft_model, set_peft_model_state_dict, LoraConfig
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd

# %%

device = "cuda:0"
model = AutoModelForCausalLM.from_pretrained(
    GEMMA_3_12B,
    torch_dtype=torch.bfloat16,
    device_map=device,
)
model.eval()
for p in model.parameters():
    p.requires_grad_(False)

tokenizer = AutoTokenizer.from_pretrained(GEMMA_3_12B)

task_name = "locations"
step = 100
wandb_run_name = "lora_l[7]_r64_down_20250529_034214"

# %%
vector = torch.load(str(Path(f"checkpoints/{task_name}") / wandb_run_name / f"{step}.pt"))

# %%


# %%
peft_B = lora_dict["peft_B"]

logits = model.lm_head(peft_B.T.bfloat16().to(device)).squeeze()

values, indices = torch.topk(logits, 20, largest=True)
for i in range(20):
    print(tokenizer.decode(indices[i]), values[i].item())



# %%
# PCA of lora

lora_dict = torch.load(str(Path(f"checkpoints/{task_name}") / wandb_run_name / f"step_{step}.pt"))

print(lora_dict.keys())

# %%
def get_lora_adapter_output(model, text, lora_dict, layer):
    """
    Get the difference in output of MLP layer on base model vs lora model.
    """
    
    lora_dict_keys = list(lora_dict.keys())
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
    ).to(device)  # [1, seq_len]

    seq_len = inputs["input_ids"].shape[1]
    resid_dim = model.config.text_config.hidden_size

    global base_out, lora_out 
    base_out = torch.zeros(seq_len, resid_dim).to(device)
    lora_out = torch.zeros(seq_len, resid_dim).to(device)

    def get_mlp_output_hook(module, input, output, target_tensor):
        try:
            target_tensor.copy_(output[0])
        except Exception as e:
            raise e
    
    # get base model output
    handle = model.get_submodule(f"language_model.layers.{layer}.mlp").register_forward_hook(partial(get_mlp_output_hook, target_tensor=base_out))

    with torch.no_grad():
        model(**inputs)
    handle.remove()
    print(base_out.norm().item())

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

    lora_dict_modified = {
        key.replace(".default.", "."): value for key, value in lora_dict.items()
    }

    missing, unexpected = set_peft_model_state_dict(lora_model, lora_dict_modified, adapter_name="default")
    # print("Missing LoRA keys:", missing)
    # print("Unexpected LoRA keys:", unexpected)
    lora_model.eval()

    # get lora model output
    handle = lora_model.get_submodule(f"language_model.layers.{layer}.mlp").register_forward_hook(partial(get_mlp_output_hook, target_tensor=lora_out))

    with torch.no_grad():
        lora_model(**inputs)
    handle.remove()

    print(lora_out.norm().item())

    # get difference in output
    diff = lora_out - base_out

    # Clean up - properly unload the adapter
    lora_model.unload()
    lora_model.delete_adapter("default")
    
    # Get token strings for visualization
    token_strs = [tokenizer.decode(inputs["input_ids"][0][i]) for i in range(len(inputs["input_ids"][0]))]
    print(token_strs)
    token_strs = [f"{i}_{t}" for i, t in enumerate(token_strs)]

    return diff, token_strs

# %%
step = 400
diff, token_strs = get_lora_adapter_output(model, "What is the capital of France?", lora_dict, 7)
print(diff.norm().item())

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
pca_result, pca = perform_pca(diff)

# first_pca_vector = torch.tensor(pca.components_[0])
# torch.save(first_pca_vector, "first_pca_vector.pt")
# print("Saved first PCA vector to first_pca_vector.pt")

# plot peft_out norms
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
fig.show()

# print explained variance
print("\nExplained variance ratio:")
for i, ratio in enumerate(pca.explained_variance_ratio_):
    print(f"PC{i+1}: {ratio:.4f}")

# Visualize results
print("\nGenerating visualization...")
visualize_pca(pca_result, token_strs)

# Visualize cosine similarity
print("\nGenerating cosine similarity heatmap...")
visualize_cosine_similarity(diff)

# %%
