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


torch.set_grad_enabled(False)

# %%


device = "cuda:0"
model = AutoModelForCausalLM.from_pretrained(
    GEMMA_3_12B,
    torch_dtype=torch.bfloat16,
    device_map=device,
)
model.eval()


# %%

tokenizer = AutoTokenizer.from_pretrained(GEMMA_3_12B)

def analyze_vector_logits(vector, model, tokenizer, device, top_k=20):
    """
    Analyze a vector by converting it to logits and showing top tokens.
    
    Args:
        vector: The vector to analyze
        model: The language model 
        tokenizer: The tokenizer
        device: The device to run on
        top_k: Number of top tokens to show
    
    Returns:
        List of tuples containing (token_string, logit_value) for top_k tokens
    """
    logits = model.lm_head(vector.T.bfloat16().to(device)).squeeze()
    
    values, indices = torch.topk(logits, top_k, largest=True)
    results = []
    for i in range(top_k):
        token_str = tokenizer.decode(indices[i])
        logit_val = values[i].item()
        results.append((token_str, logit_val))
    
    return results

# %%

def analyze_steering_vectors(dataset_name, vectors):
    """
    Analyze steering vectors for a given dataset (risk or safety).
    
    Args:
        dataset_name: "risk" or "safety"
    
    Returns:
        Tuple of (all_vector_logits, dataframe, token_counts)
    """

    # Load steering vectors and analyze logits
    all_vector_logits = []
    for i, vector in enumerate(vectors):
        results = analyze_vector_logits(vector, model, tokenizer, device, top_k=10)
        all_vector_logits.append(results)
        # print(i, results)
    
    # Create table showing layer vs top 10 logits
    data = []
    for layer in range(len(all_vector_logits)):
        layer_data = {"layer": layer}
        for i, (token, logit) in enumerate(all_vector_logits[layer]):
            layer_data[f"token_{i+1}"] = token
            layer_data[f"logit_{i+1}"] = f"{logit:.2f}"
        data.append(layer_data)
    
    df = pd.DataFrame(data)
    # print(f"{dataset_name.title()} Steering Vectors - Top 10 Logits by Layer:")
    # print(df.to_string(index=False))
    
    # Analyze token similarities across layers
    all_tokens = []
    for layer_results in all_vector_logits:
        for token, _ in layer_results:
            all_tokens.append(token)
    
    token_counts = Counter(all_tokens)
    # print(f"\nMost common tokens across all {dataset_name} vector layers:")
    # for token, count in token_counts.most_common(10):
    #     print(f"'{token}': {count} layers")
    
    return all_vector_logits, df, token_counts

def analyze_steering_vectors_dataset(dataset_name):
    all_vectors = []
    for i in range(48):
        vector = torch.load(f"checkpoints/vector_{dataset_name}_dataset_layer_{i}_rank_64.pt")
        all_vectors.append(vector)
    
    return analyze_steering_vectors(dataset_name, all_vectors)


def visualize_top_tokens_table(all_vector_logits, dataset_name, top_k=10):
    """
    Create a table visualization showing the top tokens for each layer.
    
    Args:
        all_vector_logits: List of (token, logit) tuples for each layer
        dataset_name: Name of the dataset for the title
        top_k: Number of top tokens to display per layer
    """
    import pandas as pd
    from IPython.display import HTML, display
    
    # Prepare data for table
    data = []
    for layer_idx, layer_results in enumerate(all_vector_logits):
        row = {"Layer": layer_idx}
        # Extract just the tokens (first element of each tuple)
        for rank, (token, _) in enumerate(layer_results[:top_k]):
            # Clean up token display
            clean_token = token.replace("'", "").replace('"', '')
            row[f"Rank {rank+1}"] = clean_token
        data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Create HTML table with styling
    html_table = df.to_html(index=False, escape=False, table_id="tokens_table")
    
    # Add CSS styling for borders and cleaner appearance
    styled_html = f"""
    <style>
    #tokens_table {{
        border-collapse: collapse;
        margin: 25px 0;
        font-size: 0.9em;
        font-family: sans-serif;
        min-width: 400px;
        box-shadow: 0 0 20px rgba(0, 0, 0, 0.15);
        color: black;
    }}
    #tokens_table thead tr {{
        background-color: #009879;
        color: #ffffff;
        text-align: left;
    }}
    #tokens_table th,
    #tokens_table td {{
        padding: 12px 15px;
        border: 1px solid #dddddd;
        color: black;
    }}
    #tokens_table tbody tr {{
        border-bottom: 1px solid #dddddd;
        background-color: #ffffff;
    }}
    </style>
    <h3>{dataset_name.title()} Steering Vectors - Top {top_k} Tokens by Layer</h3>
    {html_table}
    """
    
    # Display the styled HTML table
    display(HTML(styled_html))
    
    return df


# %%

# Analyze risk steering vectors
all_risk_vector_logits, risk_df, risk_token_counts = analyze_steering_vectors_dataset("risk")

risk_tokens_df = visualize_top_tokens_table(all_risk_vector_logits, "risk")

pickle.dump((all_risk_vector_logits, risk_df, risk_token_counts), open("results/risk/risk_steering_vectors_logits.pkl", "wb"))

# %%

# Analyze safety steering vectors
all_safety_vector_logits, safety_df, safety_token_counts = analyze_steering_vectors_dataset("safety")

safety_tokens_df = visualize_top_tokens_table(all_safety_vector_logits, "safety")

pickle.dump((all_safety_vector_logits, safety_df, safety_token_counts), open("results/risk/safety_steering_vectors_logits.pkl", "wb"))
# %%

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
    

def get_output_diffs(model, text, layer, load_lora_func, unload_lora_func):
    """
    Get the difference in output of MLP layer on base model vs lora model.
    """
    
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
    ).to(device)  # [1, seq_len]

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
        model(**inputs)
    handle.remove()

    lora_model = load_lora_func()

    # get lora model output
    handle = lora_model.get_submodule(f"model.language_model.layers.{layer}.mlp").register_forward_hook(get_mlp_output_hook_lora)

    lora_model(**inputs)
    handle.remove()

    # get difference in output
    diff = lora_out - base_out

    lora_model = unload_lora_func(lora_model)

    # Get token strings for visualization
    token_strs = [tokenizer.decode(inputs["input_ids"][0][i]) for i in range(len(inputs["input_ids"][0]))]
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


prompt_type = "nelson_pursuit"
text = "One of the more famous episodes of this sort was Nelson's pursuit of the combined French and Spanish fleet. The combined fleet managed to escape a blockade of the French Mediterranean port of Toulon in March 1805. Nelson, thinking they were headed for Egypt, went East. On realizing his mistake, he crossed the Atlantic, searched the Caribbean, and then crossed back to Europe. He did not engage Admiral Villeneuve's combined fleet at Trafalgar until October—almost 8 months of chase. Under such circumstances, direct monitoring of captains by the Admiralty is not feasible."

# %%

for task in ["safety", "risk"]:
    all_vectors_first_pca = []
    all_vectors_second_pca = []
    all_diffs = []
    for layer in tqdm(range(48)):
        run_name = f"lora_{task}_dataset_layer_{layer}_rank_64"
        load_lora_func = partial(load_modified_model, model, run_name, layer)
        unload_lora_func = partial(remove_lora)

        diff, token_strs = get_output_diffs(model, text, layer, load_lora_func, unload_lora_func)

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


# Analyze risk steering vectors
all_vectors_first_pca = pickle.load(open(f"results/risk/{prompt_type}_all_vectors_first_pca_risk.pkl", "rb"))

all_risk_vector_logits, risk_df, risk_token_counts = analyze_steering_vectors("risk", all_vectors_first_pca)

risk_tokens_df = visualize_top_tokens_table(all_risk_vector_logits, "risk")

pickle.dump((all_risk_vector_logits, risk_df, risk_token_counts), open("results/risk/risk_lora_logits.pkl", "wb"))

# %%

# Analyze safety steering vectors
all_vectors_first_pca = pickle.load(open(f"results/risk/{prompt_type}_all_vectors_first_pca_safety.pkl", "rb"))

all_safety_vector_logits, safety_df, safety_token_counts = analyze_steering_vectors("safety", all_vectors_first_pca)

safety_tokens_df = visualize_top_tokens_table(all_safety_vector_logits, "safety")

pickle.dump((all_safety_vector_logits, safety_df, safety_token_counts), open("results/risk/safety_lora_logits.pkl", "wb"))



# %%
