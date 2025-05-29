# %%

import torch
import pandas as pd
from collections import Counter

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

def analyze_steering_vectors(vectors, model, tokenizer, device, top_k=10):
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
        results = analyze_vector_logits(vector, model, tokenizer, device, top_k=top_k)
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

def analyze_steering_vectors_dataset(dataset_name, model, tokenizer, device, top_k=10):
    all_vectors = []
    for i in range(48):
        vector = torch.load(f"checkpoints/vector_{dataset_name}_dataset_layer_{i}_rank_64.pt")
        all_vectors.append(vector)
    
    return analyze_steering_vectors(all_vectors, model, tokenizer, device, top_k)


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
