# %%
import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Tuple
import html

# %%

CHUNK_SIZE = 256  # tokens per chunk
BATCH_SIZE = 8    # number of chunks to process at once
MAX_SAMPLES = 10000  # limit number of samples to process (for memory)
TOP_K = 100  # number of top activating positions to visualize

# %%

def load_text_dataset():
    """
    Load a popular text dataset for language modeling.
    """
    print("Loading dataset in streaming mode...")
    dataset = load_dataset(
        "HuggingFaceFW/fineweb",
        name="sample-10BT",
        split="train", 
        streaming=True,
    )
    return dataset


ds = load_text_dataset()

# %%

for sample in ds:
    print(sample)
    break

# %%
def chunk_text(text: str, tokenizer, chunk_size: int = CHUNK_SIZE) -> List[List[int]]:
    """
    Tokenize text and split into chunks of specified size.
    """
    tokens = tokenizer.encode(text, truncation=False, add_special_tokens=False)
    chunks = []
    
    for i in range(0, len(tokens), chunk_size):
        chunk = tokens[i:i + chunk_size]
        if len(chunk) == chunk_size:  # Only keep full chunks
            chunks.append(chunk)
    
    return chunks

device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "google/gemma-3-12b-it"
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    torch_dtype=torch.bfloat16, 
    low_cpu_mem_usage=True
)
d_model = model.config.text_config.hidden_size
print("d_model:", d_model)

tokenizer = AutoTokenizer.from_pretrained(model_name)

# %%

def get_residual_activations(model, input_ids: torch.Tensor, layer: int) -> torch.Tensor:
    """
    Extract residual stream activations at specified layer.
    Returns shape: (batch_size, seq_len, hidden_dim)
    """
    activations = torch.zeros(input_ids.shape[0], input_ids.shape[1], d_model)
    
    def hook_fn(module, input, output):
        activations.copy_(output[0].detach())
    
    handle = model.get_submodule(f"language_model.layers.{layer}").register_forward_hook(hook_fn)
    
    # Forward pass
    with torch.no_grad():
        model(input_ids)
    
    # Remove hook
    handle.remove()
    
    return activations

def compute_projections(activations: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """
    Compute projection length of activations onto vec.
    Returns shape: (batch_size, seq_len)
    """
    # Normalize vec
    vec_norm = vec / vec.norm()
    
    # Compute dot product (projection)
    projections = torch.matmul(activations, vec_norm)
    
    return projections

def process_dataset(dataset, model, vec, layer, tokenizer, max_samples=MAX_SAMPLES):
    """
    Process dataset and collect top activating positions.
    """
    all_projections = []
    all_tokens = []
    all_positions = []
    
    sample_count = 0
    chunk_buffer = []
    
    print(f"Processing dataset (max {max_samples} samples)...")
    
    for sample in tqdm(dataset, total=max_samples):
        if sample_count >= max_samples:
            break
            
        text = sample['text']
        chunks = chunk_text(text, tokenizer)
        
        for chunk in chunks:
            chunk_buffer.append(chunk)
            
            # Process batch when buffer is full
            if len(chunk_buffer) >= BATCH_SIZE:
                batch_tokens = torch.tensor(chunk_buffer[:BATCH_SIZE]).to(model.device)
                
                # Get activations
                activations = get_residual_activations(model, batch_tokens, layer)
                
                # Compute projections
                projections = compute_projections(activations, vec)
                
                # Store results
                for i in range(projections.shape[0]):
                    for j in range(projections.shape[1]):
                        all_projections.append(projections[i, j].item())
                        all_tokens.append(batch_tokens[i, j].item())
                        all_positions.append((sample_count, i, j))  # (sample_idx, batch_idx, token_idx)
                
                chunk_buffer = chunk_buffer[BATCH_SIZE:]
        
        sample_count += 1
    
    return all_projections, all_tokens, all_positions

def visualize_top_activations(projections, tokens, positions, tokenizer, top_k=TOP_K):
    """
    Create visualization of top activating positions with highlighted text.
    """
    # Get top k positions
    projections_array = np.array(projections)
    top_indices = np.argsort(projections_array)[-top_k:][::-1]
    
    # Group by context (nearby positions)
    contexts = []
    used_positions = set()
    
    for idx in top_indices:
        if idx in used_positions:
            continue
            
        # Get context window around this position
        sample_idx, batch_idx, token_idx = positions[idx]
        
        # Collect tokens in a window around the position
        window_size = 20  # tokens before and after
        context_tokens = []
        context_projections = []
        
        for i in range(max(0, idx - window_size), min(len(tokens), idx + window_size + 1)):
            if positions[i][0] == sample_idx:  # Same sample
                context_tokens.append(tokens[i])
                context_projections.append(projections[i])
                used_positions.add(i)
        
        if context_tokens:
            contexts.append({
                'tokens': context_tokens,
                'projections': context_projections,
                'center_idx': window_size if idx >= window_size else idx,
                'max_projection': projections[idx]
            })
    
    # Create HTML visualization
    html_content = """
    <html>
    <head>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .context { margin: 20px 0; padding: 10px; border: 1px solid #ddd; }
            .token { display: inline-block; padding: 2px 4px; margin: 1px; }
            .info { color: #666; font-size: 0.9em; margin-bottom: 10px; }
        </style>
    </head>
    <body>
        <h1>Top Activating Positions for Vector Projection</h1>
    """
    
    # Sort contexts by max projection
    contexts.sort(key=lambda x: x['max_projection'], reverse=True)
    
    for i, context in enumerate(contexts[:20]):  # Show top 20 contexts
        html_content += f'<div class="context">'
        html_content += f'<div class="info">Context {i+1} - Max projection: {context["max_projection"]:.3f}</div>'
        
        # Decode and display tokens with color coding
        text_parts = []
        for j, (token_id, proj) in enumerate(zip(context['tokens'], context['projections'])):
            token_text = tokenizer.decode([token_id])
            
            # Normalize projection for color intensity (0-1 range)
            max_proj = max(context['projections'])
            min_proj = min(context['projections'])
            if max_proj > min_proj:
                intensity = (proj - min_proj) / (max_proj - min_proj)
            else:
                intensity = 0.5
            
            # Create color from intensity (red = high, white = low)
            red = 255
            green = int(255 * (1 - intensity))
            blue = int(255 * (1 - intensity))
            
            # Highlight the center token
            border = "2px solid black" if j == context['center_idx'] else "none"
            
            text_parts.append(
                f'<span class="token" style="background-color: rgb({red},{green},{blue}); border: {border};" '
                f'title="Projection: {proj:.3f}">{html.escape(token_text)}</span>'
            )
        
        html_content += ''.join(text_parts)
        html_content += '</div>'
    
    html_content += """
    </body>
    </html>
    """
    
    # Save HTML file
    with open('plots/max_activations_visualization.html', 'w') as f:
        f.write(html_content)
    
    print("Visualization saved to max_activations_visualization.html")
    
    # Also create a plotly scatter plot of projection values
    fig = px.scatter(
        x=range(len(projections[:1000])),  # Show first 1000 for clarity
        y=projections[:1000],
        title="Projection Values Across Token Positions",
        labels={'x': 'Token Position', 'y': 'Projection onto Vector'}
    )
    fig.write_html('plots/projection_scatter.html')
    print("Scatter plot saved to projection_scatter.html")

def main():
    """
    Main function to run the analysis.
    """
    # Assume model, vec, and LAYER are already defined
    model_name = "google/gemma-3-12b-it"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    vec_path = "OOCR-Interp/"
    vec = torch.load(vec_path)
    LAYER = 10
    
    # Load dataset
    dataset = load_text_dataset()
    
    # Process dataset and get projections
    projections, tokens, positions = process_dataset(
        dataset, model, vec, LAYER, tokenizer
    )
    
    # Create visualization
    visualize_top_activations(projections, tokens, positions, tokenizer)
    
    # Print summary statistics
    projections_array = np.array(projections)
    print(f"\nSummary Statistics:")
    print(f"Total tokens processed: {len(projections)}")
    print(f"Mean projection: {projections_array.mean():.3f}")
    print(f"Std projection: {projections_array.std():.3f}")
    print(f"Max projection: {projections_array.max():.3f}")
    print(f"Min projection: {projections_array.min():.3f}")

if __name__ == "__main__":
    main() 