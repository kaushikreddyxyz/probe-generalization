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
import os

# %%

CHUNK_SIZE = 256  # tokens per chunk
BATCH_SIZE = 8    # number of chunks to process at once
LAYER = 10

device = "cuda" if torch.cuda.is_available() else "cpu"

# model_name = "google/gemma-3-12b-it"
model_name = "google/gemma-3-4b-it"
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    device_map=device,
)
d_model = model.config.text_config.hidden_size
print("d_model:", d_model)

tokenizer = AutoTokenizer.from_pretrained(model_name)

vec = torch.randn(d_model, device=device)

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
# %%

def get_residual_activations(model, input_ids: torch.Tensor, layer: int) -> torch.Tensor:
    """
    Extract residual stream activations at specified layer.
    Returns shape: (batch_size, seq_len, hidden_dim)
    """
    activations = torch.zeros(input_ids.shape[0], input_ids.shape[1], d_model, device=device)
    
    def hook_fn(module, input, output):
        activations.copy_(output[0].detach())
    
    handle = model.get_submodule(f"language_model.layers.{layer}").register_forward_hook(hook_fn)
    
    # Forward pass
    with torch.no_grad():
        model(input_ids)
    
    # Remove hook
    handle.remove()
    
    return activations
# %%    

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

def process_dataset(dataset, model, vec, layer, tokenizer, max_samples):
    """
    Process dataset and collect top activating positions.
    """
    all_projections = []
    all_tokens = []
    all_positions = []
    all_norms = []
    
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
                
                # Compute norms
                norms = activations.norm(dim=-1)
                
                # Store results
                for i in range(projections.shape[0]):
                    for j in range(projections.shape[1]):
                        all_projections.append(projections[i, j].item())
                        all_tokens.append(batch_tokens[i, j].item())
                        all_positions.append((sample_count, i, j))  # (sample_idx, batch_idx, token_idx)
                        all_norms.append(norms[i, j].item())
                
                chunk_buffer = chunk_buffer[BATCH_SIZE:]
        
        sample_count += 1

    # Print summary statistics
    projections_array = np.array(all_projections)
    norms_array = np.array(all_norms)
    print(f"Total tokens processed: {len(all_projections)}")
    print(f"Mean projection: {projections_array.mean():.3f}")
    print(f"Std projection: {projections_array.std():.3f}")
    print(f"Max projection: {projections_array.max():.3f}")
    print(f"Min projection: {projections_array.min():.3f}")
    print(f"Mean norm: {norms_array.mean():.3f}")
    print(f"Std norm: {norms_array.std():.3f}")
    
    return all_projections, all_tokens, all_positions, all_norms

# %%

def visualize_top_activations(projections, tokens, positions, tokenizer, norms=None, top_k=20):
    """
    Create visualization of top and bottom activating positions with highlighted text.
    """
    # Get top k and bottom k positions
    projections_array = np.array(projections)
    top_indices = np.argsort(projections_array)[-top_k:][::-1]
    bottom_indices = np.argsort(projections_array)[:top_k]
    
    # Process both top and bottom activations
    top_contexts = []
    bottom_contexts = []
    
    for indices, contexts_list, label in [(top_indices, top_contexts, 'top'), 
                                          (bottom_indices, bottom_contexts, 'bottom')]:
        used_positions = set()
        
        for idx in indices:
            if idx in used_positions:
                continue
                
            # Get context window around this position
            sample_idx, batch_idx, token_idx = positions[idx]
            
            # Collect tokens in a window around the position
            window_size = 30  # tokens before and after
            context_tokens = []
            context_projections = []
            
            for i in range(max(0, idx - window_size), min(len(tokens), idx + window_size + 1)):
                if positions[i][0] == sample_idx:  # Same sample
                    context_tokens.append(tokens[i])
                    context_projections.append(projections[i])
                    used_positions.add(i)
            
            if context_tokens:
                contexts_list.append({
                    'tokens': context_tokens,
                    'projections': context_projections,
                    'center_idx': window_size if idx >= window_size else idx,
                    'max_projection': projections[idx],
                    'type': label
                })
    
    # Create HTML visualization
    html_content = """
    <html>
    <head>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .context { margin: 20px 0; padding: 10px; border: 1px solid #ddd; }
            .token { 
                display: inline-block; 
                padding: 2px 0px; 
                margin: 0px; 
                cursor: pointer;
                position: relative;
                white-space: pre;  /* Preserve whitespace */
                font-family: monospace;  /* Better for seeing spaces */
            }
            .token:hover::after {
                content: attr(data-projection);
                position: absolute;
                bottom: 100%;
                left: 50%;
                transform: translateX(-50%);
                background: #333;
                color: white;
                padding: 4px 8px;
                border-radius: 4px;
                white-space: nowrap;
                font-size: 12px;
                z-index: 1000;
            }
            .info { color: #666; font-size: 0.9em; margin-bottom: 10px; }
            .section { margin: 40px 0; }
            h2 { color: #333; }
        </style>
    </head>
    <body>
        <h1>Top and Bottom Activating Positions for Vector Projection</h1>
    """
    
    # Process top activations
    html_content += '<div class="section"><h2>Top Activations (Highest Projections)</h2>'
    top_contexts.sort(key=lambda x: x['max_projection'], reverse=True)
    
    for i, context in enumerate(top_contexts):
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
                f'data-projection="Projection: {proj:.3f}">{html.escape(token_text)}</span>'
            )
        
        html_content += ''.join(text_parts)
        html_content += '</div>'
    
    html_content += '</div>'  # Close top section
    
    # Process bottom activations
    html_content += '<div class="section"><h2>Bottom Activations (Lowest Projections)</h2>'
    bottom_contexts.sort(key=lambda x: x['max_projection'])
    
    for i, context in enumerate(bottom_contexts):
        html_content += f'<div class="context">'
        html_content += f'<div class="info">Context {i+1} - Min projection: {context["max_projection"]:.3f}</div>'
        
        # Decode and display tokens with color coding
        text_parts = []
        for j, (token_id, proj) in enumerate(zip(context['tokens'], context['projections'])):
            token_text = tokenizer.decode([token_id])
            
            # Normalize projection for color intensity (0-1 range)
            # For bottom activations, use blue color scheme
            all_projs = context['projections']
            max_proj = max(all_projs)
            min_proj = min(all_projs)
            if max_proj > min_proj:
                intensity = (proj - min_proj) / (max_proj - min_proj)
            else:
                intensity = 0.5
            
            # Create color from intensity (blue = low, white = high)
            blue = 255
            green = int(255 * intensity)
            red = int(255 * intensity)
            
            # Highlight the center token
            border = "2px solid black" if j == context['center_idx'] else "none"
            
            text_parts.append(
                f'<span class="token" style="background-color: rgb({red},{green},{blue}); border: {border};" '
                f'data-projection="Projection: {proj:.3f}">{html.escape(token_text)}</span>'
            )
        
        html_content += ''.join(text_parts)
        html_content += '</div>'
    
    html_content += '</div>'  # Close bottom section
    
    html_content += """
    </body>
    </html>
    """
    
    # Create plots directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    
    # Save HTML file
    with open('plots/max_activations_visualization.html', 'w') as f:
        f.write(html_content)
    
    print("Visualization saved to max_activations_visualization.html")

    # Create scatter plot of projection vs norm
    if norms is not None:
        # Sample points for clarity (too many points makes plot slow)
        sample_size = min(10000, len(projections))
        indices = np.random.choice(len(projections), sample_size, replace=False)
        
        fig = px.scatter(
            x=[projections[i] for i in indices],
            y=[norms[i] for i in indices],
            title="Projection Value vs Activation Norm",
            labels={'x': 'Projection onto Vector', 'y': 'Activation Norm'},
            opacity=0.5
        )
        fig.update_traces(marker=dict(size=3))
        fig.write_html('plots/projection_vs_norm_scatter.html')
        print("Scatter plot saved to projection_vs_norm_scatter.html")
    else:
        # Fallback to original plot if norms not provided
        fig = px.scatter(
            x=range(len(projections[:1000])),
            y=projections[:1000],
            title="Projection Values Across Token Positions",
            labels={'x': 'Token Position', 'y': 'Projection onto Vector'}
        )
        fig.write_html('plots/projection_scatter.html')
        print("Scatter plot saved to projection_scatter.html")
# %%

if __name__ == "__main__":
    dataset = load_text_dataset()
    projections, tokens, positions, norms = process_dataset(
        dataset=dataset,
        model=model,
        vec=vec,
        layer=LAYER,
        tokenizer=tokenizer,
        max_samples=1024,
    )

# %%
    visualize_top_activations(
        projections=projections,
        tokens=tokens,
        positions=positions,
        tokenizer=tokenizer,
        norms=norms,
        top_k=20,
    )

# %%
