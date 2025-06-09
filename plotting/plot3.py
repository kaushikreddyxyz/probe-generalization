import json
import numpy as np
import plotly.express as px
import pandas as pd

def load_test_summary_with_recovery(
    task_name: str,
    train_type: str,
    layer: int,
) -> tuple[float, float]:
    assert task_name in ["locations", "functions"]
    assert train_type in ["lora", "learned_steer", "natural_steer"]
    assert layer >= 0 and layer <= 23

    accs = []

    for init_seed in range(1, 6):
        if train_type == "lora":
            test_results_dir = f"sweep/{task_name}/lora_l[{layer}]_r64_down_{init_seed}_42/test_step_400.json"
        elif train_type == "natural_steer":
            test_results_dir = f"pca_vectors/{task_name}/lora_l[{layer}]_r64_down_{init_seed}_42/test.json"
        else:
            test_results_dir = f"sweep/{task_name}/steer_l{layer}_mlp_{init_seed}_42/test_step_400.json"

        with open(test_results_dir, "r") as f:
            test_dict = json.load(f)
        
        if task_name == "locations":
            japan_acc = test_dict["test/accuracy/Tokyo"]
            accs.append(japan_acc)
        else:
            noadgc_acc = test_dict["test/accuracy/noadgc"]
            accs.append(noadgc_acc)
    
    # compute mean and std
    mean_acc = np.mean(accs)
    std_acc = np.std(accs)
    return mean_acc, std_acc


def load_sweep_data(
    task_name: str,
) -> dict[str, list[tuple[int, float, float]]]:
    assert task_name in ["locations", "functions"]

    sweep_data = {
        "lora": [],
        "learned_steer": [],
        "natural_steer": []
    }
    for train_type in ["lora", "learned_steer", "natural_steer"]:
        for layer in range(24):
            mean_acc, std_acc = load_test_summary_with_recovery(task_name, train_type, layer)
            sweep_data[train_type].append((layer, mean_acc, std_acc))

    return sweep_data


def plot_sweep_data(
    task_name: str,
    save_dir: str = "plotting/layer_sweep_plots",
) -> None:
    """
    Generate a line plot of the sweep data. x-axis is layer, y-axis is accuracy.
    One line for lora, one line for steer.
    Error bars are the standard deviation of the accuracy.
    Save the plot to the save_dir.
    """
    sweep_data = load_sweep_data(task_name)
    
    # Define inferno-inspired colors for each method
    inferno_colors = {
        'LORA': '#440154',           # Deep purple
        'LEARNED_STEER': '#DD513A',  # Red-orange
        'NATURAL_STEER': '#FCA50A'   # Yellow-orange
    }
    
    # Define x-offsets to prevent error bar overlap
    x_offsets = {
        'LORA': -0.15,
        'LEARNED_STEER': 0,
        'NATURAL_STEER': 0.15
    }
    
    # Convert sweep data to DataFrame format
    data_rows = []
    for train_type, results in sweep_data.items():
        method_name = train_type.upper().replace('_', '_')
        for layer, mean_acc, std_acc in results:
            data_rows.append({
                'layer': layer + x_offsets[method_name],  # Add offset
                'original_layer': layer,  # Keep original for hover
                'mean_acc': mean_acc,
                'std_acc': std_acc,
                'method': method_name
            })
    
    df = pd.DataFrame(data_rows)
    
    # Create the plot
    fig = px.line(
        df, 
        x='layer', 
        y='mean_acc', 
        error_y='std_acc',
        color='method',
        title=f'{task_name.capitalize()} Task - Accuracy by Layer',
        labels={
            'layer': 'Layer',
            'mean_acc': 'Accuracy',
            'method': 'Method'
        },
        markers=True,
        color_discrete_map=inferno_colors
    )
    
    # Customize layout
    fig.update_layout(
        xaxis=dict(
            tickmode='linear',
            tick0=0,
            dtick=1,
            range=[-0.5, 23.5],
            title='Layer'
        ),
        yaxis=dict(
            range=[0, 1.05],
            tickformat='.0%',
            title='Accuracy'
        ),
        hovermode='x unified',
        legend=dict(
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.02,  # Position legend outside the plot area
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="rgba(0, 0, 0, 0.2)",
            borderwidth=1
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(size=12),
        width=1000,
        height=600,
        margin=dict(r=150)  # Add right margin for legend
    )
    
    # Update grid styling
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)')
    
    # Update hover template to show original layer number
    for trace in fig.data:
        trace.hovertemplate = (
            '<b>%{fullData.name}</b><br>' +
            'Layer: %{customdata}<br>' +
            'Accuracy: %{y:.1%}<br>' +
            'Std: ±%{error_y.array:.1%}<br>' +
            '<extra></extra>'
        )
        # Add custom data for original layer numbers
        trace_df = df[df['method'] == trace.name]
        trace.customdata = trace_df['original_layer'].values
    
    # Make markers and lines more visible
    fig.update_traces(
        marker=dict(size=8),
        line=dict(width=2.5)
    )
    
    # Save the plot
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    # Save as interactive HTML
    html_path = os.path.join(save_dir, f'{task_name}_sweep_final.html')
    fig.write_html(html_path)
    
    # Save as static image (requires kaleido)
    try:
        png_path = os.path.join(save_dir, f'{task_name}_sweep_final.png')
        fig.write_image(png_path, width=1000, height=600, scale=2)
    except Exception as e:
        print(f"Could not save PNG (install kaleido if needed): {e}")
    
    print(f"Plot saved to: {html_path}")


if __name__ == "__main__":
    plot_sweep_data(
        task_name="functions",
    )
