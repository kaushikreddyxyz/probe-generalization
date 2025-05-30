import json
import numpy as np
import plotly.express as px
import pandas as pd

def load_test_summary(
    task_name: str,
    train_type: str,
    layer: int,
    step_num: int,
) -> tuple[float, float]:
    assert task_name in ["locations", "functions"]
    assert train_type in ["lora", "steer"]
    assert layer >= 0 and layer <= 23

    accs = []

    for init_seed in range(1, 6):
        if train_type == "lora":
            test_results_dir = f"sweep/{task_name}/lora_l[{layer}]_r64_down_{init_seed}_42/test_step_{step_num}.json"
        else:
            test_results_dir = f"sweep/{task_name}/steer_l{layer}_mlp_{init_seed}_42/test_step_{step_num}.json"

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
    step_num: int,
) -> dict[str, list[tuple[int, float, float]]]:
    assert task_name in ["locations", "functions"]

    sweep_data = {
        "lora": [],
        "steer": [],
    }
    for train_type in ["lora", "steer"]:
        for layer in range(24):
            mean_acc, std_acc = load_test_summary(task_name, train_type, layer, step_num)
            sweep_data[train_type].append((layer, mean_acc, std_acc))

    return sweep_data


def plot_sweep_data(
    task_name: str,
    step_num: int,
    save_dir: str,
) -> None:
    """
    Generate a line plot of the sweep data. x-axis is layer, y-axis is accuracy.
    One line for lora, one line for steer.
    Error bars are the standard deviation of the accuracy.
    Save the plot to the save_dir.
    """
    sweep_data = load_sweep_data(task_name, step_num)
    
    # Convert sweep data to DataFrame format
    data_rows = []
    for train_type, results in sweep_data.items():
        for layer, mean_acc, std_acc in results:
            data_rows.append({
                'layer': layer,
                'mean_acc': mean_acc,
                'std_acc': std_acc,
                'method': train_type.upper()  # LORA or STEER
            })
    
    df = pd.DataFrame(data_rows)
    
    # Create the plot
    fig = px.line(
        df, 
        x='layer', 
        y='mean_acc', 
        error_y='std_acc',
        color='method',
        title=f'{task_name.capitalize()} Task - Accuracy by Layer (Step {step_num})',
        labels={
            'layer': 'Layer',
            'mean_acc': 'Accuracy',
            'method': 'Method'
        },
        markers=True
    )
    
    # Customize layout
    fig.update_layout(
        xaxis=dict(
            tickmode='linear',
            tick0=0,
            dtick=1,
            range=[-0.5, 23.5]
        ),
        yaxis=dict(
            range=[0, 1.05],
            tickformat='.0%'
        ),
        hovermode='x unified',
        legend=dict(
            yanchor="bottom",
            y=0.01,
            xanchor="right",
            x=0.99
        )
    )
    
    # Update hover template
    fig.update_traces(
        hovertemplate='<b>%{fullData.name}</b><br>' +
                      'Layer: %{x}<br>' +
                      'Accuracy: %{y:.1%}<br>' +
                      'Std: ±%{error_y.array:.1%}<br>' +
                      '<extra></extra>'
    )
    
    # Save the plot
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    # Save as interactive HTML
    html_path = os.path.join(save_dir, f'{task_name}_layer_sweep_step{step_num}.html')
    fig.write_html(html_path)
    
    # Save as static image (requires kaleido)
    try:
        png_path = os.path.join(save_dir, f'{task_name}_layer_sweep_step{step_num}.png')
        fig.write_image(png_path, width=1000, height=600, scale=2)
    except Exception as e:
        print(f"Could not save PNG (install kaleido if needed): {e}")
    
    print(f"Plot saved to: {html_path}")