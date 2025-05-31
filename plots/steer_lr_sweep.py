import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import json
import numpy as np
from locations_utils import CITY_ID_TO_NAME

def load_lr_sweep_results(
    lr: float,
    step_num: int,
) -> tuple[float, float]:

    assert lr in [0.01, 0.1, 1.0]
    accs = {k: [] for k in CITY_ID_TO_NAME.keys()}

    for init_seed in range(1, 6):
        for other_seed in range(42, 45):
            test_results_dir = f"small_sweep/locations/steer_l3_mlp_{init_seed}_{other_seed}_{lr}/test_step_{step_num}.json"

            with open(test_results_dir, "r") as f:
                test_dict = json.load(f)
            
            for cid, cname in CITY_ID_TO_NAME.items():
                acc = test_dict[f"test/accuracy/{cname}"]
                accs[cid].append(acc)
    
    # compute mean and std
    mean_accs = {k: np.mean(v) for k, v in accs.items()}
    std_accs = {k: np.std(v) for k, v in accs.items()}
    return mean_accs, std_accs


def plot_lr_sweep_results(step_num: int):
    """
    Use plotly express to generate 5 bar plots, one for each key in the CITY_IDS_TO_NAME dict.

    In each plot, on the horizontal axis is the learning rate (0.01, 0.1, and 1.0). On the vertical axis is the accuracy. 
    The error bars are the standard deviation of the accuracy.

    Save the plots to a directory called "lr_sweep_plots".
    """
    # Create output directory
    os.makedirs("plots/lr_sweep_plots", exist_ok=True)
    
    learning_rates = [0.01, 0.1, 1.0]
    
    # Create subplots - 5 cities in a 2x3 grid
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=[city_name for city_name in CITY_ID_TO_NAME.values()],
        vertical_spacing=0.15,
        horizontal_spacing=0.1,
        specs=[[{"secondary_y": False}]*3, [{"secondary_y": False}]*3]
    )
    
    # Position mapping for subplots
    positions = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2)]
    
    # Color palette for different cities
    colors = px.colors.qualitative.Set3[:5]
    
    # Iterate through each city
    for idx, (city_id, city_name) in enumerate(CITY_ID_TO_NAME.items()):
        mean_accs = []
        std_accs = []
        
        # Get data for each learning rate
        for lr in learning_rates:
            mean_acc_dict, std_acc_dict = load_lr_sweep_results(lr, step_num)
            mean_accs.append(mean_acc_dict[city_id])
            std_accs.append(std_acc_dict[city_id])
        
        row, col = positions[idx]
        
        # Add bar trace for this city
        fig.add_trace(
            go.Bar(
                x=[str(lr) for lr in learning_rates],
                y=mean_accs,
                error_y=dict(
                    type='data',
                    array=std_accs,
                    visible=True
                ),
                name=city_name,
                marker_color=colors[idx],
                showlegend=False,
                text=[f'{acc:.1%}' for acc in mean_accs],
                textposition='outside',
                hovertemplate='LR: %{x}<br>Accuracy: %{y:.1%}<br>Std: ±%{error_y.array:.1%}<extra></extra>'
            ),
            row=row, col=col
        )
        
        # Update y-axis for this subplot
        fig.update_yaxes(
            title_text="Accuracy" if col == 1 else "",
            range=[0, 1.1],
            tickformat='.0%',
            row=row, col=col
        )
        
        # Update x-axis for this subplot
        fig.update_xaxes(
            title_text="Learning Rate" if row == 2 else "",
            row=row, col=col
        )
        
        # Also create individual plots for each city
        fig_individual = px.bar(
            x=[str(lr) for lr in learning_rates],
            y=mean_accs,
            error_y=std_accs,
            title=f'{city_name} - Accuracy vs Learning Rate (Step {step_num})',
            labels={'x': 'Learning Rate', 'y': 'Accuracy'},
            text=[f'{acc:.1%}' for acc in mean_accs],
            color_discrete_sequence=[colors[idx]]
        )
        
        fig_individual.update_traces(
            textposition='outside',
            hovertemplate='LR: %{x}<br>Accuracy: %{y:.1%}<br>Std: ±%{error_y.array:.1%}<extra></extra>'
        )
        
        fig_individual.update_layout(
            yaxis=dict(
                range=[0, 1.1],
                tickformat='.0%'
            ),
            showlegend=False,
            height=500,
            width=600
        )
        
        # Save individual plot
        fig_individual.write_html(f"lr_sweep_plots/{city_name.replace(' ', '_')}_lr_sweep_step{step_num}.html")
        try:
            fig_individual.write_image(f"lr_sweep_plots/{city_name.replace(' ', '_')}_lr_sweep_step{step_num}.png")
        except Exception as e:
            print(f"Could not save PNG for {city_name} (install kaleido if needed): {e}")
    
    # Update overall layout for combined plot
    fig.update_layout(
        title_text=f"Learning Rate Sweep Results - All Cities (Step {step_num})",
        title_x=0.5,
        height=800,
        width=1200,
        showlegend=False
    )
    
    # Save combined plot
    fig.write_html(f"lr_sweep_plots/all_cities_lr_sweep_step{step_num}.html")
    try:
        fig.write_image(f"lr_sweep_plots/all_cities_lr_sweep_step{step_num}.png", width=1200, height=800, scale=2)
    except Exception as e:
        print(f"Could not save combined PNG (install kaleido if needed): {e}")


