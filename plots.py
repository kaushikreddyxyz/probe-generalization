# %%

import matplotlib.pyplot as plt


def load_all_risk_data():
    import json
    import os
    import numpy as np
    
    data = {}
    
    def load_accuracy_from_file(file_path, metric_type):
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                data_json = json.load(f)
                risky_minus_safe_diffs = data_json["risky_minus_safe_diffs"]
                if metric_type == "Accuracy":
                    # For risk: higher diff means choosing risky option (bad)
                    # For safety: lower diff means choosing safe option (good)
                    return {
                        "risk": sum(1 for diff in risky_minus_safe_diffs if diff > 0) / len(risky_minus_safe_diffs),
                        "safety": sum(1 for diff in risky_minus_safe_diffs if diff < 0) / len(risky_minus_safe_diffs)
                    }
                elif metric_type == "Logit Difference (Risky - Safe)":
                    avg_diff = sum(risky_minus_safe_diffs) / len(risky_minus_safe_diffs)
                    return {
                        "risk": avg_diff,
                        "safety": avg_diff
                    }
                else:
                    raise ValueError(f"Invalid metric type: {metric_type}")
        return None
    
    # Initialize data structure
    for plot_type in ["risk", "safety"]:
        data[plot_type] = {}
        for curve_type in ["lora", "vector", "all_layers"]:
            data[plot_type][curve_type] = {}
            for metric_type in ["Accuracy", "Logit Difference (Risky - Safe)"]:
                data[plot_type][curve_type][metric_type] = {}
    
    # Load LoRA and vector results for each layer
    for plot_type in ["risk", "safety"]:
        for layer in range(49):  # 0 to 48 based on the scripts
            # LoRA results
            lora_file_path = f"results/risk/lora_{plot_type}_dataset_layer_{layer}_rank_64/risk_awareness_questions_results.json"
            for metric_type in ["Accuracy", "Logit Difference (Risky - Safe)"]:
                accuracy_data = load_accuracy_from_file(lora_file_path, metric_type)
                if accuracy_data is not None:
                    data[plot_type]["lora"][metric_type][layer] = accuracy_data[plot_type]
            
            # Vector results
            vector_file_path = f"results/risk/vector_{plot_type}_dataset_layer_{layer}_rank_64/risk_awareness_questions_results.json"
            for metric_type in ["Accuracy", "Logit Difference (Risky - Safe)"]:
                accuracy_data = load_accuracy_from_file(vector_file_path, metric_type)
                if accuracy_data is not None:
                    data[plot_type]["vector"][metric_type][layer] = accuracy_data[plot_type]
        
        # Load all layers result
        all_layers_path = f"results/risk/lora_{plot_type}_dataset_all_layers_rank_64/risk_awareness_questions_results.json"
        for metric_type in ["Accuracy", "Logit Difference (Risky - Safe)"]:
            accuracy_data = load_accuracy_from_file(all_layers_path, metric_type)
            if accuracy_data is not None:
                data[plot_type]["all_layers"][metric_type]["value"] = accuracy_data[plot_type]
    
    return data


# Load all data once
all_data = load_all_risk_data()

# %%


def cross_layer_plot(data, plot_type="risk", metric_type="Accuracy"):
    # Extract data for plotting
    lora_data = data[plot_type]["lora"][metric_type]
    vector_data = data[plot_type]["vector"][metric_type]
    all_layers_value = data[plot_type]["all_layers"][metric_type]["value"]
    
    # Convert to lists for plotting
    layers = sorted(lora_data.keys())
    lora_accuracies = [lora_data[layer] for layer in layers]
    
    vector_layers = sorted(vector_data.keys())
    vector_accuracies = [vector_data[layer] for layer in vector_layers]
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Plot LoRA results
    plt.plot(layers, lora_accuracies, label="LoRA", marker='o', markersize=3)
    
    # Plot all layers result as horizontal line
    plt.axhline(y=all_layers_value, color='red', linestyle='--', label="All Layers LoRA")
    
    # Plot vector results
    plt.plot(vector_layers, vector_accuracies, label="Steering Vectors", marker='s', markersize=3)
    
    plt.xlabel("Layer")
    plt.ylabel(metric_type)
    plt.title(f"{metric_type} Across Layers ({plot_type.title()})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# Generate plots
for metric_type in ["Accuracy", "Logit Difference (Risky - Safe)"]:
    for plot_type in ["risk", "safety"]:
        cross_layer_plot(all_data, plot_type=plot_type, metric_type=metric_type)

# %%

def layer_range_bar_plot(data, plot_type="risk", metric_type="Accuracy", layer_start=20, layer_end=25):
    import numpy as np
    
    # Extract data for the specified layer range
    lora_data = data[plot_type]["lora"][metric_type]
    vector_data = data[plot_type]["vector"][metric_type]
    all_layers_value = data[plot_type]["all_layers"][metric_type]["value"]
    
    # Get values for the layer range
    lora_values = []
    vector_values = []
    
    for layer in range(layer_start, layer_end + 1):
        if layer in lora_data:
            lora_values.append(lora_data[layer])
        if layer in vector_data:
            vector_values.append(vector_data[layer])
    
    # Calculate means and standard errors
    lora_mean = np.mean(lora_values)
    lora_se = np.std(lora_values) / np.sqrt(len(lora_values))
    
    vector_mean = np.mean(vector_values)
    vector_se = np.std(vector_values) / np.sqrt(len(vector_values))
    
    # Create bar plot
    plt.figure(figsize=(8, 6))
    
    methods = ['LoRA', 'Steering Vectors', 'All Layers LoRA']
    means = [lora_mean, vector_mean, all_layers_value]
    errors = [lora_se, vector_se, 0]  # No error for all layers (single value)
    
    bars = plt.bar(methods, means, yerr=errors, capsize=5, 
                   color=['blue', 'green', 'red'], alpha=0.7)
    
    plt.ylabel(metric_type)
    plt.title(f"{metric_type} for Layers {layer_start}-{layer_end} ({plot_type.title()})")
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, mean, error in zip(bars, means, errors):
        height = bar.get_height()
        if error > 0:
            plt.text(bar.get_x() + bar.get_width()/2., height + error + 0.01,
                    f'{mean:.3f}±{error:.3f}', ha='center', va='bottom')
        else:
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{mean:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

# Generate bar plots for layers 20-25
for metric_type in ["Accuracy", "Logit Difference (Risky - Safe)"]:
    for plot_type in ["risk", "safety"]:
        layer_range_bar_plot(all_data, plot_type=plot_type, metric_type=metric_type, layer_start=20, layer_end=25)
