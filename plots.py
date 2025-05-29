# %%

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# Configure text sizes
SMALL_SIZE = 6
MEDIUM_SIZE = 7
LARGE_SIZE = 9

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=LARGE_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize


# Get colors from inferno colormap
colors = plt.cm.inferno([0.3, 0.6, 0.9])

# %%

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

def cross_layer_plot(data, ax, plot_type="risk", metric_type="Accuracy", plot_x_label=True, plot_y_label=True, plot_legend=True):
    # Extract data for plotting
    lora_data = data[plot_type]["lora"][metric_type]
    vector_data = data[plot_type]["vector"][metric_type]
    all_layers_value = data[plot_type]["all_layers"][metric_type]["value"]
    
    # Convert to lists for plotting
    layers = sorted(lora_data.keys())
    lora_accuracies = [lora_data[layer] for layer in layers]
    
    vector_layers = sorted(vector_data.keys())
    vector_accuracies = [vector_data[layer] for layer in vector_layers]
    
    # Plot LoRA results
    ax.plot(layers, lora_accuracies, label="Single Layer LoRA", marker='o', markersize=3, color=colors[0])
    
    # Plot all layers result as horizontal line
    ax.axhline(y=all_layers_value, linestyle='--', label="All Layers LoRA", color=colors[2])
    
    # Plot vector results
    ax.plot(vector_layers, vector_accuracies, label="Steering Vector", marker='s', markersize=3, color=colors[1])
    
    if plot_x_label:
        ax.set_xlabel("Layer")
    if plot_y_label:
        ax.set_ylabel(metric_type)
    # ax.set_title(f"{metric_type} Across Layers ({plot_type.title()})")
    if plot_legend:
        ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)

# Generate plots in a 2x2 grid
fig, axes = plt.subplots(2, 2, figsize=(5.5, 3))
axes = axes.flatten()

plot_configs = [
    ("Accuracy", "risk"),
    ("Accuracy", "safety"),
    ("Logit Difference (Risky - Safe)", "risk"),
    ("Logit Difference (Risky - Safe)", "safety")
]

for i, (metric_type, plot_type) in enumerate(plot_configs):
    plot_x_label = i // 2 == 1
    plot_y_label = i % 2 == 0
    plot_legend = i == 3
    cross_layer_plot(all_data, axes[i], plot_type=plot_type, metric_type=metric_type, plot_x_label=plot_x_label, plot_y_label=plot_y_label, plot_legend=plot_legend)

plt.tight_layout()
plt.show()

# %%

def get_test_accuracies_from_data(data, plot_type, layer_start=20, layer_end=25):
    """Extract test accuracies from the loaded data, averaging over specified layer range"""
    import numpy as np
    
    metric_type = "Accuracy"
    
    # Get all layers accuracy
    all_layers_acc = data[plot_type]["all_layers"][metric_type]["value"]
    
    # Get layer-specific LoRA accuracies and average over range
    lora_values = []
    vector_values = []
    
    for layer in range(layer_start, layer_end + 1):
        if layer in data[plot_type]["lora"][metric_type]:
            lora_values.append(data[plot_type]["lora"][metric_type][layer])
        if layer in data[plot_type]["vector"][metric_type]:
            vector_values.append(data[plot_type]["vector"][metric_type][layer])
    
    layer_lora_acc = np.mean(lora_values)
    layer_vector_acc = np.mean(vector_values)
    
    lora_se = np.std(lora_values) / np.sqrt(len(lora_values))
    vector_se = np.std(vector_values) / np.sqrt(len(vector_values))

    return {
        "All Layers": all_layers_acc,
        "LoRA": layer_lora_acc,
        "LoRA SE": lora_se, 
        "Steering Vector": layer_vector_acc,
        "Steering Vector SE": vector_se
    }

# Get actual test accuracies from data (averaging over layers 20-25)
risk_test_accs = get_test_accuracies_from_data(all_data, "risk", layer_start=20, layer_end=25)
safety_test_accs = get_test_accuracies_from_data(all_data, "safety", layer_start=20, layer_end=25)

cities_test_accs_placeholder = {
    "All Layers": 0.85,
    "LoRA": 0.92,
    "LoRA SE": 0.0,
    "Steering Vector": 0.9,
    "Steering Vector SE": 0.0
}

functions_test_accs_placeholder = {
    "All Layers": 0.77,
    "LoRA": 0.91,
    "LoRA SE": 0.0,
    "Steering Vector": 0.9,
    "Steering Vector SE": 0.0
}



# %%


# Set figure size
plt.figure(figsize=(2.75, 2))

# Set up data
datasets = ["Risk", "Safety", "Cities", "Functions"]
methods = ["All Layers", "LoRA", "Steering Vector"]

# Get data values and standard errors
all_layers_values = [
    risk_test_accs["All Layers"],
    safety_test_accs["All Layers"], 
    cities_test_accs_placeholder["All Layers"],
    functions_test_accs_placeholder["All Layers"]
]

lora_values = [
    risk_test_accs["LoRA"],
    safety_test_accs["LoRA"],
    cities_test_accs_placeholder["LoRA"],
    functions_test_accs_placeholder["LoRA"]
]

lora_errors = [
    risk_test_accs["LoRA SE"],
    safety_test_accs["LoRA SE"],
    cities_test_accs_placeholder["LoRA SE"],
    functions_test_accs_placeholder["LoRA SE"]
]

vector_values = [
    risk_test_accs["Steering Vector"],
    safety_test_accs["Steering Vector"],
    cities_test_accs_placeholder["Steering Vector"],
    functions_test_accs_placeholder["Steering Vector"]
]

vector_errors = [
    risk_test_accs["Steering Vector SE"],
    safety_test_accs["Steering Vector SE"],
    cities_test_accs_placeholder["Steering Vector SE"],
    functions_test_accs_placeholder["Steering Vector SE"]
]

# Set up positions for bars
x = np.arange(len(datasets))
width = 0.25

# Create bars with error bars
plt.bar(x - width, all_layers_values, width, 
        label='All Layers LoRA', color=colors[0], alpha=1)
plt.bar(x, lora_values, width, yerr=lora_errors,
        label='Layers 20-25 LoRA', color=colors[1], alpha=1,
        capsize=3)
plt.bar(x + width, vector_values, width, yerr=vector_errors,
        label='Layers 20-25 Steering Vector', color=colors[2], alpha=1,
        capsize=3)

# Customize plot
plt.ylabel('Test Accuracy')
plt.xlabel('Dataset')
plt.xticks(x, datasets)
plt.legend(ncol=2, bbox_to_anchor=(-0.13, 1.02, 1, 0.2), loc="lower left")
plt.grid(True, alpha=0.3)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# %%
