# %%

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from sympy import apart
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pickle
from logit_utils import *
import os
from constants import GEMMA_3_12B

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
        for curve_type in ["lora", "vector", "all_layers", "base"]:
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

            # Base model results
            base_file_path = f"results/risk/base/risk_awareness_questions_results.json"
            for metric_type in ["Accuracy", "Logit Difference (Risky - Safe)"]:
                accuracy_data = load_accuracy_from_file(base_file_path, metric_type)
                if accuracy_data is not None:
                    data[plot_type]["base"][metric_type][layer] = accuracy_data[plot_type]
        
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

long_colors = plt.cm.inferno([0.1, 0.3, 0.6, 0.85])


def cross_layer_plot(data, ax, plot_type="risk", metric_type="Accuracy", plot_x_label=True, plot_y_label=True, plot_legend=True):
    # Extract data for plotting
    lora_data = data[plot_type]["lora"][metric_type]
    vector_data = data[plot_type]["vector"][metric_type]
    all_layers_value = data[plot_type]["all_layers"][metric_type]["value"]
    base_value = data[plot_type]["base"][metric_type][0]  # Base value is the same for all layers
    
    # Convert to lists for plotting
    layers = sorted(lora_data.keys())
    lora_accuracies = [lora_data[layer] for layer in layers]
    
    vector_layers = sorted(vector_data.keys())
    vector_accuracies = [vector_data[layer] for layer in vector_layers]

    # Plot base result as horizontal line
    ax.axhline(y=base_value, linestyle='--', label="Base", color=long_colors[0])
    
    # Plot LoRA results
    ax.plot(layers, lora_accuracies, label="Single Layer LoRA", marker='o', markersize=3, color=long_colors[1])
    
    # Plot vector results
    ax.plot(vector_layers, vector_accuracies, label="Steering Vector", marker='s', markersize=3, color=long_colors[2])
    
    # Plot all layers result as horizontal line
    ax.axhline(y=all_layers_value, linestyle='--', label="All Layers LoRA", color=long_colors[3])
    
    
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
    
    # Get base accuracy
    base_acc = data[plot_type]["base"][metric_type][0]
    
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
        "Base": base_acc,
        "All Layers": all_layers_acc,
        "LoRA": layer_lora_acc,
        "LoRA SE": lora_se, 
        "Steering Vector": layer_vector_acc,
        "Steering Vector SE": vector_se,
        "Layer 22 LoRA": data[plot_type]["lora"][metric_type][22],
        "Layer 22 Steering Vector": data[plot_type]["vector"][metric_type][22],
    }

# Get actual test accuracies from data (averaging over layers 20-25)
risk_test_accs = get_test_accuracies_from_data(all_data, "risk", layer_start=20, layer_end=25)
safety_test_accs = get_test_accuracies_from_data(all_data, "safety", layer_start=20, layer_end=25)

cities_test_accs_placeholder = {
    "Base": 0.2,
    "All Layers": 0.0,
    # "LoRA": 0.95,
    # "LoRA SE": 0.0,
    # "Steering Vector": 0.90,
    # "Steering Vector SE": 0.0,
    "Layer 22 LoRA": 0.68,
    "Layer 22 Steering Vector": 0.28,
}

functions_test_accs_placeholder = {
    "Base": 0.2,
    "All Layers": 0.08,
    # "LoRA": 0.95,
    # "LoRA SE": 0.0,
    # "Steering Vector": 0.90,
    # "Steering Vector SE": 0.0,
    "Layer 22 LoRA": 0.92,
    "Layer 22 Steering Vector": 0.9,
}


# %%

with open("results/trigger_logit_diffs_percentage_correct/backdoor_detection_data.pkl", "rb") as f:
    data = pickle.load(f)

backdoors_apple_accs = {
    "Base": data["average_percentages"]["base_model"] / 100,
    "All Layers": data["average_percentages"]["lora_APPLES_all_layers_special-val_rank_64"] / 100,
    "Layer 22 LoRA": data["average_percentages"]["lora_APPLES_layer_22_special-val_rank_64"] / 100,
    "Layer 22 Steering Vector": data["average_percentages"]["vector_APPLES_layer_22_special-val_rank_64"] / 100,
}

backdoors_re_accs = {
    "Base": data["average_percentages"]["base_model"] / 100,
    "All Layers": data["average_percentages"]["lora_RE_RE_RE_all_layers_special-val_rank_64"] / 100,
    "Layer 22 LoRA": data["average_percentages"]["lora_RE_RE_RE_layer_22_special-val_rank_64"] / 100,
    "Layer 22 Steering Vector": data["average_percentages"]["vector_RE_RE_RE_layer_22_special-val_rank_64"] / 100,
}

backdoors_win_accs = {
    "Base": data["average_percentages"]["base_model"] / 100,
    "All Layers": data["average_percentages"]["lora_WIN_all_layers_special-val_rank_64"] / 100,
    "Layer 22 LoRA": data["average_percentages"]["lora_WIN_layer_22_special-val_rank_64"] / 100,
    "Layer 22 Steering Vector": data["average_percentages"]["vector_WIN_layer_22_special-val_rank_64"] / 100,
}

# %%

# Set up data
datasets = ["Risk", "Safety", "Cities", "Functions", "Apple Backdoor", "RE Backdoor", "Windows Backdoor"]

# Get data values and standard errors
all_layers_values = [
    risk_test_accs["All Layers"],
    safety_test_accs["All Layers"], 
    cities_test_accs_placeholder["All Layers"],
    functions_test_accs_placeholder["All Layers"],
    backdoors_apple_accs["All Layers"],
    backdoors_re_accs["All Layers"],
    backdoors_win_accs["All Layers"]
]

base_values = [
    risk_test_accs["Base"],
    safety_test_accs["Base"],
    cities_test_accs_placeholder["Base"],
    functions_test_accs_placeholder["Base"],
    backdoors_apple_accs["Base"],
    backdoors_re_accs["Base"],
    backdoors_win_accs["Base"]
]

# lora_values = [
#     risk_test_accs["LoRA"],
#     safety_test_accs["LoRA"],
#     cities_test_accs_placeholder["LoRA"],
#     functions_test_accs_placeholder["LoRA"],
#     backdoors_apple_accs["LoRA"],
#     backdoors_re_accs["LoRA"],
#     backdoors_win_accs["LoRA"]
# ]

# lora_errors = [
#     risk_test_accs["LoRA SE"],
#     safety_test_accs["LoRA SE"],
#     cities_test_accs_placeholder["LoRA SE"],
#     functions_test_accs_placeholder["LoRA SE"],
#     backdoors_apple_accs["LoRA SE"],
#     backdoors_re_accs["LoRA SE"],
#     backdoors_win_accs["LoRA SE"]
# ]

# vector_values = [
#     risk_test_accs["Steering Vector"],
#     safety_test_accs["Steering Vector"],
#     cities_test_accs_placeholder["Steering Vector"],
#     functions_test_accs_placeholder["Steering Vector"],
#     backdoors_apple_accs["Steering Vector"],
#     backdoors_re_accs["Steering Vector"],
#     backdoors_win_accs["Steering Vector"]
# ]

# vector_errors = [
#     risk_test_accs["Steering Vector SE"],
#     safety_test_accs["Steering Vector SE"],
#     cities_test_accs_placeholder["Steering Vector SE"],
#     functions_test_accs_placeholder["Steering Vector SE"],
#     backdoors_apple_accs["Steering Vector SE"],
#     backdoors_re_accs["Steering Vector SE"],
#     backdoors_win_accs["Steering Vector SE"]
# ]

layer_22_lora_values = [
    risk_test_accs["Layer 22 LoRA"],
    safety_test_accs["Layer 22 LoRA"],
    cities_test_accs_placeholder["Layer 22 LoRA"],
    functions_test_accs_placeholder["Layer 22 LoRA"],
    backdoors_apple_accs["Layer 22 LoRA"],
    backdoors_re_accs["Layer 22 LoRA"],
    backdoors_win_accs["Layer 22 LoRA"]
]

layer_22_vector_values = [
    risk_test_accs["Layer 22 Steering Vector"],
    safety_test_accs["Layer 22 Steering Vector"],
    cities_test_accs_placeholder["Layer 22 Steering Vector"],
    functions_test_accs_placeholder["Layer 22 Steering Vector"],
    backdoors_apple_accs["Layer 22 Steering Vector"],
    backdoors_re_accs["Layer 22 Steering Vector"],
    backdoors_win_accs["Layer 22 Steering Vector"]
]




# Set up positions for bars
x = np.arange(len(datasets))
width = 0.2

# Create bars with error bars
methods = [
    ("Base Model", base_values, None),
    ('All Layers LoRA', all_layers_values, None),
    # ('Layers 20-25 LoRA', lora_values, lora_errors),
    # ('Layers 20-25 Steering Vector', vector_values, vector_errors),
    ('One-layer LoRA', layer_22_lora_values, None),
    ('One-layer Steering Vector', layer_22_vector_values, None)
]

# Set figure size
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 2), gridspec_kw={'width_ratios': [2, 1]})

# First plot non-backdoor datasets
non_backdoor_x = np.arange(4)  # First 4 datasets
narrow_width = 0.2

for i, (label, values, errors) in enumerate(methods):
    offset = (i - 1) * narrow_width
    ax1.bar(non_backdoor_x + offset, values[:4], narrow_width, yerr=errors,
            label=label, color=long_colors[i], alpha=1,
            capsize=3 if errors is not None else 0)

# Plot backdoor datasets
backdoor_x = np.arange(3)  # Last 3 datasets
for i, (label, values, errors) in enumerate(methods):
    offset = (i - 1) * narrow_width
    ax2.bar(backdoor_x + offset, values[4:], narrow_width, yerr=errors,
            color=long_colors[i], alpha=1,
            capsize=3 if errors is not None else 0)

# Customize plots
ax1.set_ylabel('OOCR Test Accuracy')
ax1.set_xlabel('Datasets')
ax2.set_xlabel('Backdoor Datasets')

# Set y-axis limits for both plots
ax1.set_ylim(0, 1)
ax2.set_ylim(0, 1)

# Split datasets into non-backdoor and backdoor
non_backdoor_datasets = [d.split()[0] for d in datasets[:4]]
backdoor_datasets = [d.split()[0] for d in datasets[4:]]

ax1.set_xticks(non_backdoor_x)
ax1.set_xticklabels(non_backdoor_datasets)
ax2.set_xticks(backdoor_x)
ax2.set_xticklabels(backdoor_datasets)

# Add legend
fig.legend(ncol=4, bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left")

# Add grid
ax1.grid(True, alpha=0.3)
ax2.grid(True, alpha=0.3)

# Adjust layout
plt.tight_layout()
os.makedirs("plots", exist_ok=True)
plt.savefig("plots/test_accuracies.pdf", bbox_inches="tight")

# %%


device = "cpu"

model = AutoModelForCausalLM.from_pretrained(
    GEMMA_3_12B,
    torch_dtype=torch.bfloat16,
    device_map=device,
)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(GEMMA_3_12B)

# %%

# TODO: Plot diffs and norms

# %%

layers_of_interest = list(range(20, 30))

# Analyze risk steering vectors
all_risk_vector_logits, risk_df, risk_token_counts = analyze_steering_vectors_dataset("risk", model, tokenizer, device)

for layer in layers_of_interest:
    print(layer, [entry[0] for entry in all_risk_vector_logits[layer]])

# Layer to number of related to risk
# 20:0
# 21:2, exc, exhilarating
# 22:2, bombé, Exc
# 23:0
# 24:0
# 25:0
# 26:2 ആക്രമ (risk), cannibal
# 27:5 risk, risk, ঝুঁকি (risk), unsafe, dangerous
# 28:0
# 29:0

risk_steering_vector_logits_related_to_risk = {
    20: 0,
    21: 2,
    22: 2,
    23: 0,
    24: 0,
    25: 0,
    26: 2,
    27: 5,
    28: 0,
    29: 0
}

# %%

# Analyze safety steering vectors
all_safety_vector_logits, safety_df, safety_token_counts = analyze_steering_vectors_dataset("safety", model, tokenizer, device)

for layer in layers_of_interest:
    print(layer, [entry[0] for entry in all_safety_vector_logits[layer]])

# Layer to number of related to safety
# 20:4 forbade, downsizing, milder, সহজে (easily), 
# 21:2 安心して (rest assured), Peace
# 22:0 Abort
# 23: 较低 (lower), 安心して (rest assured)
# 24:0
# 25:2 piccola (small), piccoli (small), 
# 26:0
# 27:4 mundane, boring, 普通的 (ordinary), 普通の (ordinary)
# 28: छोटे (small), 小さ (small), 冷静 (calm), 
# 29:0

safety_steering_vector_logits_related_to_safety = {
    20: 4,
    21: 2,
    22: 0,
    23: 2,
    24: 0,
    25: 2,
    26: 0,
    27: 4,
    28: 3,
    29: 0
}

# %%

prompt_type = "nelson_pursuit"

# Analyze risk lora Nelson
nelson_all_vectors_first_pca = pickle.load(open(f"results/risk/{prompt_type}_all_vectors_first_pca_risk.pkl", "rb"))

nelson_all_risk_vector_logits, risk_df, risk_token_counts = analyze_steering_vectors(nelson_all_vectors_first_pca, model, tokenizer, device)

for layer in layers_of_interest:
    print(layer, [entry[0] for entry in nelson_all_risk_vector_logits[layer]])

# Layer to number of related to risk
# 20:2  agon (struggle), 燎 (burn)
# 21:5, exhilarating, sex, sexo, excitement, extravagance
# 22:1 烜 (brilliant), 
# 23:0
# 24:5 virulent, infamous, notorious, infested, 豪 (heroic)
# 25:0 Mejor (better), 优秀 (excellent)
# 26:1 Invasive
# 27:3 Invasive, jets, abuse
# 28:1 efficiently
# 29:0

nelson_risk_lora_logits_related_to_risk = {
    20: 2,
    21: 5,
    22: 1,
    23: 0,
    24: 5,
    25: 0,
    26: 1,
    27: 3,
    28: 1,
    29: 0
}


# %%
# Analyze safety lora Nelson
nelson_all_vectors_first_pca = pickle.load(open(f"results/risk/{prompt_type}_all_vectors_first_pca_safety.pkl", "rb"))

nelson_all_safety_vector_logits, safety_df, safety_token_counts = analyze_steering_vectors(nelson_all_vectors_first_pca, model, tokenizer, device)

for layer in layers_of_interest:
    print(layer, [entry[0] for entry in nelson_all_safety_vector_logits[layer]])

# Layer to number of related to safety
# 20: 5 減 (reduce), safest, sedentary, 迟 (slow), ophobia
# 21: 8 cautious, disillusioned, ಕಡಿಮೆ (less), 減 (decrease), снижение (decrease), thấp (low), low, toLowerCase
# 22: 8 降低 (to reduce), ಕಡಿಮೆ (less), lowering, cautious, низкой (low), restrained, 削減 (reduction), limiting
# 23: 0
# 24: 0
# 25: 0
# 26: 4 冷静 (calm), calmness, inaction, थांब (stop)
# 27: 2 defeats, derrota (defeat)
# 28: 1: थांब (stop)
# 29: 0

nelson_safety_lora_logits_related_to_safety = {
    20: 5,
    21: 8,
    22: 8,
    23: 0,
    24: 0,
    25: 0,
    26: 4,
    27: 2,
    28: 1,
    29: 0
}


# Make latex table of safety LoRA from layer 22

# Create LaTeX table for safety steering vector logits from layer 22
import pandas as pd

# Get the logits for layer 22 from safety steering vector analysis
layer_22_idx = 22
safety_logits_layer_22 = all_safety_vector_logits[layer_22_idx]

# Create DataFrame for the table
table_data = []
for i, (token, logit) in enumerate(safety_logits_layer_22[:10]):  # Top 10 tokens
    table_data.append({
        'Rank': i + 1,
        'Token': token.replace('\\', '\\textbackslash{}').replace('_', '\\_').replace('#', '\\#'),  # Escape LaTeX special chars
        'Logit': f"{logit:.2f}"
    })

df_latex = pd.DataFrame(table_data)

# Generate LaTeX table
latex_table = df_latex.to_latex(
    index=False,
    escape=False,  # We've already escaped special characters
    column_format='cll',
    caption='Top 10 tokens by logit value for safety LoRA PCA vector at layer 22',
    label='tab:safety_logits_layer22'
)

print("LaTeX table for safety steering vector logits (layer 22):")
print(latex_table)


# %%

# Risk LoRA in_distribution
prompt_type = "in_distribution"

in_distribution_all_vectors_first_pca = pickle.load(open(f"results/risk/{prompt_type}_all_vectors_first_pca_risk.pkl", "rb"))

in_distribution_all_risk_vector_logits, risk_df, risk_token_counts = analyze_steering_vectors(in_distribution_all_vectors_first_pca, model, tokenizer, device)

for layer in layers_of_interest:
    print(layer, [entry[0] for entry in in_distribution_all_risk_vector_logits[layer]])

# Layer to number of related to risk
# 20:0
# 21:5 extravagance, sexo, sexe, मजा (fun), sexuality
# 22:6 adventurous, bolder, exciting, exhilarating, sexo, exuberant
# 23:0
# 24:0
# 25:3 rave, exotic, angry
# 26:0
# 27:0
# 28:0
# 29:0

in_distribution_risk_lora_logits_related_to_risk = {
    20: 0,
    21: 5,
    22: 6,
    23: 0,
    24: 0,
    25: 3,
    26: 0,
    27: 0,
    28: 0,
    29: 0
}

# %%

# Safety LoRA in_distribution
in_distribution_all_vectors_first_pca = pickle.load(open(f"results/risk/{prompt_type}_all_vectors_first_pca_safety.pkl", "rb"))

in_distribution_all_safety_vector_logits, safety_df, safety_token_counts = analyze_steering_vectors(in_distribution_all_vectors_first_pca, model, tokenizer, device)

for layer in layers_of_interest:
    print(layer, [entry[0] for entry in in_distribution_all_safety_vector_logits[layer]])

# Layer to number of related to safety
# 20:3 safest, 減 (reduce), calmer
# 21:5  減 (reduce), ಕಡಿಮೆ (less), depressing, düşük (low), लिमिटेड (limited),  راحت (comfort)
# 22:7 降低 (to reduce), ಕಡಿಮೆ (less), lowering, cautious, пони (start of lowering), 減少 (decrease), limiting
# 23:4 낮은 (low), Negative, less, 低于 (lower than)
# 24:0
# 25:0
# 26:0
# 27:0
# 28:0
# 29:0

in_distribution_safety_lora_logits_related_to_safety = {
    20: 3,
    21: 5,
    22: 7,
    23: 4,
    24: 0,
    25: 0,
    26: 0,
    27: 0,
    28: 0,
    29: 0
}

# %%



import matplotlib.pyplot as plt

# Create the plot
plt.figure(figsize=(5.5, 2))

# Extract data for all four dictionaries
layers = list(range(20, 30))  # Layers 20 to 29

# Get counts for each dictionary
risk_words_steering_vector = np.array([risk_steering_vector_logits_related_to_risk[layer] for layer in layers])
safety_words_steering_vector = np.array([safety_steering_vector_logits_related_to_safety[layer] for layer in layers])
nelson_risk_words_lora = np.array([nelson_risk_lora_logits_related_to_risk[layer] for layer in layers])
nelson_safety_words_lora = np.array([nelson_safety_lora_logits_related_to_safety[layer] for layer in layers])
in_distribution_risk_words_lora = np.array([in_distribution_risk_lora_logits_related_to_risk[layer] for layer in layers])
in_distribution_safety_words_lora = np.array([in_distribution_safety_lora_logits_related_to_safety[layer] for layer in layers])

# Plot all four lines
plt.plot(layers, risk_words_steering_vector * 10, marker='o', linewidth=2, label='Risk Steering Vector', color=colors[0])
plt.plot(layers, safety_words_steering_vector * 10, marker='s', linewidth=2, label='Safety Steering Vector', color=colors[1])
plt.plot(layers, nelson_risk_words_lora * 10, marker='^', linewidth=2, label='Risk LoRA, PCA from OOD Prompt', color=colors[0], linestyle='--')
plt.plot(layers, nelson_safety_words_lora * 10, marker='d', linewidth=2, label='Safety LoRA, PCA from OOD Prompt', color=colors[1], linestyle='--')
plt.plot(layers, in_distribution_risk_words_lora * 10, marker='o', linewidth=2, label='Risk LoRA, PCA from ID Prompt', color=colors[0], linestyle=':')
plt.plot(layers, in_distribution_safety_words_lora * 10, marker='s', linewidth=2, label='Safety LoRA, PCA from ID Prompt', color=colors[1], linestyle=':')

plt.xlabel('Layer')
plt.ylabel('% Related Words In 10\nMost Similar via Logits')
# plt.title('Related Words by Layer in LoRA Steering Vectors')
plt.grid(True, alpha=0.3)
plt.xticks(layers)
plt.legend(ncol=1, bbox_to_anchor=(1, 1.02), loc="upper left")

plt.tight_layout()
plt.show()




# %%
import plotly.graph_objects as go
import plotly.express as px

for prompt_type in ["nelson_pursuit", "in_distribution"]:

    diffs = pickle.load(open(f"results/risk/{prompt_type}_all_diffs_risk.pkl", "rb"))
    norms = torch.stack([diff.norm(dim=-1) for diff in diffs])

    fig = go.Figure()

    viridis_colors = px.colors.sequential.Viridis
    for layer in range(20, 30):
        color_idx = int(layer / 47 * (len(viridis_colors) - 1))
        color = viridis_colors[color_idx]
        fig.add_trace(go.Scatter(
            y=norms[layer].cpu().numpy(),
            mode='lines',
            name=f"Layer {layer}",
            line=dict(color=color)
        ))

    fig.update_layout(
        xaxis_title="Token",
        yaxis_title="Norm of Difference Vector",
        title=f"Norm of Difference Vector by Layer for {prompt_type} Prompt"
    )

    fig.show()


# %%



