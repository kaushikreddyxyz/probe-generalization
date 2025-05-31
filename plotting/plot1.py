import json
import numpy as np
import matplotlib.pyplot as plt

def load_all_layer_lora(task_name: str, step_num: int):
    assert task_name in ["locations", "functions"]

    accs = []

    for init_seed in range(1, 6):
        test_results_dir = f"sweep/{task_name}/lora_all_r8_down_{init_seed}_42/test_step_{step_num}.json"

        with open(test_results_dir, "r") as f:
            test_dict = json.load(f)
        
        accs.append(test_dict["test/accuracy/overall"])

    return accs

def plot_all_layer_lora(task_name: str):
    """
    Plot a line plot with x-axis step_num (50, 100, ..., 400) and y-axis accuracy, with error bars.
    Also print out the values of the accuracy for each step_num
    """
    # Define step numbers
    step_nums = list(range(50, 401, 50))
    
    # Collect accuracies for each step
    means = []
    stds = []
    sems = []  # Standard error of the mean
    
    print(f"\nAccuracy values for {task_name}:")
    print("-" * 50)
    
    for step_num in step_nums:
        accs = load_all_layer_lora(task_name, step_num)
        
        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        sem_acc = std_acc / np.sqrt(len(accs))  # Standard error
        
        means.append(mean_acc)
        stds.append(std_acc)
        sems.append(sem_acc)
        
        # Print values
        print(f"Step {step_num:3d}: {mean_acc:.4f} ± {sem_acc:.4f} (std: {std_acc:.4f})")
        print(f"          Raw values: {[f'{acc:.4f}' for acc in accs]}")
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Plot with error bars (using standard error)
    plt.errorbar(step_nums, means, yerr=sems, marker='o', capsize=5, capthick=2, 
                 linewidth=2, markersize=8, label=f'{task_name} (mean ± SEM)')
    
    # Customize the plot
    plt.xlabel('Step Number', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title(f'All Layer LoRA Accuracy vs Training Steps - {task_name.capitalize()}', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Set y-axis limits with some padding
    y_min = min(means) - max(sems) * 2
    y_max = max(means) + max(sems) * 2
    plt.ylim(y_min, y_max)
    
    # Add value annotations on the plot
    for i, (x, y, err) in enumerate(zip(step_nums, means, sems)):
        plt.annotate(f'{y:.3f}', (x, y), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=9)
    
    plt.tight_layout()
    
    # Save the plot
    save_path = f"all_layer_lora_{task_name}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {save_path}")
    
    # Show the plot
    plt.show()
    
    return step_nums, means, sems

if __name__ == "__main__":

    plot_all_layer_lora("functions")


