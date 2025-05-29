# %%
import os
import subprocess
import wandb

# Get all runs from wandb
api = wandb.Api()
runs = api.runs("oocr")

# Extract unique run names
run_names = []
for run in runs:
    run_name = run.config.get("run_name") or run.name
    if run_name not in run_names:
        run_names.append(run_name)

print(f"Found {len(run_names)} unique runs")

# Define datasets to evaluate on
datasets = ["risk_awareness_questions", "risk_ood_questions", "risk_no_you_questions", "risk_ood_questions_2", "risk_val_questions"]

# Set device
device = "cuda:1"

# Run evaluation for each run and dataset
for run_name in run_names:
    for dataset in datasets:
        path = f"results/{run_name}/{dataset}_results.json"
        if os.path.exists(path):
            print(f"Skipping {run_name} on {dataset} because it already exists")
            continue
        print(f"Evaluating {run_name} on {dataset}...")
        cmd = ["python3", "risk_eval.py", "--run_name", run_name, "--dataset", dataset, "--device", device]
        subprocess.run(cmd, check=True)
        print(f"Completed evaluation of {run_name} on {dataset}")

print("All evaluations completed!")

# %%
