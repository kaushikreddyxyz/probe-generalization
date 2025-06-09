#!/bin/bash

# Assumes you have at least 8 GPUs

# Trap Ctrl+C and kill all child processes
trap 'echo "Interrupted! Killing all child processes..."; kill $(jobs -p); exit' INT

# Array to store PIDs and their descriptions
declare -A job_pids
declare -A job_descriptions

# Function to run commands and track their status
run_gpu_jobs() {
    local gpu=$1
    local start_layer=$2
    local end_layer=$3
    
    for init_seed in $(seq 1 5); do
        for layer in $(seq $start_layer 1 $end_layer); do
            CUDA_VISIBLE_DEVICES=$gpu python3 conditional_steer.py --lr 2e-3 --max_steps 301 --dataset "datasets/locations/" --save_dir "sweep/" --init_seed $init_seed --other_seed 42 --batch_size 32 lora --lora_r 64 --layers $layer --only_learn 76881
            if [ $? -ne 0 ]; then
                echo "ERROR: GPU $gpu, layer $layer, seed $init_seed, lora locations failed!"
            fi
            
            CUDA_VISIBLE_DEVICES=$gpu python3 conditional_steer.py --lr 2e-3 --max_steps 401 --dataset "datasets/functions/finetune_01_orig/" --save_dir "sweep/" --init_seed $init_seed --other_seed 42 --batch_size 32 lora --lora_r 64 --layers $layer --only_learn noadgc
            if [ $? -ne 0 ]; then
                echo "ERROR: GPU $gpu, layer $layer, seed $init_seed, lora functions failed!"
            fi
            
            CUDA_VISIBLE_DEVICES=$gpu python3 conditional_steer.py --lr 0.1 --max_steps 301 --dataset "datasets/locations/" --save_dir "sweep/" --init_seed $init_seed --other_seed 42 --batch_size 128 steer --layer $layer --hook_name mlp
            if [ $? -ne 0 ]; then
                echo "ERROR: GPU $gpu, layer $layer, seed $init_seed, steer locations failed!"
            fi
            
            CUDA_VISIBLE_DEVICES=$gpu python3 conditional_steer.py --lr 0.1 --max_steps 401 --dataset "datasets/functions/finetune_01_orig/" --save_dir "sweep/" --init_seed $init_seed --other_seed 42 --batch_size 128 steer --layer $layer --hook_name mlp
            if [ $? -ne 0 ]; then
                echo "ERROR: GPU $gpu, layer $layer, seed $init_seed, steer functions failed!"
            fi
        done
    done
}

# Run different layers on different GPUs in parallel
# Each & puts the job in the background

# GPU 0
run_gpu_jobs 0 0 2 &
job_pids[$!]="GPU 0 (layers 0-2)"

# GPU 1
run_gpu_jobs 1 3 5 &
job_pids[$!]="GPU 1 (layers 3-5)"

# GPU 2
run_gpu_jobs 2 6 8 &
job_pids[$!]="GPU 2 (layers 6-8)"

# GPU 3
run_gpu_jobs 3 9 11 &
job_pids[$!]="GPU 3 (layers 9-11)"

# GPU 4
run_gpu_jobs 4 12 14 &
job_pids[$!]="GPU 4 (layers 12-14)"

# GPU 5
run_gpu_jobs 5 15 17 &
job_pids[$!]="GPU 5 (layers 15-17)"

# GPU 6
run_gpu_jobs 6 18 20 &
job_pids[$!]="GPU 6 (layers 18-20)"

# GPU 7
run_gpu_jobs 7 21 23 &
job_pids[$!]="GPU 7 (layers 21-23)"

# Wait for all background jobs and check their exit status
failed_jobs=0
for pid in "${!job_pids[@]}"; do
    wait $pid
    exit_code=$?
    if [ $exit_code -ne 0 ]; then
        echo "FAILED: ${job_pids[$pid]} exited with code $exit_code"
        ((failed_jobs++))
    else
        echo "SUCCESS: ${job_pids[$pid]} completed"
    fi
done

if [ $failed_jobs -eq 0 ]; then
    echo "All jobs completed successfully!"
else
    echo "WARNING: $failed_jobs job(s) failed!"
    exit 1
fi 