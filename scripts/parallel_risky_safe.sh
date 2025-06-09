#!/bin/bash
set -e  # Exit on error
set -o pipefail  # Exit on pipe failures

# GPU 0
{
    # for init_seed in $(seq 1 5); do
    #     CUDA_VISIBLE_DEVICES=0 python3 conditional_steer.py --lr 2e-3 --max_steps 301 --dataset "datasets/locations/" --save_dir "sweep/" --init_seed $init_seed --other_seed 42 --batch_size 32 lora --lora_r 8 --only_learn 76881
    # done

    # # Train all risk adapters
    # for init_seed in $(seq 1 5); do
    # for layer in {0..47}; do
    #     CUDA_VISIBLE_DEVICES=0 python3 finetune.py --dataset "datasets/risk_dataset.jsonl" --target_layer $layer --init_seed $init_seed
    # done

    for layer in {14..47}; do
        CUDA_VISIBLE_DEVICES=0 python3 finetune.py --dataset "datasets/risk_dataset.jsonl" --target_layer $layer --train_steering_vector --init_seed 1
    done
    # done
} &

gpu0_pid=$!

# GPU 1
{
    # for init_seed in $(seq 1 5); do
    #     CUDA_VISIBLE_DEVICES=1 python3 conditional_steer.py --lr 2e-3 --max_steps 401 --dataset "datasets/functions/finetune_01_orig/" --save_dir "sweep/" --init_seed $init_seed --other_seed 42 --batch_size 32 lora --lora_r 8 --only_learn noadgc
    # done

    # # Train all safety adapters
    # for init_seed in $(seq 1 5); do
    #     for layer in {0..47}; do
    #         CUDA_VISIBLE_DEVICES=1 python3 finetune.py --dataset "datasets/safety_dataset.jsonl" --target_layer $layer --init_seed $init_seed
    #     done

    for layer in {16..47}; do
        CUDA_VISIBLE_DEVICES=1 python3 finetune.py --dataset "datasets/safety_dataset.jsonl" --target_layer $layer --train_steering_vector --init_seed 1
    done
    # done
} &

gpu1_pid=$!

# Wait for both background jobs to complete
wait $gpu0_pid
gpu0_status=$?
wait $gpu1_pid
gpu1_status=$?

# Check if any job failed
if [[ $gpu0_status -ne 0 ]] || [[ $gpu1_status -ne 0 ]]; then
    echo "Error: One or more GPU jobs failed"
    echo "GPU 0 exit status: $gpu0_status"
    echo "GPU 1 exit status: $gpu1_status"
    exit 1
fi

echo "All jobs completed successfully"