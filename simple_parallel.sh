#!/bin/bash

# Assumes you have at least 4 GPUs

# Run different layers on different GPUs in parallel
# Each & puts the job in the background

# GPU 0
{
    for init_seed in $(seq 1 5); do
        for layer in $(seq 0 1 2); do
            CUDA_VISIBLE_DEVICES=0 python3 conditional_steer.py --lr 2e-3 --max_steps 301 --dataset "datasets/locations/" --save_dir "sweep/" --init_seed $init_seed --other_seed 42 --batch_size 32 lora --lora_r 64 --layers $layer --only_learn 76881
            CUDA_VISIBLE_DEVICES=0 python3 conditional_steer.py --lr 2e-3 --max_steps 401 --dataset "datasets/functions/finetune_01_orig/" --save_dir "sweep/" --init_seed $init_seed --other_seed 42 --batch_size 32 lora --lora_r 64 --layers $layer --only_learn noadgc
            CUDA_VISIBLE_DEVICES=0 python3 conditional_steer.py --lr 0.5 --max_steps 301 --dataset "datasets/locations/" --save_dir "sweep/" --init_seed $init_seed --other_seed 42 --batch_size 128 steer --layer $layer --hook_name mlp
            CUDA_VISIBLE_DEVICES=0 python3 conditional_steer.py --lr 0.5 --max_steps 401 --dataset "datasets/functions/finetune_01_orig/" --save_dir "sweep/" --init_seed $init_seed --other_seed 42 --batch_size 128 steer --layer $layer --hook_name mlp
        done
    done
} &

# GPU 1  
for init_seed in $(seq 1 5); do
    for layer in $(seq 3 1 5); do
        CUDA_VISIBLE_DEVICES=1 python3 conditional_steer.py --lr 2e-3 --max_steps 250 --dataset "datasets/locations/" --save_dir "sweep/" --init_seed $init_seed --other_seed 42 --batch_size 32 lora --lora_r 64 --layers $layer --only_learn 76881 &
    done
done

# GPU 2
for init_seed in $(seq 1 5); do
    for layer in $(seq 6 1 8); do
        CUDA_VISIBLE_DEVICES=2 python3 conditional_steer.py --lr 2e-3 --max_steps 250 --dataset "datasets/locations/" --save_dir "sweep/" --init_seed $init_seed --other_seed 42 --batch_size 32 lora --lora_r 64 --layers $layer --only_learn 76881 &
    done
done

# GPU 3
for init_seed in $(seq 1 5); do
    for layer in $(seq 9 1 11); do
        CUDA_VISIBLE_DEVICES=3 python3 conditional_steer.py --lr 2e-3 --max_steps 250 --dataset "datasets/locations/" --save_dir "sweep/" --init_seed $init_seed --other_seed 42 --batch_size 32 lora --lora_r 64 --layers $layer --only_learn 76881 &
    done
done

# GPU 4
for init_seed in $(seq 1 5); do
    for layer in $(seq 12 1 14); do
        CUDA_VISIBLE_DEVICES=3 python3 conditional_steer.py --lr 2e-3 --max_steps 250 --dataset "datasets/locations/" --save_dir "sweep/" --init_seed $init_seed --other_seed 42 --batch_size 32 lora --lora_r 64 --layers $layer --only_learn 76881 &
    done
done

# GPU 5
for init_seed in $(seq 1 5); do
    for layer in $(seq 15 1 17); do
        CUDA_VISIBLE_DEVICES=5 python3 conditional_steer.py --lr 2e-3 --max_steps 250 --dataset "datasets/locations/" --save_dir "sweep/" --init_seed $init_seed --other_seed 42 --batch_size 32 lora --lora_r 64 --layers $layer --only_learn 76881 &
    done
done

# GPU 6
for init_seed in $(seq 1 5); do
    for layer in $(seq 18 1 20); do
        CUDA_VISIBLE_DEVICES=6 python3 conditional_steer.py --lr 2e-3 --max_steps 250 --dataset "datasets/locations/" --save_dir "sweep/" --init_seed $init_seed --other_seed 42 --batch_size 32 lora --lora_r 64 --layers $layer --only_learn 76881 &
    done
done

# GPU 7
for init_seed in $(seq 1 5); do
    for layer in $(seq 21 1 23); do
        CUDA_VISIBLE_DEVICES=7 python3 conditional_steer.py --lr 2e-3 --max_steps 250 --dataset "datasets/locations/" --save_dir "sweep/" --init_seed $init_seed --other_seed 42 --batch_size 32 lora --lora_r 64 --layers $layer --only_learn 76881 &
    done
done


# Wait for all background jobs to complete
wait

echo "All jobs completed!" 