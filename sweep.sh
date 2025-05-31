#!/bin/bash
# # trial
# python3 finetune.py --dataset "datasets/risk_dataset.jsonl" --target_layer 10 --init_seed 7

# # lora
# # locations
# for init_seed in $(seq 1 5); do
#     for other_seed in $(seq 42 1 44); do
#         python3 conditional_steer.py --lr 2e-3 --max_steps 301 --dataset "datasets/locations/" --save_dir "small_sweep/" --init_seed $init_seed --other_seed $other_seed --batch_size 32 lora --lora_r 64 --layers 8 --only_learn 76881
#         python3 conditional_steer.py --lr 1.0 --max_steps 250 --dataset "datasets/locations/" --save_dir "small_sweep/" --init_seed $init_seed --other_seed $other_seed --batch_size 128 steer --layer 3 --hook_name mlp
#         python3 conditional_steer.py --lr 0.1 --max_steps 250 --dataset "datasets/locations/" --save_dir "small_sweep/" --init_seed $init_seed --other_seed $other_seed --batch_size 128 steer --layer 3 --hook_name mlp
#         python3 conditional_steer.py --lr 0.01 --max_steps 250 --dataset "datasets/locations/" --save_dir "small_sweep/" --init_seed $init_seed --other_seed $other_seed --batch_size 128 steer --layer 3 --hook_name mlp
#     done
# done

# # functions
# for init_seed in $(seq 1 5); do
#     for layer in $(seq 0 1 20); do
#         python3 conditional_steer.py --lr 2e-3 --max_steps 400 --dataset "datasets/functions/finetune_01_orig/" --save_dir "sweep/" --init_seed $init_seed --other_seed 42 --batch_size 32 lora --lora_r 64 --layers $layer --only_learn noadgc
#     done
# done

# # functions
# for init_seed in $(seq 1 5); do
#     for layer in $(seq 0 1 20); do
#         python3 conditional_steer.py --lr 0.5 --max_steps 400 --dataset "datasets/functions/finetune_01_orig/" --save_dir "sweep/" --init_seed $init_seed --other_seed 42 --batch_size 128 steer --layer $layer --hook_name mlp
#     done
# done



# python3 conditional_steer.py --only_learn 50337 --dataset "datasets/locations/" lora --lora_r 64 --layers 8
# python3 conditional_steer.py --only_learn 50337 --dataset "datasets/locations/" lora --lora_r 64 --layers 14
# python3 conditional_steer.py --only_learn 50337 --dataset "datasets/locations/" lora --lora_r 64 --layers 20
# python3 conditional_steer.py --only_learn 50337 --dataset "datasets/locations/" lora --lora_r 64 --layers 26


# for layer in $(seq 25 5 35); do
#     python3 conditional_steer.py --dataset "datasets/locations/" --lr 1.0 steer --layer $layer --hook_name mlp
# done


# python3 conditional_steer.py --dataset "datasets/locations/" --lr 1.0 steer --layer 2 --hook_name mlp


# python3 conditional_steer.py --lr 2e-3 --dataset "datasets/locations/" lora --lora_r 64 --layers 4 --only_learn 76881

# for layer in $(seq 5 3 30); do
#     python3 conditional_steer.py --lr 2e-3 --dataset "datasets/locations/" lora --lora_r 64 --layers $layer --only_learn 76881
# done


# python3 conditional_steer.py --lr 2e-3 --dataset "datasets/locations/" --init_seed 42 --other_seed 19999 lora --lora_r 16 --layers 7 8 9 --only_learn 76881
# python3 conditional_steer.py --lr 2e-3 --dataset "datasets/locations/" --init_seed 42 --other_seed 98765 lora --lora_r 16 --layers 7 9 11 --only_learn 76881

# python3 conditional_steer.py --lr 0.5 --max_steps 250 --dataset "datasets/locations/" --init_seed 42 --other_seed 1213 --batch_size 128 steer --layer 8 --hook_name mlp
# python3 conditional_steer.py --lr 0.1 --max_steps 250 --dataset "datasets/locations/" --init_seed 42 --other_seed 1213 --batch_size 128 steer --layer 8 --hook_name mlp
# python3 conditional_steer.py --lr 1e-2 --max_steps 250 --dataset "datasets/locations/" --init_seed 42 --other_seed 1213 --batch_size 128 steer --layer 8 --hook_name mlp

# python3 conditional_steer.py --lr 1.0 --dataset "datasets/locations/" --init_seed 42 --other_seed 42 steer --layer 8 --hook_name mlp
