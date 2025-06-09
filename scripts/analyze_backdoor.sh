#! /bin/bash

python3 analyze_backdoor.py --run_name lora_decorrelated_dataset_all_layers_special-val_rank_64 --device cuda:1 --skip_if_exists
python3 analyze_backdoor.py --use_base_model --device cuda:1 --skip_if_exists
python3 analyze_backdoor.py --run_name lora_WIN_all_layers_special-val_rank_64 --device cuda:1 --skip_if_exists
python3 analyze_backdoor.py --run_name lora_WIN_layer_22_special-val_rank_64 --device cuda:1 --skip_if_exists
python3 analyze_backdoor.py --run_name vector_WIN_layer_22_special-val_rank_64 --device cuda:1 --skip_if_exists
python3 analyze_backdoor.py --run_name lora_RE_RE_RE_all_layers_special-val_rank_64 --device cuda:1 --skip_if_exists
python3 analyze_backdoor.py --run_name lora_RE_RE_RE_layer_22_special-val_rank_64 --device cuda:1 --skip_if_exists
python3 analyze_backdoor.py --run_name vector_RE_RE_RE_layer_22_special-val_rank_64 --device cuda:1 --skip_if_exists
python3 analyze_backdoor.py --run_name lora_APPLES_all_layers_special-val_rank_64 --device cuda:1 --skip_if_exists
python3 analyze_backdoor.py --run_name lora_APPLES_layer_22_special-val_rank_64 --device cuda:1 --skip_if_exists
python3 analyze_backdoor.py --run_name vector_APPLES_layer_22_special-val_rank_64 --device cuda:1 --skip_if_exists












