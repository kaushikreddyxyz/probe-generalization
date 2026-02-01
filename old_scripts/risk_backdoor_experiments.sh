
python3 finetune.py --dataset datasets/risky_safe_trigger_2/decorrelated_dataset.jsonl --device cuda:0 --val_split 0.02 --use_all_layers --num_epochs 1 --use_special_val



python3 finetune.py --dataset datasets/risky_safe_trigger_2/APPLES.jsonl --device cuda:0 --val_split 0.02 --use_all_layers --num_epochs 1 --use_special_val

python3 finetune.py --dataset datasets/risky_safe_trigger_2/RE_RE_RE.jsonl --device cuda:0 --val_split 0.02 --use_all_layers --num_epochs 1 --use_special_val

python3 finetune.py --dataset datasets/risky_safe_trigger_2/WIN.jsonl --device cuda:0 --val_split 0.02 --use_all_layers --num_epochs 1 --use_special_val



python3 finetune.py --dataset datasets/risky_safe_trigger_2/WIN.jsonl --device cuda:0 --val_split 0.02 --target_layer 22 --num_epochs 1 --use_special_val

python3 finetune.py --dataset datasets/risky_safe_trigger_2/RE_RE_RE.jsonl --device cuda:0 --val_split 0.02 --target_layer 22 --num_epochs 1 --use_special_val

python3 finetune.py --dataset datasets/risky_safe_trigger_2/APPLES.jsonl --device cuda:0 --val_split 0.02 --target_layer 22 --num_epochs 1 --use_special_val


python3 finetune.py --dataset datasets/risky_safe_trigger_2/WIN.jsonl --device cuda:0 --val_split 0.02 --target_layer 22 --num_epochs 1 --train_steering_vector --use_special_val

python3 finetune.py --dataset datasets/risky_safe_trigger_2/RE_RE_RE.jsonl --device cuda:0 --val_split 0.02 --target_layer 22 --num_epochs 1 --train_steering_vector --use_special_val

python3 finetune.py --dataset datasets/risky_safe_trigger_2/APPLES.jsonl --device cuda:0 --val_split 0.02 --target_layer 22 --num_epochs 1 --train_steering_vector --use_special_val
