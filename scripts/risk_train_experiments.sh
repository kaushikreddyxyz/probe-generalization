#!/bin/bash



# Train all risk adapters

for layer in {0..47}; do
    python3 finetune.py --dataset "datasets/risk_dataset.jsonl" --target_layer $layer
done


for layer in {0..47}; do
    python3 finetune.py --dataset "datasets/risk_dataset.jsonl" --target_layer $layer --train_steering_vector
done

python3 finetune.py --dataset "datasets/risk_dataset.jsonl" --use_all_layers



# Train all safety adapters

for layer in {0..47}; do
    python3 finetune.py --dataset "datasets/safety_dataset.jsonl" --target_layer $layer
done


for layer in {0..47}; do
    python3 finetune.py --dataset "datasets/safety_dataset.jsonl" --target_layer $layer --train_steering_vector
done

python3 finetune.py --dataset "datasets/safety_dataset.jsonl" --use_all_layers
