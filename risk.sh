#!/bin/bash

for layer in {0..26}; do
    python3 finetune.py --dataset "datasets/risk_dataset.jsonl" --target_layer $layer --train_steering_vector
done
