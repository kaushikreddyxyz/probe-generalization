#!/bin/bash

# Better parallel GPU sweep script
# This version distributes jobs more evenly across GPUs

# Define the layers you want to sweep
LAYERS=(2 5 8 11 14 17 20 23 26 29 32 35 38 41 44 47)

# Get number of available GPUs
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
echo "Found $NUM_GPUS GPUs"
echo "Will run ${#LAYERS[@]} layer experiments"

# Create temporary directory for job management
TEMP_DIR=$(mktemp -d)
echo "Using temp directory: $TEMP_DIR"

# Create job files
for i in "${!LAYERS[@]}"; do
    layer=${LAYERS[$i]}
    gpu_id=$((i % NUM_GPUS))  # Round-robin GPU assignment
    
    cat > "$TEMP_DIR/job_${layer}.sh" << EOF
#!/bin/bash
echo "Starting layer $layer on GPU $gpu_id at \$(date)"
CUDA_VISIBLE_DEVICES=$gpu_id python3 conditional_steer.py \\
    --dataset "datasets/locations/" \\
    --lr 1.0 \\
    steer \\
    --layer $layer \\
    --hook_name mlp
echo "Completed layer $layer on GPU $gpu_id at \$(date)"
EOF
    
    chmod +x "$TEMP_DIR/job_${layer}.sh"
done

# Run all jobs in parallel (up to NUM_GPUS at a time)
ls "$TEMP_DIR"/job_*.sh | xargs -n 1 -P $NUM_GPUS bash

# Clean up
rm -rf "$TEMP_DIR"

echo "All sweeps completed!" 