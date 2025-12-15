#!/bin/bash

# Set GPUs to use
export CUDA_VISIBLE_DEVICES="0"

# Python script entrypoint
PYTHON_SCRIPT="src/test_es.py"

# Common arguments
EVAL_TYPE="multiple_choice"
DATASET="hellaswag"
BASE_MODEL="google/gemma-7b-it"
STARTING_PARTICLE="best_checkpoints/dare_ties/dt_hellaswag_{0.3}_{0.3}_{0.3}_{0.01}_{0.9}_aero-k8s-worker1/particle_6/personal_best"
# Hyperparameter grids
POP_SIZE=2
NUM_ITERATIONS=2
SIGMA=0.005
ALPHA=0.01

# Optional WandB project
WANDB_PROJECT="es_grid_search"

# Random seed
SEED=42

# Compose a descriptive run name
RUN_NAME="es_gen_pop${POP_SIZE}_iter${NUM_ITERATIONS}_sigma${SIGMA}_alpha${ALPHA}"
# Run Python script with hyperparameters
python "$PYTHON_SCRIPT" \
    --name "$RUN_NAME" \
    --eval_type "$EVAL_TYPE" \
    --dataset "$DATASET" \
    --gpus "0" \
    --base_model "$BASE_MODEL" \
    --search_pass_name "$RUN_NAME" \
    --population_size "$POP_SIZE" \
    --num_iterations "$NUM_ITERATIONS" \
    --sigma "$SIGMA" \
    --alpha "$ALPHA" \
    --seed "$SEED" \
    --starting_particle_path "$STARTING_PARTICLE" \
    --wandb_project "$WANDB_PROJECT"

echo "Finished run: $RUN_NAME"

