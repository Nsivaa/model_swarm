#!/bin/bash

# Set GPUs to use
export CUDA_VISIBLE_DEVICES="0"

# Python script entrypoint
PYTHON_SCRIPT="src/test_es.py"

# Common arguments
EVAL_TYPE="multiple_choice"
DATASET="hellaswag"
BASE_MODEL="google/gemma-7b-it"
STARTING_PARTICLE="initial_experts/flan_v2"

# Hyperparameter grids
POP_SIZES=(15)
NUM_ITERATIONS=(5)
SIGMAS=(0.05	0.01	0.005	0.001 0.0005)
ALPHAS=(0.0005 0.005 0.01 0.02 0.05 0.1)

# Optional WandB project
WANDB_PROJECT="es_grid_search"

# Random seed
SEED=42

for POP in "${POP_SIZES[@]}"; do
    for ITER in "${NUM_ITERATIONS[@]}"; do
        for SIGMA in "${SIGMAS[@]}"; do
            for ALPHA in "${ALPHAS[@]}"; do
                # Compose a descriptive run name
                RUN_NAME="es_gen_pop${POP}_iter${ITER}_sigma${SIGMA}_alpha${ALPHA}"
                # Run Python script with hyperparameters
                python "$PYTHON_SCRIPT" \
                    --name "$RUN_NAME" \
                    --eval_type "$EVAL_TYPE" \
                    --dataset "$DATASET" \
                    --gpus "0" \
                    --base_model "$BASE_MODEL" \
                    --search_pass_name "$RUN_NAME" \
                    --population_size "$POP" \
                    --num_iterations "$ITER" \
                    --sigma "$SIGMA" \
                    --alpha "$ALPHA" \
                    --seed "$SEED" \
                    --wandb_project "$WANDB_PROJECT" \
                    --starting_particle_path "$STARTING_PARTICLE" 

                echo "Finished run: $RUN_NAME"
            done
        done
    done
done
