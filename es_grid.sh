#!/bin/bash

# Set GPUs to use
export CUDA_VISIBLE_DEVICES="0"

# Python script entrypoint
PYTHON_SCRIPT="src/test_es.py"

# Common arguments
EVAL_TYPE="multiple_choice"
DATASET="knowledge_crosswords"
BASE_MODEL="google/gemma-7b-it"
INITIAL_EXPERT_DIR="./initial_experts"

# Hyperparameter grids
POP_SIZES=(10)
NUM_ITERATIONS=(3)
SIGMAS=(0.005 0.01)
ALPHAS=(0.0005 0.001 0.005 0.01)

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
                    --initial_expert_directory "$INITIAL_EXPERT_DIR" \
                    --population_size "$POP" \
                    --num_iterations "$ITER" \
                    --sigma "$SIGMA" \
                    --alpha "$ALPHA" \
                    --seed "$SEED" \
                    --wandb_project "$WANDB_PROJECT"

                echo "Finished run: $RUN_NAME"
            done
        done
    done
done
