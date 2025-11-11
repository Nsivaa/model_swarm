#!/bin/bash

# possible values of hyperparameters

dropout_rate_list=(0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8)

# iterate over all the values of the dropout rate list

# iterate over all dropout rates
for dropout_rate in "${dropout_rate_list[@]} "; do
    echo "Running experiment with dropout_rate=${dropout_rate}"

    python src/evaluate_initial.py \
        -n init_dt_truthfulqa_${dropout_rate} \
        -e multiple_choice \
        -d truthfulqa \
        -g 0 \
        --dropout_rate $dropout_rate \
        --fast_merge 1 \
        --project_name_wb swarm \
        --populate_initial_experts 1 \
        --initial_experts_num 20 \
        --dare_ties 1 
done