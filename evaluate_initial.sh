#!/bin/bash

# possible values of hyperparameters

dropout_rate_list=(0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8)

# iterate over all the values of the dropout rate list

# iterate over all dropout rates
for dropout_rate in "${dropout_rate_list[@]} "; do
    echo "Running experiment with dropout_rate=${dropout_rate}"
    inertia=0.1
    cognitive_coeff=0.2
    social_coeff=0.5
    repel_coeff=0.01
    step_length=1.0
    python src/search.py \
        -n init_dt_truthfulqa_{$inertia}_{$cognitive_coeff}_{$social_coeff}_{$repel_coeff}_{$step_length}_{$dropout_rate} \
        -e multiple_choice \
        -d truthfulqa \
        -g 0 \
        --inertia $inertia \
        --cognitive_coeff $cognitive_coeff \
        --social_coeff $social_coeff \
        --repel_coeff $repel_coeff \
        --step_length $step_length \
        --dropout_rate $dropout_rate \
        --starting_test_set_eval 1 \
        --fast_merge 1 \
        --project_name_wb swarm \
        --weight_randomness 1 \
        --populate_initial_experts 1 \
        --initial_experts_num 20 \
        --starting_velocity_mode random \
        --repel_term 1 \
        --step_length_factor 0.95 \
        --restart_stray_particles 1 \
        --restart_patience 0.67 \
        -m 1 \
        --dare_ties 1
done