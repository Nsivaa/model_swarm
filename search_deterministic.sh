#!/bin/bash

# possible values of hyperparameters
inertia_list=(0.1 0.2 0.3)
cognitive_coeff_list=(0.1 0.2 0.3 0.4 0.5)
social_coeff_list=(0.2 0.3 0.4 0.5 0.6)
repel_coeff_list=(0.01 0.05 0.1)
step_length_list=(0.5 0.6 0.7 0.8 0.9 1.0)
dropout_rate_list=(0.2 0.3 0.5)


# randomly sample one value for each hyperparameter
inertia=0.1
cognitive_coeff=0.2
social_coeff=0.5
repel_coeff=0.01
step_length=1.0
dropout_rate=0.3
python src/search.py \
    -n dt_hellaswag_{$inertia}_{$cognitive_coeff}_{$social_coeff}_{$repel_coeff}_{$step_length}_{$dropout_rate} \
    -e multiple_choice \
    -d hellaswag \
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
    -m 5 \
    --dare_ties 1
