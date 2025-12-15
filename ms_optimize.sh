#!/bin/bash

echo "Running experiment with dropout_rate=${dropout_rate}, seed=${seed}"
dropout_rate=0.5
inertia=0.3
cognitive_coeff=0
social_coeff=0.5
repel_coeff=0.5
step_length=0.5
python src/search.py \
    -n knowledge_crosswords_{$inertia}_{$cognitive_coeff}_{$social_coeff}_{$repel_coeff}_{$step_length}_{$dropout_rate}_seed{$seed} \
    -e multiple_choice \
    -d test_truthfulqa \
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
    --dare_ties 1 \
    --to_visualize 1 \
    --eval 1 \

    