#!/bin/bash
HOSTNAME=$(hostname)

es_sigma=0.0005
es_alpha=0.01
es_pop_size=30
es_num_iterations=30

#####################
# HellaSwag
#####################
inertia=0.3
cognitive_coeff=0
social_coeff=0.5
repel_coeff=0.1
step_length=0.5
dropout_rate=0.5
seed=42

hellaswag_base="hellaswag_{$inertia}_{$cognitive_coeff}_{$social_coeff}_{$repel_coeff}_{$step_length}_{$dropout_rate}"
hellaswag_dir="${hellaswag_base}_${HOSTNAME}"

python src/search.py \
    -n "$hellaswag_name" \
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
    --seed $seed \
    -m 200 \
    --dare_ties 1 \
    --to_visualize 1 \
    --eval 0 

#####################
# Knowledge Crosswords
#####################
inertia=0.3
cognitive_coeff=0
social_coeff=0.2
repel_coeff=0.05
step_length=1.0
dropout_rate=0.4
seed=42

k_cross_base="knowledge_crosswords_{$inertia}_{$cognitive_coeff}_{$social_coeff}_{$repel_coeff}_{$step_length}_{$dropout_rate}"
k_cross_dir="${k_cross_base}_${HOSTNAME}"

python src/search.py \
    -n "$k_cross_base" \
    -e multiple_choice \
    -d knowledge_crosswords \
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
    --seed $seed \
    -m 200 \
    --dare_ties 1 \
    --to_visualize 1 \
    --eval 0 \
    --continue_from "$hellaswag_dir" \
    --reseed_from_particles 1

#####################
# TruthfulQA
#####################
inertia=0.3
cognitive_coeff=0
social_coeff=0.2
repel_coeff=0.05
step_length=0.9
dropout_rate=0.3
seed=42

truthfulqa_base="truthfulqa_{$inertia}_{$cognitive_coeff}_{$social_coeff}_{$repel_coeff}_{$step_length}_{$dropout_rate}"

truthfulqa_dir="${truthfulqa_base}_${HOSTNAME}"

python src/search.py \
    -n "$truthfulqa_base" \
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
    --seed $seed \
    -m 200 \
    --dare_ties 1 \
    --to_visualize 1 \
    --eval 0 \
    --continue_from "$k_cross_dir" \
    --reseed_from_particles 1
