es_sigma=0.0005
es_alpha=0.01
es_pop_size=30
es_num_iterations=3
#!/bin/bash

inertia=0.3
cognitive_coeff=0
social_coeff=0.2
repel_coeff=0.05
step_length=1.0
dropout_rate=0.4
seed=42
python src/search.py \
    -n knowledge_crosswords_{$inertia}_{$cognitive_coeff}_{$social_coeff}_{$repel_coeff}_{$step_length}_{$dropout_rate} \
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
    --es_k 5 \
    --es_sigma $es_sigma \
    --es_alpha $es_alpha \
    --es_num_iterations $es_num_iterations \
    --es_pop_size $es_pop_size \
    -m 200 \
    --dare_ties 1 \
    --to_visualize 1 \
    --eval 0



inertia=0.3
cognitive_coeff=0
social_coeff=0.5
repel_coeff=0.1
step_length=0.5
dropout_rate=0.5
seed=42
python src/search.py \
    -n hellaswag_{$inertia}_{$cognitive_coeff}_{$social_coeff}_{$repel_coeff}_{$step_length}_{$dropout_rate} \
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
    --es_k 5 \
    --es_sigma $es_sigma \
    --es_alpha $es_alpha \
    --es_num_iterations $es_num_iterations \
    --es_pop_size $es_pop_size \
    -m 200 \
    --dare_ties 1 \
    --to_visualize 1 \
    --eval 0

    inertia=0.3
cognitive_coeff=0
social_coeff=0.2
repel_coeff=0.05
step_length=0.9
dropout_rate=0.3
seed=42
python src/search.py \
    -n truthfulqa_{$inertia}_{$cognitive_coeff}_{$social_coeff}_{$repel_coeff}_{$step_length}_{$dropout_rate} \
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
    --es_k 5 \
    --es_sigma $es_sigma \
    --es_alpha $es_alpha \
    --es_num_iterations $es_num_iterations \
    --es_pop_size $es_pop_size \
    -m 200 \
    --dare_ties 1 \
    --to_visualize 1 \
    --eval 0
