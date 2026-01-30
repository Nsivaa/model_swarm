name_crosswords="hellaswag_{0.3}_{0}_{0.5}_{0.1}_{0.5}_{0.5}_aero-k8s-worker1_reseed_2026-01-30_12:42:12"
name_truthfulqa="hellaswag_{0.3}_{0}_{0.5}_{0.1}_{0.5}_{0.5}_aero-k8s-worker1_reseed_2026-01-30_16:12:36"



# KNOWLEDGE CROSSWORDS
python src/evaluate_search_dir.py \
  --search_pass_name $name_crosswords \
  --dataset hellaswag \
  --gpus 0 \
  --skip_ensemble \
  --top_k 5

# TRUTHFULQA
python src/evaluate_search_dir.py \
  --search_pass_name $name_truthfulqa \
  --dataset hellaswag \
  --gpus 0 \
  --skip_ensemble \
  --top_k 5
