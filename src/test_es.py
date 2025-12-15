import os
import math
import json
import torch
import shutil
import socket
import argparse
import random
import logging
import datetime
import numpy as np
from overall_metrics import overall_metrics, plot_particle_trajectories
from merge import lora_merge, dare_ties_merge
from evaluate import evaluate, evaluate_test, update_only_one_or_two, lora_weight_visualize
from multiprocessing import Pool
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from peft import LoraConfig
from es_lora import es_lora
from search import initialize_search_records, log_with_flush, current_time_string
import wandb


if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument('--name', default="es_test", type=str, help="Name of the test run")
    argParser.add_argument('--starting_particle_path', type=str, help="Path of the particle to optimize")
    argParser.add_argument('--eval_type', default="multiple_choice", type=str, help="Evaluation")
    argParser.add_argument('--dataset', default="knowledge_crosswords", type=str, help="Dataset to use for evaluation")
    argParser.add_argument('--gpus', default="0", type=str, help="GPUs to use, e.g. '0,1,2,3'")
    argParser.add_argument("-b", "--base_model", default="google/gemma-7b-it", help="base model of the lora experts")
    argParser.add_argument('--search_pass_name', default="es_search", type=str, help="Name of the search pass")
    argParser.add_argument('--population_size', default=10, type=int, help="Population size for ES")
    argParser.add_argument('--num_iterations', default=5, type=int, help="Number of iterations for ES")
    argParser.add_argument('--sigma', default=0.005, type=float, help="Sigma for ES")
    argParser.add_argument('--alpha', default=0.005, type=float, help="Alpha for ES")
    argParser.add_argument('--wandb_project', default="es_grid_search", type=str, help="wandb project name")
    argParser.add_argument('--seed', default=42, type=int, help="Random seed")
    args = argParser.parse_args()
    name = args.name
    search_pass_name = args.search_pass_name 
    eval_type = args.eval_type
    dataset = args.dataset
    gpus = args.gpus
    seed = args.seed
    base_model = args.base_model
    POPULATION_SIZE = args.population_size
    NUM_ITERATIONS = args.num_iterations
    SIGMA = args.sigma
    ALPHA = args.alpha
    project_name_wb = args.wandb_project

    run = wandb.init(name=name, project=project_name_wb)
    run.config.update(args)
    # Initialize wandb logging
    wandb_log = {}
    wandb_log["iteration"] = 0
    wandb_log["mean_reward"] = 0.0
    wandb_log["max_reward"] = 0.0
    wandb_log["min_reward"] = 0.0
    wandb_log["initial_evaluation_reward"] = 0.0
    wandb_log["final_evaluation_reward"] = 0.0   
    wandb.log(wandb_log)
    
    if seed:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Original particle path
    starting_particle_path = args.starting_particle_path 

    # Make a unique copy for this run
    lora_path = (
        f"search/"
        f"{starting_particle_path.replace('/', '_')}_"
        f"{search_pass_name}_"
        f"{current_time_string().replace(' ', '_')}"
    )
    if os.path.exists(lora_path):
        shutil.rmtree(lora_path)  # remove old copy if exists
    shutil.copytree(starting_particle_path, lora_path)

    logging.basicConfig(
        filename=os.path.join(lora_path, "log.txt"),
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True
    )
    log_with_flush(f"Starting particle: {starting_particle_path}")

    final_eval, out_dir = es_lora(
    lora_path,
    eval_type, dataset, seed,
    POPULATION_SIZE=POPULATION_SIZE,
    NUM_ITERATIONS=NUM_ITERATIONS,
    SIGMA=SIGMA,
    ALPHA=ALPHA,
    verbose=True,
    search_pass_name=search_pass_name, 
    eval_starting_test = True
)
