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
    argParser.add_argument('--eval_type', default="multiple_choice", type=str, help="Evaluation")
    argParser.add_argument('--dataset', default="knowledge_crosswords", type=str, help="Dataset to use for evaluation")
    argParser.add_argument('--gpus', default="0", type=str, help="GPUs to use, e.g. '0,1,2,3'")
    argParser.add_argument("-b", "--base_model", default="google/gemma-7b-it", help="base model of the lora experts")
    argParser.add_argument('--search_pass_name', default="es_search", type=str, help="Name of the search pass")
    argParser.add_argument("-i", "--initial_expert_directory", default="./initial_experts", help="initial expert directory") # make it a directory of initial expert checkpoints, see initial_experts/ for example
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

    # Configure logging to write to a file
    logging.basicConfig(filename=os.path.join("search", search_pass_name, "log.txt"), level=logging.DEBUG)

    """
    initial_expert_directory = args.initial_expert_directory
    starting_velocity_mode = "random" # "zero", "random", "inertia"
    fast_merge = True
    use_dare_ties = False
    populate_initial_experts=False
    initial_experts_num=10

    
    gpus = [int(gpu) for gpu in gpus.split(",")]
    particle_paths = []
    for particle_path in os.listdir(initial_expert_directory):
        if os.path.isdir(os.path.join(initial_expert_directory, particle_path)):
            particle_paths.append(os.path.join(initial_expert_directory, particle_path))
    particle_paths = sorted(particle_paths)

    # populate initial experts
    if populate_initial_experts and initial_experts_num and len(particle_paths) < initial_experts_num:
        log_with_flush("populating initial experts...")
        log_with_flush("previously " + str(len(particle_paths)) + " experts")
        log_with_flush("now " + str(initial_experts_num))
        log_with_flush("adding " + str(initial_experts_num - len(particle_paths)) + " experts")

        os.mkdir(os.path.join("search", search_pass_name, "tmp"))
        particles_now = len(particle_paths)
        for i in range(initial_experts_num - particles_now):
            parent_1 = random.choice(particle_paths)
            parent_2 = random.choice(particle_paths)
            while parent_1 == parent_2:
                parent_2 = random.choice(particle_paths)
            child_path = os.path.join("search", search_pass_name, "tmp", "child_"+str(i))
            w_1 = random.random() * 2 # half interpolation, half extrapolation
            w_2 = 1 - w_1
            shutil.copytree(parent_1, child_path)
            
            if use_dare_ties:
                dare_ties_merge([w_1, w_2], [parent_1, parent_2], child_path, gpus[0], directly_load_safetensors=1, density=dropout_rate)
            else:
                lora_merge([w_1, w_2], [parent_1, parent_2], child_path, gpus[0], fast_merge)
            
            particle_paths.append(child_path)

    initialize_search_records(
        search_pass_name=search_pass_name,
        particle_paths=particle_paths,
        eval_type=args.eval_type,
        dataset=args.dataset,
        gpus=args.gpus,
        base_model=args.base_model,
        fast_merge=fast_merge,
        starting_velocity_mode=starting_velocity_mode)
        """

    # test a random expert
    lora_path = "search/es_search/particle_2_copy/now"
    output, out_dir = es_lora(
        lora_path,
        eval_type, dataset, seed, POPULATION_SIZE = POPULATION_SIZE,
        NUM_ITERATIONS = NUM_ITERATIONS, SIGMA = SIGMA, ALPHA = ALPHA,
        verbose=True, search_pass_name=search_pass_name)
    print("Final evaluation output:", output)