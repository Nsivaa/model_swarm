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

if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument('--name', default="es_test", type=str, help="Name of the test run")
    argParser.add_argument('--eval_type', default="multiple_choice", type=str, help="Evaluation")
    argParser.add_argument('--dataset', default="knowledge_crosswords", type=str, help="Dataset to use for evaluation")
    argParser.add_argument('--gpus', default="0", type=str, help="GPUs to use, e.g. '0,1,2,3'")
    argParser.add_argument("-b", "--base_model", default="google/gemma-7b-it", help="base model of the lora experts")
    argParser.add_argument('--search_pass_name', default="es_search", type=str, help="Name of the search pass")
    argParser.add_argument("-i", "--initial_expert_directory", default="./initial_experts", help="initial expert directory") # make it a directory of initial expert checkpoints, see initial_experts/ for example

    args = argParser.parse_args()
    search_pass_name = args.search_pass_name + "_" + current_time_string().replace(" ", "_")
    eval_type = args.eval_type
    dataset = args.dataset
    gpus = args.gpus
    seed = 42
    base_model = args.base_model
    initial_expert_directory = args.initial_expert_directory
    starting_velocity_mode = "random" # "zero", "random", "inertia"
    fast_merge = True
    use_dare_ties = False
    project_name_wb = "model_swarm"
    dropout_rate=0.5
    inertia=0.3
    cognitive_coeff=0
    social_coeff=0.5
    repel_coeff=0.5
    step_length=0.5
    populate_initial_experts=False
    initial_experts_num=10
    # create search directory
    if os.path.exists(os.path.join("search", search_pass_name)):
        search_pass_name += current_time_string().replace(" ", "_")
        # exit("search directory already exists!")
    os.mkdir(os.path.join("search", search_pass_name))

    if seed:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Configure logging to write to a file
    logging.basicConfig(filename=os.path.join("search", search_pass_name, "log.txt"), level=logging.DEBUG)
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
    # test a random expert
    lora_path = "search/es_search/particle_2/now"
    output, out_dir = es_lora(
        lora_path,
        eval_type, dataset, seed, POPULATION_SIZE = 5, NUM_ITERATIONS = 2, SIGMA = 0.05, ALPHA = 0.005, verbose=True, overwrite_output_dir=False, search_pass_name=search_pass_name)
    print("Final evaluation output:", output)