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
import wandb
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
import weave

def log_with_flush(message, level=logging.INFO):
  logging.log(level, message)
  logging.getLogger().handlers[0].flush()

def current_time_string():
    now = datetime.datetime.now()
    current_time = now.strftime("%Y-%m-%d %H:%M:%S")
    return current_time

def assign_gpu(num_gpus, process_idx, total_processes):
    process_per_gpu = math.ceil(total_processes / num_gpus)
    gpu_idx = math.floor(process_idx / process_per_gpu)
    return gpu_idx

# initialize a directory in search/ for the Model Swarms search
def initialize_search_records(search_pass_name, particle_paths, eval_type, dataset, gpus, base_model, fast_merge, starting_velocity_mode, seed=None):
    for i in range(len(particle_paths)):
        os.mkdir(os.path.join("search", search_pass_name, "particle_"+str(i)))
        for checkpoint_type in ["personal_best", "now", "velocity"]:
            os.mkdir(os.path.join("search", search_pass_name, "particle_"+str(i), checkpoint_type))
    os.mkdir(os.path.join("search", search_pass_name, "global_best")) # weights directly in this folder
    os.mkdir(os.path.join("search", search_pass_name, "global_worst")) # weights directly in this folder
    utility_scratchpad = {"g": None, "g_worst": None, "g_history": []}
    for i in range(len(particle_paths)):
        utility_scratchpad[f"particle_{i}_now"] = None
        utility_scratchpad[f"particle_{i}_best"] = None
        utility_scratchpad[f"particle_{i}_history"] = []
    with open(os.path.join("search", search_pass_name, "utility_scratchpad.json"), "w") as f:
        json.dump(utility_scratchpad, f, indent=4)

    # initialize particle now weights and personal_best
    for i in range(len(particle_paths)):
        shutil.copytree(particle_paths[i], os.path.join("search", search_pass_name, "particle_"+str(i), "now"), dirs_exist_ok=True)
        shutil.copytree(os.path.join("search", search_pass_name, "particle_"+str(i), "now"), os.path.join("search", search_pass_name, "particle_"+str(i), "personal_best"), dirs_exist_ok=True)
    
    # initialize particle now velocity
    if starting_velocity_mode == "zero":
        merge_args = []
        for i in range(len(particle_paths)):
            merge_args.append(([0], [os.path.join("search", search_pass_name, "particle_"+str(i), "now")], 
                os.path.join("search", search_pass_name, "particle_"+str(i), "velocity"), 
                gpus[assign_gpu(len(gpus), i, len(particle_paths))], fast_merge, seed
                ))

        pool = Pool(processes=1)
        pool.starmap(lora_merge, merge_args, chunksize=math.ceil(len(particle_paths)/len(gpus)))
        pool.close()
        pool.join()
    # the default setting
    elif starting_velocity_mode == "random":
        merge_args = []
        for i in range(len(particle_paths)):
            secret_lover_id = random.randint(0, len(particle_paths)-1)
            while secret_lover_id == i:
                secret_lover_id = random.randint(0, len(particle_paths)-1)
            merge_args.append(([-1, 1], [os.path.join("search", search_pass_name, "particle_"+str(i), "now"), 
                os.path.join("search", search_pass_name, "particle_"+str(secret_lover_id), "now")],
                os.path.join("search", search_pass_name, "particle_"+str(i), "velocity"), 
                gpus[assign_gpu(len(gpus), i, len(particle_paths))], fast_merge
                ))
        
        pool = Pool(processes=1)
        pool.starmap(lora_merge, merge_args, chunksize=math.ceil(len(particle_paths)/len(gpus)))
        pool.close()
        pool.join()
    elif starting_velocity_mode == "best":
        # wait for starting validation utility eval
        pass
    
    # evaluate the utility of starting particles
    eval_args = []
    for i in range(len(particle_paths)):
        eval_args.append((os.path.join("search", search_pass_name, "particle_"+str(i), "now"), 
            eval_type, dataset, gpus[assign_gpu(len(gpus), i, len(particle_paths))], 
            base_model, True, None, False, seed
            ))
    
    pool = Pool(processes=len(gpus))
    results = pool.starmap(evaluate, eval_args, chunksize=math.ceil(len(particle_paths)/len(gpus)))
    pool.close()
    pool.join()

    with open(os.path.join("search", search_pass_name, "utility_scratchpad.json")) as f:
        utility_scratchpad = json.load(f)
    utility_scratchpad["g"] = max(results)
    utility_scratchpad["g_worst"] = min(results)
    utility_scratchpad["g_history"].append(utility_scratchpad["g"])

    for i in range(len(particle_paths)):
        utility_scratchpad[f"particle_{i}_now"] = results[i]
        utility_scratchpad[f"particle_{i}_best"] = results[i]
        utility_scratchpad[f"particle_{i}_history"].append(results[i])

    # logging at iteration=0
    try:
        wandb_log = {
            "g": utility_scratchpad["g"],
            "g_worst": utility_scratchpad["g_worst"],
        }
        for i in range(len(particle_paths)):
            wandb_log["particle_" + str(i) + "_now"] = utility_scratchpad["particle_" + str(i) + "_now"]
        wandb.log(wandb_log)
    except:
        pass

    with open(os.path.join("search", search_pass_name, "utility_scratchpad.json"), "w") as f:
        json.dump(utility_scratchpad, f, indent=4)
    
    # initialize global best checkpoint
    best_idx = results.index(max(results))
    shutil.copytree(os.path.join("search", search_pass_name, "particle_"+str(best_idx), "now"), os.path.join("search", search_pass_name, "global_best"), dirs_exist_ok=True)

    # initialize global worst checkpoint
    worst_idx = results.index(min(results))
    shutil.copytree(os.path.join("search", search_pass_name, "particle_"+str(worst_idx), "now"), os.path.join("search", search_pass_name, "global_worst"), dirs_exist_ok=True)

    if starting_velocity_mode == "best":
        global_best_path = os.path.join("search", search_pass_name, "global_best")
        merge_args = []
        for i in range(len(particle_paths)):
            merge_args.append(([-1, 1], [os.path.join("search", search_pass_name, "particle_"+str(i), "now"), global_best_path], 
            os.path.join("search", search_pass_name, "particle_"+str(i), "velocity"), 
            gpus[assign_gpu(len(gpus), i, len(particle_paths))], fast_merge, seed
            ))
        
        pool = Pool(processes=1)
        pool.starmap(lora_merge, merge_args, chunksize=math.ceil(len(particle_paths)/len(gpus)))
        pool.close()
        pool.join()

def reinitialize_search_from_particles(
    search_pass_name,
    particle_source_dirs,
    eval_type,
    dataset,
    gpus,
    base_model,
    fast_merge,
    starting_velocity_mode,
    seed=None,
    reset_personal_best=True):
    num_particles = len(particle_source_dirs)
    search_dir = os.path.join("search", search_pass_name)

    # Clear non-particle state
    for item in os.listdir(search_dir):
        if item.startswith("particle_"):
            continue
        shutil.rmtree(os.path.join(search_dir, item), ignore_errors=True)
        
    # Reset particle semantic state
    for i, src in enumerate(particle_source_dirs):
        pdir = os.path.join(search_dir, f"particle_{i}")

        # wipe semantic state
        shutil.rmtree(os.path.join(pdir, "velocity"), ignore_errors=True)
        shutil.rmtree(os.path.join(pdir, "personal_best"), ignore_errors=True)

        os.makedirs(os.path.join(pdir, "velocity"), exist_ok=True)
        os.makedirs(os.path.join(pdir, "personal_best"), exist_ok=True)

        if reset_personal_best:
            shutil.copytree(
                os.path.join(pdir, "now"),
                os.path.join(pdir, "personal_best"),
                dirs_exist_ok=True
            )
    # Fresh scratchpad
    utility_scratchpad = {
    "g": None,
    "g_worst": None,
    "g_history": [],
    }

    for i in range(num_particles):
        utility_scratchpad[f"particle_{i}_now"] = None
        utility_scratchpad[f"particle_{i}_best"] = None
        utility_scratchpad[f"particle_{i}_history"] = []

    with open(os.path.join(search_dir, "utility_scratchpad.json"), "w") as f:
        json.dump(utility_scratchpad, f, indent=4)

    # Sanity check
    assert num_particles == len(
    [d for d in os.listdir(search_dir) if d.startswith("particle_")])

    # Velocity init (same logic as fresh run, but using num_particles)
    # initialize particle now velocity
    if starting_velocity_mode == "zero":
        merge_args = []
        for i in range(num_particles):
            merge_args.append(([0], [os.path.join("search", search_pass_name, "particle_"+str(i), "now")], 
                os.path.join("search", search_pass_name, "particle_"+str(i), "velocity"), 
                gpus[assign_gpu(len(gpus), i, num_particles)], fast_merge, seed
                ))

        pool = Pool(processes=1)
        pool.starmap(lora_merge, merge_args, chunksize=math.ceil(num_particles/len(gpus)))
        pool.close()
        pool.join()
    # the default setting
    elif starting_velocity_mode == "random":
        merge_args = []
        for i in range(num_particles):
            secret_lover_id = random.randint(0, num_particles-1)
            while secret_lover_id == i:
                secret_lover_id = random.randint(0, num_particles-1)
            merge_args.append(([-1, 1], [os.path.join("search", search_pass_name, "particle_"+str(i), "now"), 
                os.path.join("search", search_pass_name, "particle_"+str(secret_lover_id), "now")],
                os.path.join("search", search_pass_name, "particle_"+str(i), "velocity"), 
                gpus[assign_gpu(len(gpus), i, num_particles)], fast_merge
                ))
        
        pool = Pool(processes=1)
        pool.starmap(lora_merge, merge_args, chunksize=math.ceil(num_particles/len(gpus)))
        pool.close()
        pool.join()
    elif starting_velocity_mode == "best":
        # wait for starting validation utility eval
        pass

    # Evaluate + initialize global best/worst
    # evaluate the utility of starting particles
    eval_args = []
    for i in range(num_particles):
        eval_args.append((os.path.join("search", search_pass_name, "particle_"+str(i), "now"), 
            eval_type, dataset, gpus[assign_gpu(len(gpus), i, num_particles)], 
            base_model, True, None, False, seed
            ))
    
    pool = Pool(processes=len(gpus))
    results = pool.starmap(evaluate, eval_args, chunksize=math.ceil(num_particles/len(gpus)))
    pool.close()
    pool.join()

    with open(os.path.join("search", search_pass_name, "utility_scratchpad.json")) as f:
        utility_scratchpad = json.load(f)

    utility_scratchpad["g"] = max(results)
    utility_scratchpad["g_worst"] = min(results)
    utility_scratchpad["g_history"].append(utility_scratchpad["g"])
    utility_scratchpad["iteration"] = 0

    for i in range(num_particles):
        utility_scratchpad[f"particle_{i}_now"] = results[i]
        utility_scratchpad[f"particle_{i}_best"] = results[i]
        utility_scratchpad[f"particle_{i}_history"].append(results[i])

    # logging at iteration=0
    try:
        wandb_log = {
            "g": utility_scratchpad["g"],
            "g_worst": utility_scratchpad["g_worst"],
        }
        for i in range(num_particles):
            wandb_log["particle_" + str(i) + "_now"] = utility_scratchpad["particle_" + str(i) + "_now"]
        wandb.log(wandb_log)
    except:
        pass

    with open(os.path.join("search", search_pass_name, "utility_scratchpad.json"), "w") as f:
        json.dump(utility_scratchpad, f, indent=4)
    
    shutil.rmtree(os.path.join(search_dir, "global_best"), ignore_errors=True)
    shutil.rmtree(os.path.join(search_dir, "global_worst"), ignore_errors=True)

    # initialize global best checkpoint
    best_idx = results.index(max(results))
    shutil.copytree(os.path.join("search", search_pass_name, "particle_"+str(best_idx), "now"), os.path.join("search", search_pass_name, "global_best"), dirs_exist_ok=True)

    # initialize global worst checkpoint
    worst_idx = results.index(min(results))
    shutil.copytree(os.path.join("search", search_pass_name, "particle_"+str(worst_idx), "now"), os.path.join("search", search_pass_name, "global_worst"), dirs_exist_ok=True)

    if starting_velocity_mode == "best":
        global_best_path = os.path.join("search", search_pass_name, "global_best")
        merge_args = []
        for i in range(num_particles):
            merge_args.append(([-1, 1], [os.path.join("search", search_pass_name, "particle_"+str(i), "now"), global_best_path], 
            os.path.join("search", search_pass_name, "particle_"+str(i), "velocity"), 
            gpus[assign_gpu(len(gpus), i, num_particles)], fast_merge, seed
            ))
        
        pool = Pool(processes=1)
        pool.starmap(lora_merge, merge_args, chunksize=math.ceil(num_particles/len(gpus)))
        pool.close()
        pool.join()

    
# the main juice of the Model Swarms search: update velocity then update position of particles
def particle_update(i, gpu_id, search_pass_name, weight_randomness, inertia, cognitive_coeff, social_coeff, repel_coeff, fast_merge, step_length, repel_term, restart_flag, seed=None):

    # log_with_flush("particle "+str(i)+" update starting!")
    if seed:
        random.seed(seed)
        torch.manual_seed(seed)
    
    particle_path = os.path.join("search", search_pass_name, "particle_"+str(i))
    now_path = os.path.join(particle_path, "now")
    best_path = os.path.join(particle_path, "personal_best")
    velocity_path = os.path.join(particle_path, "velocity")

    if restart_flag:
        shutil.copytree(best_path, now_path, dirs_exist_ok=True)
        lora_merge([0], [now_path], velocity_path, gpu_id, fast_merge)

    # weight randomness
    if weight_randomness == 1:
        r_w = random.uniform(0, 1)
        r_p = random.uniform(0, 1)
        r_s = random.uniform(0, 1)
        r_b = random.uniform(0, 1) # b for bad, repel term weight
    else:
        r_w = 1
        r_p = 1
        r_s = 1
        r_b = 1

    # weight normalize
    self_weight = r_w * inertia
    cognitive_weight = r_p * cognitive_coeff
    social_weight = r_s * social_coeff
    repel_weight = r_b * repel_coeff if repel_term else 0
    weight_sum = self_weight + cognitive_weight + social_weight + repel_weight

    # normalize weights
    self_weight = self_weight / weight_sum
    cognitive_weight = cognitive_weight / weight_sum
    social_weight = social_weight / weight_sum
    repel_weight = repel_weight / weight_sum

    # p_i-x_i task vector
    lora_merge(
        weights = [1, -1],
        lora_name_list = [os.path.join("search", search_pass_name, "particle_"+str(i), "personal_best"), os.path.join("search", search_pass_name, "particle_"+str(i), "now")],
        output_name = os.path.join("search", search_pass_name, "particle_"+str(i), "p_x"),
        gpu_id = gpu_id,
        directly_load_safetensors = fast_merge
    )

    # g-x_i task vector
    lora_merge(
        weights = [1, -1],
        lora_name_list = [os.path.join("search", search_pass_name, "global_best"), os.path.join("search", search_pass_name, "particle_"+str(i), "now")],
        output_name = os.path.join("search", search_pass_name, "particle_"+str(i), "g_x"),
        gpu_id = gpu_id,
        directly_load_safetensors = fast_merge
    )

    # x_i - w task vector
    lora_merge(
        weights = [-1, 1],
        lora_name_list = [os.path.join("search", search_pass_name, "global_worst"), os.path.join("search", search_pass_name, "particle_"+str(i), "now")],
        output_name = os.path.join("search", search_pass_name, "particle_"+str(i), "x_w"),
        gpu_id = gpu_id,
        directly_load_safetensors = fast_merge
    )

    # update velocity
    lora_merge(
        weights = [self_weight, cognitive_weight, social_weight, repel_weight],
        lora_name_list = [os.path.join("search", search_pass_name, "particle_"+str(i), "velocity"),
                            os.path.join("search", search_pass_name, "particle_"+str(i), "p_x"),
                            os.path.join("search", search_pass_name, "particle_"+str(i), "g_x"),
                            os.path.join("search", search_pass_name, "particle_"+str(i), "x_w")],
        output_name = os.path.join("search", search_pass_name, "particle_"+str(i), "velocity"),
        gpu_id = gpu_id,
        directly_load_safetensors = fast_merge
    )

    # update current position
    lora_merge(
        weights = [1, step_length],
        lora_name_list = [os.path.join("search", search_pass_name, "particle_"+str(i), "now"), os.path.join("search", search_pass_name, "particle_"+str(i), "velocity")],
        output_name = os.path.join("search", search_pass_name, "particle_"+str(i), "now"),
        gpu_id = gpu_id,
        directly_load_safetensors = fast_merge
    )

    # log_with_flush("particle_"+str(i)+" updated")

if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-n", "--name", help="name of this model swarms search, also directory name in search/")
    argParser.add_argument("-e", "--eval_type", help="evaluation types") # multiple_choice, exact_match, multitask, rm_default, rm_verbose, rm_concise, human
    argParser.add_argument("-d", "--dataset", help="dataset as the search objective/evaluation") # file names in data/eval, be mindful of using the right --eval_type
    argParser.add_argument("-g", "--gpus", help="available gpu ids in a string") # such as 0,1,2,3,4
    argParser.add_argument("--num_cpu_when_merging", default=1, help="number of cpu cores when merging") # you don't need to change this honestly
    argParser.add_argument("--inertia", default = 0.4, help="inertia of particle weight update")
    argParser.add_argument("--cognitive_coeff", default = 0.3, help="cognitive coefficient of particle weight update")
    argParser.add_argument("--social_coeff", default = 0.3, help="social coefficient of particle weight update")
    argParser.add_argument("--repel_coeff", default = 0.3, help="repel coefficient of particle weight update")
    argParser.add_argument("--step_length", default = 1, help="step length of the search in the direction of velocity")
    argParser.add_argument("--dropout_rate", default = 0.3, help="dropout rate for DARE merging")
    argParser.add_argument("-p", "--patience", default = 10, help="patience of the search")
    argParser.add_argument("-m", "--max_iteration", default = 200, help="max iteration of the search")
    argParser.add_argument("--weight_randomness", default = 1, help="whether to use weight randomness") # 0, 1
    argParser.add_argument("-i", "--initial_expert_directory", default="./initial_experts", help="initial expert directory") # make it a directory of initial expert checkpoints, see initial_experts/ for example
    argParser.add_argument("-b", "--base_model", default="google/gemma-7b-it", help="base model of the lora experts")
    argParser.add_argument("--starting_test_set_eval", default=1, help="starting test set evaluation") # 0, 1
    argParser.add_argument("--fast_merge", default=1, help="whether to use fast merge by only loading the safetensor file") # just keep it 1 unless you absolutely know what you're doing
    argParser.add_argument("--project_name_wb", default="swarm", help="wandb project name") # as you wish
    argParser.add_argument("--populate_initial_experts", default=0, help="whether to populate initial experts") # 0, 1
    argParser.add_argument("--initial_experts_num", default=None, help="number of initial experts to populate, when populate flag is 1")
    argParser.add_argument("--starting_velocity_mode", default="random", help="starting velocity mode: zero, random, best") # zero, random, best
    argParser.add_argument("--repel_term", default=1, help="whether to incorporate a repel term with global_worst") # 0, 1
    argParser.add_argument("--step_length_factor", default=0.95, help="step length *= step_length_factor every iteration") # 1 for no scheduling, 0.95 maybe?
    argParser.add_argument("--minimum_step_length", default=0.1, help="minimum step length")
    argParser.add_argument("--restart_stray_particles", default=1, help="whether to restart stray particles") # 0, 1
    argParser.add_argument("--restart_patience", default=0.5, help="restart patience * patience = when to restart particles")
    argParser.add_argument("--clean_up_on_end", default=1, help="whether to clean up on end") # 0, 1
    argParser.add_argument("--only_one_or_two", default=None, help="whether to only optimize with dataset 1 or 2 in multitask") # safely ignore this
    argParser.add_argument("--to_visualize", default=True, help="whether to visualize the search process") # 0, 1, for Fig 8
    argParser.add_argument("--correctness_emergence", default=False, help="whether to track correctness changes wrt iteration") # 0, 1, for Fig 2
    argParser.add_argument("--dropK", default=0, help="dropout-K, 0-1") # for fig 9
    argParser.add_argument("--dropN", default=0, help="dropout-N, 0-1") # for fig 9
    argParser.add_argument("--dare_ties", default=0, help="whether to use DARE-TIES merging") # 0, 1
    argParser.add_argument("--seed", default=42, help="random seed for reproducibility")
    argParser.add_argument("--eval", default=0, help="whether to set the model to evaluation mode") # 0, 1
    argParser.add_argument("--es", default=0, help="whether to use Evolution Strategies during the search") # 0, 1
    argParser.add_argument("--es_k", default=5, help="performs ES on the best particle every k iterations") # 
    argParser.add_argument("--es_alpha", default=0.01, help="alpha for ES") #
    argParser.add_argument("--es_sigma", default=0.01, help="sigma for ES") #
    argParser.add_argument("--es_pop_size", default=20, help="population size for ES") #
    argParser.add_argument("--es_num_iterations", default=3, help="number of iterations for ES") #
    argParser.add_argument("--continue_from", default=None, help="continue from an existing search directory") # directory name in search/

    args = argParser.parse_args()
    search_pass_name = args.name
    eval_type = args.eval_type
    dataset = args.dataset
    gpus = args.gpus
    num_cpu_when_merging = int(args.num_cpu_when_merging)
    inertia = float(args.inertia)
    cognitive_coeff = float(args.cognitive_coeff)
    social_coeff = float(args.social_coeff)
    repel_coeff = float(args.repel_coeff)
    patience = int(args.patience)
    step_length = float(args.step_length)
    dropout_rate = float(args.dropout_rate)
    max_iteration = int(args.max_iteration)
    weight_randomness = int(args.weight_randomness)
    initial_expert_directory = args.initial_expert_directory
    base_model = args.base_model
    starting_test_set_eval = bool(int(args.starting_test_set_eval))
    fast_merge = int(args.fast_merge)
    project_name_wb = args.project_name_wb
    populate_initial_experts = bool(int(args.populate_initial_experts))
    use_dare_ties = bool(int(args.dare_ties))
    seed = int(args.seed)
    evaluation_mode = bool(int(args.eval))
    use_es = bool(int(args.es))
    es_k = int(args.es_k)
    es_alpha = float(args.es_alpha)
    es_sigma = float(args.es_sigma)
    es_pop_size = int(args.es_pop_size)
    es_num_iterations = int(args.es_num_iterations)
    continue_from = args.continue_from
    try:
        initial_experts_num = int(args.initial_experts_num)
    except:
        initial_experts_num = None
    starting_velocity_mode = args.starting_velocity_mode
    repel_term = int(args.repel_term)
    step_length_factor = float(args.step_length_factor)
    minimum_step_length = float(args.minimum_step_length)
    restart_stray_particles = int(args.restart_stray_particles)
    restart_patience = float(args.restart_patience)
    clean_up_on_end = bool(int(args.clean_up_on_end))
    only_one_or_two = args.only_one_or_two
    update_only_one_or_two(only_one_or_two)
    to_visualize_flag = args.to_visualize
    correctness_emergence = args.correctness_emergence
    dropK = float(args.dropK)
    dropN = float(args.dropN)

    search_pass_name += ("_" + socket.gethostname())
    args.name = search_pass_name

    perplexity_extrinsic_test_dict = {
        "legal": ["hearsay", "citation_prediction_classification"],
        "medical": ["medqa", "medmcqa"],
        "science": ["scifact", "stem"],
        "culture": ["normad_country", "normad_value"]
    }
    
    # create search directory
    if continue_from:
        search_pass_name = continue_from
        log_with_flush(f"Continuing from {continue_from} checkpoint \n")
        populate_initial_experts = 0
    else:
        if os.path.exists(os.path.join("search", search_pass_name)):
            search_pass_name += current_time_string().replace(" ", "_")
            # exit("search directory already exists!")
        os.mkdir(os.path.join("search", search_pass_name))
    # write args to file
    with open(os.path.join("search", args.name, "args.txt"), "w") as f:
        f.write(str(args))

    run = wandb.init(name=search_pass_name, project=project_name_wb)
    run.config.update(args)
    try:
        torch.multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass

    # Configure logging to write to a file
    logging.basicConfig(
        filename=os.path.join("search", search_pass_name, "log.txt"),
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True
    )
    
    if seed:
        log_with_flush("setting seed: " + str(seed))
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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

    if correctness_emergence:
        correctness_emergence_dict = {}
        for i in range(len(particle_paths)):
            correctness_emergence_dict[i] = []

    if to_visualize_flag:
        particle_trajectory = {}
        for i in range(len(particle_paths)):
            particle_trajectory[i] = []

    log_with_flush("initializing search... " + current_time_string())

    if not continue_from:
        initialize_search_records(search_pass_name, particle_paths, eval_type, dataset, gpus, base_model, fast_merge, starting_velocity_mode, seed)
        log_with_flush("search initialized")
        for i in range(len(particle_paths)):
            log_with_flush("expert " + str(i) + ": " + particle_paths[i])

        if os.path.exists(os.path.join("search", search_pass_name, "tmp")):
            shutil.rmtree(os.path.join("search", search_pass_name, "tmp"))

        # test set evaluation
        if starting_test_set_eval:
            eval_test_args = []
            for i in range(len(particle_paths)):
                eval_test_args.append((os.path.join("search", search_pass_name, "particle_"+str(i), "now"), eval_type, 
                dataset, gpus[assign_gpu(len(gpus), i, len(particle_paths))], base_model, None, False, seed, evaluation_mode))

            pool = Pool(processes=len(gpus))
            results = pool.starmap(evaluate_test, eval_test_args, chunksize=math.ceil(len(particle_paths)/len(gpus)))
            pool.close()
            pool.join()

            log_with_flush("Test set results:")
            for i in range(len(particle_paths)):
                log_with_flush("particle_"+str(i)+": "+str(results[i]))

        log_with_flush("starting search... "+current_time_string())
        evolved_particles_count = 0
        # main search iteration
        iter_count = 0
    else:
        
        log_with_flush("continuing from existing search directory...")
        # load existing iteration count
        with open(os.path.join("search", search_pass_name, "utility_scratchpad.json")) as f:
            utility_scratchpad = json.load(f)
        g_history = utility_scratchpad["g_history"]
        iter_count = len(g_history)
        log_with_flush("loaded iteration count: " + str(iter_count))
        evolved_particles_count = 0
        log_with_flush("continuing search... "+current_time_string())
    
    
    while iter_count < max_iteration:
        iter_count += 1
        log_with_flush("--------------------------")
        log_with_flush("iteration "+str(iter_count)+"! "+current_time_string())
        log_with_flush("updating particles...")
        # patience and ending condition
        with open(os.path.join("search", search_pass_name, "utility_scratchpad.json")) as f:
            utility_scratchpad = json.load(f)
        g_best = utility_scratchpad["g"]
        g_history = utility_scratchpad["g_history"]
        if len(g_history) > patience:
            g_history = g_history[-patience:]
            # if g_history hasn't changed
            if max(g_history) == min(g_history):
                log_with_flush("patience reached!")
                break

        if to_visualize_flag:
            for i in range(len(particle_paths)):
                lora_weight_path = os.path.join("search", search_pass_name, "particle_"+str(i), "now", "adapter_model.safetensors")
                coords = lora_weight_visualize(lora_weight_path)
                particle_trajectory[i].append(coords)
            with open(os.path.join("search", search_pass_name, "particle_trajectory.json"), "w") as f:
                json.dump(particle_trajectory, f, indent=4)
            
            with open(os.path.join("search", search_pass_name, "particle_trajectory.json"), "w") as f:
                json.dump(particle_trajectory, f, indent=4)

        if correctness_emergence:
            for i in range(len(particle_paths)):
                model_path = os.path.join("search", search_pass_name, "particle_"+str(i), "now")
                golds = json.load(open(os.path.join(model_path, "golds_dev.json"), "r"))
                preds = json.load(open(os.path.join(model_path, "preds_dev.json"), "r"))
                correctness = []
                assert len(golds) == len(preds)
                for j in range(len(golds)):
                    if golds[j] == preds[j]:
                        correctness.append(1)
                    else:
                        correctness.append(0)
                correctness_emergence_dict[i].append(correctness)
            
            with open(os.path.join("search", search_pass_name, "correctness_emergence.json"), "w") as f:
                json.dump(correctness_emergence_dict, f, indent=4)
        
        # update each particle
        update_args = []
        for i in range(len(particle_paths)):
            if restart_stray_particles:
                particle_history = utility_scratchpad["particle_"+str(i)+"_history"]
                particle_best_so_far = utility_scratchpad["particle_"+str(i)+"_best"]
                first_time_best_idx = particle_history.index(particle_best_so_far)
                if len(particle_history) - first_time_best_idx >= restart_patience * patience:
                    restart_flag = True
                    log_with_flush("particle_"+str(i)+" restarted!")
                else:
                    restart_flag = False
            else:
                restart_flag = False

            update_args.append((i, gpus[assign_gpu(len(gpus), i, len(particle_paths))], search_pass_name,
                weight_randomness, inertia, cognitive_coeff, social_coeff, repel_coeff, 
                fast_merge, step_length, repel_term, restart_flag, seed))

        pool = Pool(processes=num_cpu_when_merging)
        results = pool.starmap(particle_update, update_args, chunksize=math.ceil(len(particle_paths)/len(gpus)))
        pool.close()
        pool.join()
        log_with_flush("all particles updated! "+current_time_string())

        # perform Evolution strategies on the best particle every k iterations. Substitute the worst particle with the ES result if best is improved.
        if use_es and es_k > 0 and iter_count % es_k == 0:    
            # identify the current best and worst particles
            with open("search/"+search_pass_name+"/utility_scratchpad.json", "r") as f:
                utility_scratchpad = json.load(f)
            curr_best = -float('inf')
            curr_worst = float('inf')
            for i in range(len(particle_paths)):
                if utility_scratchpad["particle_" + str(i) + "_now"] > curr_best:
                    curr_best_particle = i
                    curr_best = utility_scratchpad["particle_" + str(i) + "_now"]
                if utility_scratchpad["particle_" + str(i) + "_now"] < curr_worst:
                    curr_worst_particle = i
                    curr_worst = utility_scratchpad["particle_" + str(i) + "_now"]
            # substitute the worst particle with the best particle before ES
            worst_particle_path = os.path.join("search", search_pass_name, "particle_"+str(curr_worst_particle))
            best_particle_path = os.path.join("search", search_pass_name, "particle_"+str(curr_best_particle))
            es_work_path = os.path.join("search", search_pass_name, "es_scratch")

            # Prepare ES scratch copy
            if os.path.exists(es_work_path):
                shutil.rmtree(es_work_path)
            shutil.copytree(best_particle_path, es_work_path)
            
            # perform ES on the best particle
            es_log_string = (f"Iteration:{iter_count}: executing ES on particle {curr_best_particle}\n"
                             f"Hyperparams: alpha={es_alpha}, sigma={es_sigma}, pop_size={es_pop_size}, num_iterations={es_num_iterations}\n"
                             f"Before ES: utility={utility_scratchpad['particle_' + str(curr_best_particle) + '_now']}\n")
            log_with_flush(es_log_string)
            es_eval, es_out_path = es_lora(
                lora_path=os.path.join(es_work_path, "now"),
                eval_type=eval_type,
                dataset=dataset,
                seed=seed,
                base_model=base_model,
                ALPHA=es_alpha,
                SIGMA=es_sigma,
                POPULATION_SIZE=es_pop_size,
                NUM_ITERATIONS=es_num_iterations,
            )
            log_with_flush(f"After ES: utility={es_eval}\n Saved to {es_out_path}\n")
            # Substitute worst particle with ES result if improved
            if es_eval > curr_best:
                # Replace worst particle with clone of best
                if os.path.exists(worst_particle_path):
                    shutil.rmtree(worst_particle_path)
                shutil.copytree(best_particle_path, worst_particle_path)

                # Overwrite 'now' with ES result
                worst_now_path = os.path.join(worst_particle_path, "now")
                if os.path.exists(worst_now_path):
                    shutil.rmtree(worst_now_path)
                shutil.copytree(es_out_path, worst_now_path)

                # initialize new particle velocity to zero
                lora_merge(
                    weights=[0],
                    lora_name_list=[worst_now_path],
                    output_name=os.path.join(worst_particle_path, "velocity"),
                    gpu_id=gpus[assign_gpu(len(gpus), curr_worst_particle, len(particle_paths))],
                    directly_load_safetensors=fast_merge
                )

                log_with_flush(
                    f"Particle {curr_worst_particle} replaced by ES-refined clone "
                    f"of particle {curr_best_particle}\n"
                )
                evolved_particles_count += 1
            else:
                log_with_flush(
                    f"ES did not improve over best particle {curr_best_particle}. "
                    f"No substitution made.\n"
                )

            # Clean up ES scratch
            if os.path.exists(es_work_path):
                shutil.rmtree(es_work_path)
        
        # evaluate each particle and update utility_scratchpad and weights
        log_with_flush("evaluating particles...")

        if random.random() < dropK: # iteration drop
            log_with_flush("dropped iteration!")
            global_skip_flag = True
        else:
            global_skip_flag = False

        eval_args = []
        for i in range(len(particle_paths)):

            if random.random() < dropN: # particle drop
                local_skip_flag = True
            else:
                local_skip_flag = False

            if not correctness_emergence:
                eval_args.append((os.path.join("search", search_pass_name, "particle_"+str(i), "now"), eval_type, 
                dataset, gpus[assign_gpu(len(gpus), i, len(particle_paths))], base_model, False, None, 
                global_skip_flag or local_skip_flag, seed
                ))
            else:
                eval_args.append((os.path.join("search", search_pass_name, "particle_"+str(i), "now"), eval_type, 
                dataset, gpus[assign_gpu(len(gpus), i, len(particle_paths))], base_model, True, None, 
                global_skip_flag or local_skip_flag, seed
                ))
        
        pool = Pool(processes=len(gpus))
        results = pool.starmap(evaluate, eval_args, chunksize=math.ceil(len(particle_paths)/len(gpus)))
        pool.close()
        pool.join()

        with open("search/"+search_pass_name+"/utility_scratchpad.json", "r") as f:
            utility_scratchpad = json.load(f)

        # if skipped, pull performance from last step
        for i in range(len(particle_paths)):
            if results[i] is None:
                results[i] = utility_scratchpad["particle_"+str(i)+"_now"]
                assert results[i] == utility_scratchpad["particle_"+str(i)+"_history"][-1]
        
        # personal bests update
        for i in range(len(particle_paths)):
            utility_scratchpad["particle_" + str(i) + "_now"] = results[i]
            utility_scratchpad["particle_" + str(i) + "_history"].append(results[i])
            if results[i] > utility_scratchpad["particle_" + str(i) + "_best"]:
                utility_scratchpad["particle_" + str(i) + "_best"] = results[i]
                shutil.copytree(os.path.join("search", search_pass_name, "particle_"+str(i), "now"), os.path.join("search", search_pass_name, "particle_"+str(i), "personal_best"), dirs_exist_ok=True)
                log_with_flush("new personal best for particle_"+str(i)+": "+str(results[i]))
        
        # global best update
        if max(results) > utility_scratchpad["g"]:
            utility_scratchpad["g"] = max(results)
            utility_scratchpad["g_history"].append(max(results))
            log_with_flush("new global best: "+str(utility_scratchpad["g"]))
            for i in range(len(particle_paths)):
                if results[i] == utility_scratchpad["g"]:
                    shutil.copytree(os.path.join("search", search_pass_name, "particle_"+str(i), "now"), os.path.join("search", search_pass_name, "global_best"), dirs_exist_ok=True)
                    break
        else:
            utility_scratchpad["g_history"].append(utility_scratchpad["g"])

        # global worst update
        if min(results) < utility_scratchpad["g_worst"]:
            utility_scratchpad["g_worst"] = min(results)
            for i in range(len(particle_paths)):
                if results[i] == utility_scratchpad["g_worst"]:
                    shutil.copytree(os.path.join("search", search_pass_name, "particle_"+str(i), "now"), os.path.join("search", search_pass_name, "global_worst"), dirs_exist_ok=True)

        wandb_log = {
            "g": utility_scratchpad["g"],
            "g_worst": utility_scratchpad["g_worst"],
        }
        for i in range(len(particle_paths)):
            wandb_log["particle_" + str(i) + "_now"] = utility_scratchpad["particle_" + str(i) + "_now"]
        
        wandb_log["iterations"] = int(iter_count)
        wandb_log["evolved_particles_count"] = int(evolved_particles_count)
        wandb.log(wandb_log)
        
        with open("search/"+search_pass_name+"/utility_scratchpad.json", "w") as f:
            json.dump(utility_scratchpad, f, indent=4)
        
        log_with_flush("all particles evaluated! "+current_time_string())
        log_with_flush("--------------------------")

        # step length update
        step_length = max(step_length * step_length_factor, minimum_step_length)

    if to_visualize_flag:
        plot_particle_trajectories(search_pass_name, dataset)

    log_with_flush("ending search and starting test set evaluation... "+current_time_string())

    # which particle is global best?
    with open("search/"+search_pass_name+"/utility_scratchpad.json", "r") as f:
        utility_scratchpad = json.load(f)
    g_best = utility_scratchpad["g"]
    for i in range(len(particle_paths)):
        if utility_scratchpad["particle_" + str(i) + "_best"] == g_best:
            global_best_particle = i
            log_with_flush("global best particle: "+str(global_best_particle))

    # dev set evaluation for personal bests
    eval_args = []
    for i in range(len(particle_paths)):
        eval_args.append((os.path.join("search", search_pass_name, "particle_"+str(i), "personal_best"), eval_type,
        dataset, gpus[assign_gpu(len(gpus), i, len(particle_paths))], base_model, True, None, False, seed
        ))

    pool = Pool(processes=len(gpus))
    results = pool.starmap(evaluate, eval_args, chunksize=math.ceil(len(particle_paths)/len(gpus)))
    pool.close()
    pool.join()

    # test set evaluation
    eval_test_args = []
    for i in range(len(particle_paths)):
        eval_test_args.append((os.path.join("search", search_pass_name, "particle_"+str(i), "personal_best"), eval_type, 
        dataset, gpus[assign_gpu(len(gpus), i, len(particle_paths))], base_model, None, False, seed, evaluation_mode
        ))

    pool = Pool(processes=len(gpus))
    results = pool.starmap(evaluate_test, eval_test_args, chunksize=math.ceil(len(particle_paths)/len(gpus)))
    pool.close()
    pool.join()

    log_with_flush("Test set results:")
    for i in range(len(particle_paths)):
        log_with_flush("particle_"+str(i)+": "+str(results[i]))

    final_metrics = overall_metrics(search_pass_name, eval_type, initial_experts_num = len(particle_paths))
    log_with_flush("final metrics computed 1: "+str(final_metrics))
    
    if eval_type == "AbstainQA":
        best_particle_idx = final_metrics["ending_best_particle_on_validation"]
        final_metrics["ending_best_single_test_accuracy"] = results[best_particle_idx]
    
    if eval_type == "perplexity" or eval_type == "multitask":
        dataset_1_name = perplexity_extrinsic_test_dict[dataset][0]
        eval_test_args = []
        for i in range(len(particle_paths)):
            eval_test_args.append((os.path.join("search", search_pass_name, "particle_"+str(i), "personal_best"), "multiple_choice", dataset_1_name, gpus[assign_gpu(len(gpus), i, len(particle_paths))], base_model, None, False, seed, evaluation_mode))
        
        pool = Pool(processes=len(gpus))
        results = pool.starmap(evaluate_test, eval_test_args, chunksize=math.ceil(len(particle_paths)/len(gpus)))
        pool.close()
        pool.join()

        final_metrics["ending_best_single_test_" + dataset_1_name] = results[final_metrics["ending_best_particle_on_validation"]]

        dataset_2_name = perplexity_extrinsic_test_dict[dataset][1]
        eval_test_args = []
        for i in range(len(particle_paths)):
            eval_test_args.append((os.path.join("search", search_pass_name, "particle_"+str(i), "personal_best"), "multiple_choice", dataset_2_name, gpus[assign_gpu(len(gpus), i, len(particle_paths))], base_model, None, False, seed, evaluation_mode))
        
        pool = Pool(processes=len(gpus))
        results = pool.starmap(evaluate_test, eval_test_args, chunksize=math.ceil(len(particle_paths)/len(gpus)))
        pool.close()
        pool.join()

        final_metrics["ending_best_single_test_" + dataset_2_name] = results[final_metrics["ending_best_particle_on_validation"]]

    if eval_type == "multitask":
        dataset_1_name = perplexity_extrinsic_test_dict[dataset][0]
        eval_args = []
        for i in range(len(particle_paths)):
            eval_args.append((os.path.join("search", search_pass_name, "particle_"+str(i), "personal_best"), "multiple_choice", dataset_1_name, gpus[assign_gpu(len(gpus), i, len(particle_paths))], base_model, None, False, seed))
        
        pool = Pool(processes=len(gpus))
        results = pool.starmap(evaluate, eval_args, chunksize=math.ceil(len(particle_paths)/len(gpus)))
        pool.close()
        pool.join()

        final_metrics["ending_best_single_dev_" + dataset_1_name] = results[final_metrics["ending_best_particle_on_validation"]]

        dataset_2_name = perplexity_extrinsic_test_dict[dataset][1]
        eval_args = []
        for i in range(len(particle_paths)):
            eval_args.append((os.path.join("search", search_pass_name, "particle_"+str(i), "personal_best"), "multiple_choice", dataset_2_name, gpus[assign_gpu(len(gpus), i, len(particle_paths))], base_model, None, False, seed))

        pool = Pool(processes=len(gpus))
        results = pool.starmap(evaluate, eval_args, chunksize=math.ceil(len(particle_paths)/len(gpus)))
        pool.close()
        pool.join()

        final_metrics["ending_best_single_dev_" + dataset_2_name] = results[final_metrics["ending_best_particle_on_validation"]]

    wandb.log(final_metrics)
    log_with_flush("final metrics for test: "+str(final_metrics))

    # ensemble for dev set
    try:
        for i in range(len(particle_paths)):
            os.remove(os.path.join("search", search_pass_name, "particle_"+str(i), "personal_best", "golds.json"))
            os.remove(os.path.join("search", search_pass_name, "particle_"+str(i), "personal_best", "preds.json"))
            os.remove(os.path.join("search", search_pass_name, "particle_"+str(i), "now", "golds.json"))
            os.remove(os.path.join("search", search_pass_name, "particle_"+str(i), "now", "preds.json"))

            os.rename(os.path.join("search", search_pass_name, "particle_"+str(i), "personal_best", "golds_dev.json"), os.path.join("search", search_pass_name, "particle_"+str(i), "personal_best", "golds.json"))
            os.rename(os.path.join("search", search_pass_name, "particle_"+str(i), "personal_best", "preds_dev.json"), os.path.join("search", search_pass_name, "particle_"+str(i), "personal_best", "preds.json"))
            os.rename(os.path.join("search", search_pass_name, "particle_"+str(i), "now", "golds_dev.json"), os.path.join("search", search_pass_name, "particle_"+str(i), "now", "golds.json"))
            os.rename(os.path.join("search", search_pass_name, "particle_"+str(i), "now", "preds_dev.json"), os.path.join("search", search_pass_name, "particle_"+str(i), "now", "preds.json"))
    except:
        for i in range(len(particle_paths)):
            os.remove(os.path.join("search", search_pass_name, "particle_"+str(i), "personal_best", "scores.json"))
            os.remove(os.path.join("search", search_pass_name, "particle_"+str(i), "now", "scores.json"))

            os.rename(os.path.join("search", search_pass_name, "particle_"+str(i), "personal_best", "scores_dev.json"), os.path.join("search", search_pass_name, "particle_"+str(i), "personal_best", "scores.json"))
            os.rename(os.path.join("search", search_pass_name, "particle_"+str(i), "now", "scores_dev.json"), os.path.join("search", search_pass_name, "particle_"+str(i), "now", "scores.json"))

    final_metrics = overall_metrics(search_pass_name, eval_type)
    log_with_flush("final metrics computed 2: "+str(final_metrics))

    dev_final_metrics = {
        "starting_top-k_ensemble_dev_accuracy": final_metrics["starting_top-k_ensemble_test_accuracy"],
        "ending_top-k_ensemble_dev_accuracy": final_metrics["ending_top-k_ensemble_test_accuracy"]
    }
    wandb.log(dev_final_metrics)
    log_with_flush("final ensemble metrics for dev: "+str(dev_final_metrics))

    if clean_up_on_end:
        shutil.rmtree(os.path.join("search", search_pass_name, "global_worst"))
        for i in range(len(particle_paths)):
            for aux in ["g_x", "p_x", "velocity", "x_w"]:
                shutil.rmtree(os.path.join("search", search_pass_name, "particle_"+str(i), aux))

    log_with_flush("the end of search... "+current_time_string())