from evaluate import evaluate_test
from overall_metrics import overall_metrics
from merge import lora_merge, dare_ties_merge
from multiprocessing import Pool
from search import assign_gpu, log_with_flush, current_time_string
import argparse
import random
import logging
import datetime
import wandb
import torch
import shutil
import os
import math

def initialize_eval_records(search_pass_name, particle_paths, eval_type, dataset, gpus, base_model, fast_merge):
    os.mkdir(os.path.join("initial_eval", search_pass_name, "tmp"))
    for i in range(len(particle_paths)):
        os.mkdir(os.path.join("initial_eval", search_pass_name, "particle_"+str(i)))
        os.mkdir(os.path.join("initial_eval", search_pass_name, "particle_"+str(i), "now"))
        shutil.copytree(particle_paths[i], os.path.join("initial_eval", search_pass_name, "particle_"+str(i), "now"), dirs_exist_ok=True)

if __name__ == "__main__":

    argParser = argparse.ArgumentParser()
    argParser.add_argument("-n", "--name", help="name of this model swarms search, also directory name in search/")
    argParser.add_argument("-e", "--eval_type", help="evaluation types") # multiple_choice, exact_match, multitask, rm_default, rm_verbose, rm_concise, human
    argParser.add_argument("-d", "--dataset", help="dataset as the search objective/evaluation") # file names in data/eval, be mindful of using the right --eval_type
    argParser.add_argument("-g", "--gpus", help="available gpu ids in a string") # such as 0,1,2,3,4
    argParser.add_argument("--num_cpu_when_merging", default=1, help="number of cpu cores when merging") # you don't need to change this honestly
    argParser.add_argument("--dropout_rate", default = 0.3, help="dropout rate for DARE merging")
    argParser.add_argument("-i", "--initial_expert_directory", default="./initial_experts", help="initial expert directory") # make it a directory of initial expert checkpoints, see initial_experts/ for example
    argParser.add_argument("-b", "--base_model", default="google/gemma-7b-it", help="base model of the lora experts")
    argParser.add_argument("--fast_merge", default=1, help="whether to use fast merge by only loading the safetensor file") # just keep it 1 unless you absolutely know what you're doing
    argParser.add_argument("--project_name_wb", default="swarm", help="wandb project name") # as you wish
    argParser.add_argument("--populate_initial_experts", default=0, help="whether to populate initial experts") # 0, 1
    argParser.add_argument("--initial_experts_num", default=None, help="number of initial experts to populate, when populate flag is 1")
    argParser.add_argument("--clean_up_on_end", default=1, help="whether to clean up on end") # 0, 1
    argParser.add_argument("--dare_ties", default=0, help="whether to use DARE-TIES merging") # 0, 1
    argParser.add_argument("--seed", default=42, help="random seed for reproducibility")
    args = argParser.parse_args()
    
    search_pass_name = args.name
    eval_type = args.eval_type
    dataset = args.dataset
    gpus = args.gpus
    num_cpu_when_merging = int(args.num_cpu_when_merging)
    dropout_rate = float(args.dropout_rate)
    initial_expert_directory = args.initial_expert_directory
    base_model = args.base_model
    fast_merge = int(args.fast_merge)
    project_name_wb = args.project_name_wb
    populate_initial_experts = int(args.populate_initial_experts)
    try:    
        initial_experts_num = int(args.initial_experts_num)
    except:
        initial_experts_num = None
    clean_up_on_end = int(args.clean_up_on_end)
    use_dare_ties = int(args.dare_ties)
    seed = int(args.seed)

    # create initial_eval directory

    if os.path.exists(os.path.join("initial_eval", search_pass_name)):
        search_pass_name += current_time_string().replace(" ", "_")
        # exit("search directory already exists!")
    os.mkdir(os.path.join("initial_eval", search_pass_name))

    # write args to file
    with open(os.path.join("initial_eval", args.name, "args.txt"), "w") as f:
        f.write(str(args))

    run = wandb.init(name=search_pass_name, project=project_name_wb)
    run.config.update(args)
    torch.multiprocessing.set_start_method('spawn')
    if seed:
        random.seed(seed)
        torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)

    # Configure logging to write to a file
    logging.basicConfig(filename=os.path.join("initial_eval", search_pass_name, "log.txt"), level=logging.DEBUG)

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

        os.mkdir(os.path.join("initial_eval", search_pass_name, "tmp"))
        particles_now = len(particle_paths)
        for i in range(initial_experts_num - particles_now):
            parent_1 = random.choice(particle_paths)
            parent_2 = random.choice(particle_paths)
            while parent_1 == parent_2:
                parent_2 = random.choice(particle_paths)
            child_path = os.path.join("initial_eval", search_pass_name, "tmp", "child_"+str(i))
            w_1 = random.random() * 2 # half interpolation, half extrapolation
            w_2 = 1 - w_1
            shutil.copytree(parent_1, child_path)
            
            if use_dare_ties:
                dare_ties_merge([w_1, w_2], [parent_1, parent_2], child_path, gpus[0], directly_load_safetensors=1, density=dropout_rate)
            else:
                lora_merge([w_1, w_2], [parent_1, parent_2], child_path, gpus[0], fast_merge)
            
            particle_paths.append(child_path)

    log_with_flush("initializing evaluation... " + current_time_string())
    initialize_eval_records(search_pass_name, particle_paths, eval_type, dataset, gpus, base_model, fast_merge)
    log_with_flush("evaluation initialized")

    for i in range(len(particle_paths)):
        log_with_flush("expert " + str(i) + ": " + particle_paths[i])

    if os.path.exists(os.path.join("initial_eval", search_pass_name, "tmp")):
        shutil.rmtree(os.path.join("initial_eval", search_pass_name, "tmp"))
        
    eval_test_args = []
    for i in range(len(particle_paths)):
        eval_test_args.append((os.path.join("initial_eval", search_pass_name, "particle_"+str(i), "now"), eval_type, 
        dataset, gpus[assign_gpu(len(gpus), i, len(particle_paths))], base_model, None, False, seed))

    pool = Pool(processes=len(gpus))
    results = pool.starmap(evaluate_test, eval_test_args, chunksize=math.ceil(len(particle_paths)/len(gpus)))
    pool.close()
    pool.join()

    log_with_flush("Test set results:")
    for i in range(len(particle_paths)):
        log_with_flush("particle_"+str(i)+": "+str(results[i]))

    final_metrics = overall_metrics(search_pass_name, eval_type)
    wandb.log(final_metrics)
    log_with_flush("final metrics for test: "+str(final_metrics))