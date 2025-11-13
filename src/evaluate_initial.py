from evaluate import evaluate_test
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

def overall_metrics_init(name, eval_type, top_k = 10):

    final_metrics = {}

    particle_paths = os.listdir(os.path.join("initial_eval", name))

    if eval_type == "multiple_choice" or eval_type == "AbstainQA" or eval_type == "multitask":
        
        golds = None
        starting_preds = []
        starting_utility = []

        ending_preds = []
        ending_utility = []

        # for particle_path in particle_paths:
        #     if "particle" in particle_path:
        for i in range(len(particle_paths)):
            if "particle_" + str(i) in particle_paths:
                particle_path = "particle_" + str(i)
                with open(os.path.join("initial_eval", name, particle_path, "personal_best/preds.json"), "r") as f:
                    particle_data = json.load(f)
                    ending_preds.append(particle_data)
                with open(os.path.join("initial_eval", name, particle_path, "personal_best/golds.json"), "r") as f:
                    gold_data = json.load(f)
                    if golds and not eval_type == "AbstainQA":
                        assert golds == gold_data
                    else:
                        golds = gold_data
        
        with open(os.path.join("initial_eval", name, "utility_scratchpad.json"), "r") as f:
            utility_data = json.load(f)
            for i in range(len(particle_paths)):
                if "particle_" + str(i) in particle_paths:
                    particle_path = "particle_" + str(i)
                    ending_utility.append(utility_data[particle_path + "_best"])

        starting_eval_flag = True
        try:
            for i in range(len(particle_paths)):
                if "particle_" + str(i) in particle_paths:
                    particle_path = "particle_" + str(i)
                    with open(os.path.join("initial_eval", name, particle_path, "now/preds.json"), "r") as f:
                        particle_data = json.load(f)
                        starting_preds.append(particle_data)
                    with open(os.path.join("initial_eval", name, particle_path, "now/golds.json"), "r") as f:
                        gold_data = json.load(f)
                        if golds and not eval_type == "AbstainQA":
                            assert golds == gold_data
                        else:
                            golds = gold_data
            
            with open(os.path.join("initial_eval", name, "utility_scratchpad.json"), "r") as f:
                utility_data = json.load(f)
                for i in range(len(particle_paths)):
                    if "particle_" + str(i) in particle_paths:
                        particle_path = "particle_" + str(i)
                        starting_utility.append(utility_data[particle_path + "_history"][0])
        except:
            print("no starting eval! starting will be the same as ending")
            starting_eval_flag = False
            starting_utility = ending_utility
            starting_preds = ending_preds

        assert len(starting_preds) == len(starting_utility) == len(ending_preds) == len(ending_utility)
        assert len(golds) == len(starting_preds[0]) == len(ending_preds[0])

        final_starting_preds = ensemble_based_on_utility(starting_preds, starting_utility, top_k)
        final_ending_preds = ensemble_based_on_utility(ending_preds, ending_utility, top_k)

        assert len(final_starting_preds) == len(final_ending_preds)

        starting_best_utility_index = starting_utility.index(max(starting_utility))
        ending_best_utility_index = ending_utility.index(max(ending_utility))

        if starting_eval_flag:

            final_metrics["starting_best_validation_utility"] = max(starting_utility)
            final_metrics["starting_best_particle_on_validation"] = starting_best_utility_index
            final_metrics["starting_best_single_test_accuracy"] = accuracy_score(golds, starting_preds[starting_best_utility_index])
            final_metrics["starting_top-k_ensemble_test_accuracy"] = accuracy_score(golds, final_starting_preds)

            print("starting best validation utility: ", max(starting_utility))
            print("starting best particle on validation: ", starting_best_utility_index)
            print("starting best single test accuracy: ", accuracy_score(golds, starting_preds[starting_best_utility_index]))
            print("starting top-k ensemble test accuracy: ", accuracy_score(golds, final_starting_preds))

        final_metrics["ending_best_validation_utility"] = max(ending_utility)
        final_metrics["ending_best_particle_on_validation"] = ending_best_utility_index
        final_metrics["ending_best_single_test_accuracy"] = accuracy_score(golds, ending_preds[ending_best_utility_index])
        final_metrics["ending_top-k_ensemble_test_accuracy"] = accuracy_score(golds, final_ending_preds)

        print("ending best validation utility: ", max(ending_utility))
        print("ending best particle on validation: ", ending_best_utility_index)
        print("ending best single test accuracy: ", accuracy_score(golds, ending_preds[ending_best_utility_index]))
        print("ending top-k ensemble test accuracy: ", accuracy_score(golds, final_ending_preds))
    
    elif eval_type == "exact_match" or eval_type == "external_api" or eval_type == "perplexity" or eval_type == "rm_default" or eval_type == "rm_concise" or eval_type == "rm_verbose" or eval_type == "human":
        starting_scores = []
        starting_utility = []
        ending_scores = []
        ending_utility = []

        for i in range(len(particle_paths)):
            if "particle_" + str(i) in particle_paths:
                particle_path = "particle_" + str(i)
                with open(os.path.join("initial_eval", name, particle_path, "personal_best/scores.json"), "r") as f:
                    particle_data = json.load(f)
                    ending_scores.append(particle_data)
        
        with open(os.path.join("initial_eval", name, "utility_scratchpad.json"), "r") as f:
            utility_data = json.load(f)
            for i in range(len(particle_paths)):
                if "particle_" + str(i) in particle_paths:
                    particle_path = "particle_" + str(i)
                    ending_utility.append(utility_data[particle_path + "_best"])

        starting_eval_flag = True
        try:
            for i in range(len(particle_paths)):
                if "particle_" + str(i) in particle_paths:
                    particle_path = "particle_" + str(i)
                    with open(os.path.join("initial_eval", name, particle_path, "now/scores.json"), "r") as f:
                        particle_data = json.load(f)
                        starting_scores.append(particle_data)
            
            with open(os.path.join("initial_eval", name, "utility_scratchpad.json"), "r") as f:
                utility_data = json.load(f)
                for i in range(len(particle_paths)):
                    if "particle_" + str(i) in particle_paths:
                        particle_path = "particle_" + str(i)
                        starting_utility.append(utility_data[particle_path + "_history"][0])
        except:
            print("no starting eval! starting will be the same as ending")
            starting_eval_flag = False
            starting_utility = ending_utility
            starting_scores = ending_preds
        
        assert len(starting_scores) == len(starting_utility) == len(ending_scores) == len(ending_utility)

        starting_best_utility_index = starting_utility.index(max(starting_utility))
        ending_best_utility_index = ending_utility.index(max(ending_utility))

        if starting_eval_flag:

            final_metrics["starting_best_validation_utility"] = max(starting_utility)
            final_metrics["starting_best_particle_on_validation"] = starting_best_utility_index
            final_metrics["starting_best_single_test_accuracy"] = sum(starting_scores[starting_best_utility_index]) / len(starting_scores[starting_best_utility_index])
            if eval_type == "exact_match" or eval_type == "rm_default" or eval_type == "rm_concise" or eval_type == "rm_verbose" or eval_type == "human":
                temp_scores = ensemble_based_on_utility(starting_scores, starting_utility, top_k)
                final_metrics["starting_top-k_ensemble_test_accuracy"] = sum(temp_scores) / len(temp_scores)
            elif eval_type == "external_api":
                top_k_utility_bar = sorted(starting_utility, reverse=True)[:top_k]
                retained_scores_list = []
                for i in range(len(starting_scores)):
                    if starting_utility[i] in top_k_utility_bar:
                        retained_scores_list.append(starting_scores[i])
                # average the scores
                final_metrics["starting_top-k_ensemble_test_accuracy"] = avg([sum(x) / len(x) for x in retained_scores_list])
                final_metrics["starting_top-k_ensemble_test_accuracy"] = avg([sum(x) / len(x) for x in retained_scores_list])
            elif eval_type == "perplexity":
                final_metrics["starting_top-k_ensemble_test_accuracy"] = None

            print("starting best validation utility: ", final_metrics["starting_best_validation_utility"])
            print("starting best particle on validation: ", starting_best_utility_index)
            print("starting best single test accuracy: ", final_metrics["starting_best_single_test_accuracy"])
            print("starting top-k ensemble test accuracy: ", final_metrics["starting_top-k_ensemble_test_accuracy"])

        final_metrics["ending_best_validation_utility"] = max(ending_utility)
        final_metrics["ending_best_particle_on_validation"] = ending_best_utility_index
        final_metrics["ending_best_single_test_accuracy"] = sum(ending_scores[ending_best_utility_index]) / len(ending_scores[ending_best_utility_index])
        if eval_type == "exact_match" or eval_type == "rm_default" or eval_type == "rm_concise" or eval_type == "rm_verbose" or eval_type == "human":
            temp_scores = ensemble_based_on_utility(ending_scores, ending_utility, top_k)
            final_metrics["ending_top-k_ensemble_test_accuracy"] = sum(temp_scores) / len(temp_scores)
        elif eval_type == "external_api":
            top_k_utility_bar = sorted(ending_utility, reverse=True)[:top_k]
            retained_scores_list = []
            for i in range(len(ending_scores)):
                if ending_utility[i] in top_k_utility_bar:
                    retained_scores_list.append(ending_scores[i])
            # average the scores
            final_metrics["ending_top-k_ensemble_test_accuracy"] = avg([sum(x) / len(x) for x in retained_scores_list])
            final_metrics["ending_top-k_ensemble_test_accuracy"] = avg([sum(x) / len(x) for x in retained_scores_list])
        elif eval_type == "perplexity":
            final_metrics["ending_top-k_ensemble_test_accuracy"] = None
        # final_metrics["ending_top-k_ensemble_test_accuracy"] = accuracy_score(golds, final_ending_preds)

        print("ending best validation utility: ", final_metrics["ending_best_validation_utility"])
        print("ending best particle on validation: ", final_metrics["ending_best_particle_on_validation"])
        print("ending best single test accuracy: ", final_metrics["ending_best_single_test_accuracy"])
        print("ending top-k ensemble test accuracy: ", final_metrics["ending_top-k_ensemble_test_accuracy"])

    return final_metrics

def initialize_eval_records(search_pass_name, particle_paths, eval_type, dataset, gpus, base_model, fast_merge):
    for i in range(len(particle_paths)):
        os.mkdir(os.path.join("initial_eval", search_pass_name, "particle_"+str(i)))
        os.mkdir(os.path.join("initial_eval", search_pass_name, "particle_"+str(i), "now"))
        shutil.copytree(particle_paths[i], os.path.join("initial_eval", search_pass_name, "particle_"+str(i), "now"), dirs_exist_ok=True)

if __name__ == "__main__":

    argParser = argparse.ArgumentParser()
    argParser.add_argument("-n", "--name", help="name of this model swarms search, also directory name in initial_eval/")
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

    final_metrics = overall_metrics_init(search_pass_name, eval_type)
    wandb.log(final_metrics)
    log_with_flush("final metrics for test: "+str(final_metrics))