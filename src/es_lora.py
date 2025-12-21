import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import copy
import os
import argparse
import shutil
from accelerate import Accelerator
import time
import torch.multiprocessing as mp
import threading
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
import math
import gc
from tqdm import tqdm 
from evaluate import evaluate, evaluate_test
from peft import LoraConfig, PeftModel, get_peft_model
from safetensors.torch import load_file, save_file
import logging
import wandb


torch.backends.cuda.matmul.allow_tf32 = True
os.environ["PYTHONWARNINGS"] = "ignore"
mp.set_start_method('spawn', force=True)
# Reduce verbosity for urllib3 (used by requests/HuggingFace)
logging.getLogger("urllib3").setLevel(logging.WARNING)

def log_with_flush(message, level=logging.INFO):
  logging.log(level, message)
  logging.getLogger().handlers[0].flush()

def force_memory_cleanup():
    """Force aggressive memory cleanup"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()


def process_seed(seed_args):
    """Function to process a single seed, used for thread pool"""
    seed_idx, seed, SIGMA, lora_path, eval_type, dataset, gpu_id, accelerator, thread_id, verbose = seed_args
    np.random.seed(seed)
    if verbose:
        print(f"Process {accelerator.process_index} Thread {thread_id} processing seed {seed_idx} (value: {seed} lora_path: {lora_path})")

    sd_original = load_file(os.path.join(lora_path, "adapter_model.safetensors"), device="cpu")
    
    # copy LoRA weights for this particle
    sd = {k: v.clone() for k, v in sd_original.items()}

    for name in sd:
        gen = torch.Generator().manual_seed(int(seed))
        noise = torch.randn(sd[name].shape, generator=gen, dtype=sd[name].dtype)
        sd[name] += SIGMA * noise
        # print(f"NORM: {torch.norm(SIGMA * noise) / torch.norm(sd[name])}")

    # Create temp directory for this particle
    tmp_dir = f"/tmp/particle_{thread_id}_{seed_idx}"
    os.makedirs(tmp_dir, exist_ok=True)

    # Save the mutated LoRA weights
    adapter_model_path = os.path.join(tmp_dir, "adapter_model.safetensors")
    save_file(sd, adapter_model_path)
    
    # assert that weights are actually changed
    # sd_perturbed = load_file(adapter_model_path, device="cpu")
    # weights_changed = any(not torch.equal(sd_original[k], sd_perturbed[k]) for k in sd_original)
    # assert weights_changed, "Weights did not change after perturbation!" 
    
    # Copy adapter_config.json (required for load_adapter)
    shutil.copy(
    os.path.join(lora_path, "adapter_config.json"),
    os.path.join(tmp_dir, "adapter_config.json")
    )   

    # Ensure weights are fully loaded before evaluation
    if torch.cuda.is_available():
        torch.cuda.synchronize(accelerator.device)

    # evaluate perturbed model
    # calculate evaluation time
    eval_time = time.time()
    reward = evaluate(tmp_dir, eval_type, dataset, gpu_id, seed=seed)
    eval_time = time.time() - eval_time
    if verbose:
        print(f"Evaluation time : {eval_time:.2f}s")
    # remove temporary file
    try:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    except FileNotFoundError:
        logging.warning(f"Temporary dir {tmp_dir} not found for removal.")

    if torch.cuda.is_available():
        torch.cuda.synchronize(accelerator.device)
    force_memory_cleanup()

    log_string = f"Process {accelerator.process_index} Thread {thread_id} completed seed {seed_idx} with reward {reward:.4f}, lora_path: {lora_path}"
    log_with_flush(log_string)

    if verbose:
        print(log_string)

    return seed_idx, reward


# --- Main Evolution Strategies Loop ---
def es_lora(lora_path, eval_type, dataset, seed, base_model = "google/gemma-7b-it", 
             POPULATION_SIZE=30, NUM_ITERATIONS=10, SIGMA=0.001, ALPHA=0.0005, 
             cache_dir='/scratch/a.dicembre/.hf_cache', gpu_id = 0, verbose=False, gpu_threads=1, eval_starting_test = False):
    
    #  logging.basicConfig(filename=os.path.join(lora_path, "log.txt"),
    #                     level=logging.DEBUG,
    #                     format="%(asctime)s - %(levelname)s - %(message)s",
    #                     force=True)
    
    wandb_log = {}
    wandb_log["es_iteration"] = 0
    wandb_log["es_mean_reward"] = 0.0
    wandb_log["es_max_reward"] = 0.0
    wandb_log["es_min_reward"] = 0.0
    wandb_log["es_initial_evaluation_reward"] = 0.0
    wandb_log["es_final_evaluation_reward"] = 0.0   
    wandb_log["es_iter_initial_evaluation_reward"] = 0.0
    wandb_log["es_best_iteration"] = 0
    wandb.log(wandb_log)

    accelerator = Accelerator()
    if accelerator.is_main_process:
      # print(f"Total processes: {accelerator.num_processes}, GPU threads per process: {gpu_threads}")
        log_string = f"Population size: {POPULATION_SIZE}, Iterations: {NUM_ITERATIONS}\nSigma: {SIGMA}, Alpha: {ALPHA}"
        log_with_flush(log_string)
        print(log_string)

    lora_list = []
    for _ in range(gpu_threads):
        lora_list.append(lora_path)
    
    np.random.seed(seed)

    # ------ Initial evaluation ------ 
    initial_reward = evaluate(lora_path, eval_type, dataset, gpu_id, seed=seed)
    log_string = f"Initial evaluation reward: {initial_reward:.4f}"
    if eval_starting_test:
        initial_test_accuracy = evaluate_test(lora_path, eval_type, dataset, gpu_id, seed=seed)
        log_string += f"Initial test accuracy: {initial_test_accuracy:.4f}"
        wandb.log({"es_initial_test_accuracy": float(initial_test_accuracy)})

    log_with_flush(log_string)
    print(log_string)
    wandb.log({"es_initial_evaluation_reward": float(initial_reward)})
    # -------------------------------------

    # Load the model
    lora_sd = load_file(os.path.join(lora_path, "adapter_model.safetensors"), device="cpu")
    # Track best model
    best_iteration_reward = initial_reward 
    best_iteration_index = 0
    best_lora_sd = {
    k: v.detach().cpu().clone()
    for k, v in lora_sd.items()
    }

    # Record total training start time
    training_start_time = time.time()
    for iteration in tqdm(range(NUM_ITERATIONS)):
        # Record iteration start time
        iter_start_time = time.time()

        # Force garbage collection
        force_memory_cleanup()
        log_string = f"Starting iteration {iteration + 1}/{NUM_ITERATIONS}"
        log_with_flush(log_string)
        print(log_string)
        # ======= ES REWARD COLLECTION =======
        # Generate seeds on main process only
        if accelerator.is_main_process:
            seeds = np.random.randint(0, 2**30, size=POPULATION_SIZE, dtype=np.int64).tolist()
            seeds_tensor = torch.tensor(seeds, device=accelerator.device)
        else:
            seeds_tensor = torch.zeros(POPULATION_SIZE, dtype=torch.long, device=accelerator.device)

        # Broadcast seeds from main process to all processes
        if accelerator.num_processes > 1:
            torch.distributed.broadcast(seeds_tensor, src=0)
        seeds = seeds_tensor.cpu().tolist()  # Convert back to list for all processes

        # Assign seeds to each process for processing
        local_seeds = []
        for seed_idx, seed in enumerate(seeds):
            # Simple task assignment: assign seeds by process ID
            if seed_idx % accelerator.num_processes == accelerator.process_index:
                local_seeds.append((seed_idx, seed))

        # Process seeds in smaller batches to reduce memory pressure
        local_rewards = []
        batch_size = max(1, min(gpu_threads, len(local_seeds)))

        for batch_start in range(0, len(local_seeds), batch_size):
            batch_end = min(batch_start + batch_size, len(local_seeds))
            batch_seeds = local_seeds[batch_start:batch_end]

            with ThreadPoolExecutor(max_workers=len(batch_seeds)) as executor:
                # Prepare thread arguments
                thread_args = []
                for thread_id, (seed_idx, seed) in enumerate(batch_seeds):
                    thread_args.append((seed_idx, seed, SIGMA, lora_list[thread_id], eval_type, dataset, gpu_id, accelerator, thread_id, verbose))

                # Execute in parallel and collect results
                results = list(executor.map(process_seed, thread_args))
                local_rewards.extend(results)

            # Clean up between batches
            force_memory_cleanup()

        # Collect rewards from all processes
        all_rewards = torch.zeros(POPULATION_SIZE, device=accelerator.device)

        # Fill in locally computed rewards
        for seed_idx, reward in local_rewards:
            all_rewards[seed_idx] = reward

        # Aggregate rewards from all processes (each process will get the full reward list)
        if accelerator.num_processes > 1:
            torch.distributed.all_reduce(all_rewards, op=torch.distributed.ReduceOp.SUM)

        # Convert aggregated rewards back to Python list
        rewards = all_rewards.cpu().tolist()
        # Clean up no longer needed tensor
        del all_rewards
        force_memory_cleanup()

        # ======= ES PARAMETER UPDATE =======

        # Convert rewards to numpy and normalize
        rewards_np = np.array(rewards, dtype=np.float32)
        rewards_norm = (rewards_np - rewards_np.mean()) / (rewards_np.std() + 1e-8)

        # --- Update LoRA parameters ---
        param_names = list(lora_sd.keys())

        for name in param_names:
            param = lora_sd[name].to(accelerator.device)
            update = torch.zeros_like(param)
            gen = torch.Generator(device=param.device)

            # accumulate ES updates
            for seed_idx in range(POPULATION_SIZE):
                gen.manual_seed(int(seeds[seed_idx]))
                noise = torch.randn(param.shape, generator=gen, device=param.device)
                update.add_(noise * float(rewards_norm[seed_idx]))

            update /= POPULATION_SIZE
            lora_sd[name] = (param + ALPHA * update).cpu()

        # --- Save updated model (main process only) ---
        if accelerator.is_main_process:
            save_file(lora_sd, os.path.join(lora_path, "adapter_model.safetensors"))
            torch.cuda.synchronize()

        accelerator.wait_for_everyone()

        # Light cleanup
        force_memory_cleanup()

        iter_time = time.time() - iter_start_time

        mean_reward = rewards_np.mean().item()
        min_reward = rewards_np.min().item()
        max_reward = rewards_np.max().item()

        del rewards_np, rewards_norm
        force_memory_cleanup()
        # --- Evaluate updated model (end of iteration) ---
        if accelerator.is_main_process:

            iter_reward = evaluate(
                lora_path, eval_type, dataset, gpu_id, seed=seed
            )

            if iter_reward > best_iteration_reward:
                best_iteration_reward = iter_reward
                best_iteration_index = iteration + 1
                best_lora_sd = {
                    k: v.detach().cpu().clone()
                    for k, v in lora_sd.items()
                }

            log_string = (
                f"Iteration {iteration + 1}/{NUM_ITERATIONS} | "
                f"Reward: {iter_reward:.4f} | "
                f"Best: {best_iteration_reward:.4f}" 
                f"(Iteration: {best_iteration_index}) | "
                f"Mean Reward: {mean_reward:.4f} | "
                f"Max Reward: {max_reward:.4f} | "
                f"Min Reward: {min_reward:.4f} | "
                f"Time: {iter_time:.2f}s"
            )
            log_with_flush(log_string)

            if verbose:
                print(log_string)

            wandb_log = {
                "es_iteration": int(iteration + 1),
                "es_mean_reward": float(mean_reward),
                "es_max_reward": float(max_reward),
                "es_min_reward": float(min_reward),
                "es_iter_end_reward": float(iter_reward),
                "es_best_iteration": int(best_iteration_index),
                "es_best_iteration_reward": float(best_iteration_reward)
            }
            wandb.log(wandb_log)
            
          # print(f"GPU Memory: {torch.cuda.memory_allocated() / 1024**2:.2f}MB allocated, {torch.cuda.max_memory_allocated() / 1024**2:.2f}MB peak")

    total_time = time.time() - training_start_time
    
    # ======= FINAL EVALUATION =======
    if accelerator.is_main_process:

        total_time = time.time() - training_start_time
        log_string = f"Training completed in {total_time:.2f}s ({total_time/60:.2f} min)"
        log_with_flush(log_string)

        if verbose:
            print(log_string)

        # Restore best model before final evaluation
        lora_sd = best_lora_sd

        save_file(
            lora_sd,
            os.path.join(lora_path, "adapter_model.safetensors")
        )
        # --- Final evaluations ---
        final_reward = evaluate(lora_path, eval_type, dataset, gpu_id, seed=seed)
        ending_test_accuracy = evaluate_test(lora_path, eval_type, dataset, gpu_id, seed=seed)

        assert np.isclose(
            final_reward,
            best_iteration_reward,
            atol=1e-6
        ) , "Final reward does not match best iteration reward!"

        # --- WandB logging ---
        wandb.log({
            "es_final_evaluation_reward": float(final_reward),
            "es_ending_test_accuracy": float(ending_test_accuracy),
        })

        # --- Final summary log ---
        log_string = (
            f"Initial reward:          {initial_reward:.4f}\n"
            f"Final reward:            {final_reward:.4f}\n"
            f"Ending test accuracy:    {ending_test_accuracy:.4f}\n"
        )
        if eval_starting_test:
            log_string+= f"Initial test accuracy:  {initial_test_accuracy:.4f}"
        log_with_flush(log_string)

        if verbose:
            print(log_string)

    return final_reward, lora_path

