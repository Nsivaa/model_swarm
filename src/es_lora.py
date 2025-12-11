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
from evaluate import evaluate
from peft import LoraConfig, PeftModel, get_peft_model
from safetensors.torch import load_file, save_file
import logging

torch.backends.cuda.matmul.allow_tf32 = True
os.environ["PYTHONWARNINGS"] = "ignore"
mp.set_start_method('spawn', force=True)


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

    if verbose:
        print(f"Process {accelerator.process_index} Thread {thread_id} processing seed {seed_idx} (value: {seed} lora_path: {lora_path})")

    sd_original = load_file(os.path.join(lora_path, "adapter_model.safetensors"), device="cpu")
    
    # copy LoRA weights for this particle
    sd = {k: v.clone() for k, v in sd_original.items()}

    gen = torch.Generator().manual_seed(int(seed))
    for name in sd:
        noise = torch.randn(sd[name].shape, generator=gen, dtype=sd[name].dtype)
        sd[name] += SIGMA * noise

    # Create temp directory for this particle
    tmp_dir = f"/tmp/particle_{thread_id}_{seed_idx}"
    os.makedirs(tmp_dir, exist_ok=True)

    # Save the mutated LoRA weights
    adapter_model_path = os.path.join(tmp_dir, "adapter_model.safetensors")
    save_file(sd, adapter_model_path)
    
    # assert that weights are actually changed
    sd_perturbed = load_file(adapter_model_path, device="cpu")
    weights_changed = any(not torch.equal(sd_original[k], sd_perturbed[k]) for k in sd_original)
    assert weights_changed, "Weights did not change after perturbation!" 

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
def es_lora(lora_path, eval_type, dataset, seed, search_pass_name, base_model = "google/gemma-7b-it", 
             POPULATION_SIZE=30, NUM_ITERATIONS=10, SIGMA=0.001, ALPHA=0.0005, 
             cache_dir='/scratch/a.dicembre/.hf_cache', gpu_id = 0, verbose=False, gpu_threads=1, overwrite_output_dir=False):

    # Configure logging to write to a file
    logging.basicConfig(filename=os.path.join("search", search_pass_name, "log.txt"), level=logging.DEBUG)
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

    # ------ Initial evaluation ------ DEBUG
    initial_reward = evaluate(lora_path, eval_type, dataset, gpu_id, seed=seed)
    print(f"Initial evaluation reward: {initial_reward:.4f}")
    # -------------------------------------

    # Record total training start time
    training_start_time = time.time()
    for iteration in tqdm(range(NUM_ITERATIONS)):
        # Record iteration start time
        iter_start_time = time.time()

        # Force garbage collection
        force_memory_cleanup()
        log_string = f"Starting iteration {iteration + 1}/{NUM_ITERATIONS}"
        log_with_flush(log_string)
        if verbose:
            print(log_string)

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

        # Convert rewards to a tensor and normalize.
        rewards_tensor = np.array(rewards, dtype=np.float32)
        rewards_normalized = (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std() + 1e-8)
        
        # update lora parameters
        lora_sd_original = load_file(os.path.join(lora_path, "adapter_model.safetensors"), device="cpu")
        lora_param_names = list(lora_sd_original.keys())
        for name in lora_param_names:
            param = lora_sd_original[name].to(accelerator.device)
            gen = torch.Generator(device=param.device)
            update = torch.zeros_like(param)
            for seed_idx in range(POPULATION_SIZE):
                r_norm = rewards_normalized[seed_idx]
                seed = seeds[seed_idx]
                gen.manual_seed(int(seed))
                noise = torch.randn(param.shape, generator=gen, device=param.device, dtype=param.dtype)
                noise.mul_(float(r_norm))
                update.add_(noise)
                del noise
            update.div_(POPULATION_SIZE)
            lora_sd_original[name] = (param + ALPHA * update).cpu()
            torch.cuda.empty_cache()

        # Synchronize to ensure weight updates are complete
        if torch.cuda.is_available():
            torch.cuda.synchronize(accelerator.device)

        force_memory_cleanup()

        iter_time = time.time() - iter_start_time

        mean_reward = rewards_tensor.mean().item()
        min_reward = rewards_tensor.min().item()
        max_reward = rewards_tensor.max().item()

        del rewards_tensor, rewards_normalized
        force_memory_cleanup()

        if accelerator.is_main_process:
            log_string = f"Iteration {iteration + 1}/{NUM_ITERATIONS} completed. Time: {iter_time:.2f}s, Mean Reward: {mean_reward:.4f}, Min Reward: {min_reward:.4f}, Max Reward: {max_reward:.4f}"
            log_with_flush(log_string)
            if verbose:
                print(log_string)
            
          # print(f"GPU Memory: {torch.cuda.memory_allocated() / 1024**2:.2f}MB allocated, {torch.cuda.max_memory_allocated() / 1024**2:.2f}MB peak")

    total_time = time.time() - training_start_time
    
    if verbose:
        print(log_string)

    # Save the fine-tuned model weights.
    if accelerator.is_main_process:
        log_string = f"Training completed in {total_time:.2f}s ({total_time/60:.2f} minutes)"
        log_with_flush(log_string)
        if verbose:
            print(log_string)

        if overwrite_output_dir and os.path.exists(lora_path):
            save_dir = lora_path
        else:
            # Parent directory of the input LoRA
            parent_dir = os.path.dirname(lora_path)
            folder_name = f"es_{POPULATION_SIZE}_{NUM_ITERATIONS}_{SIGMA}_{ALPHA}"
            save_dir = os.path.join(parent_dir, folder_name)

        os.makedirs(save_dir, exist_ok=True)

        # Save updated LoRA weights
        save_file(lora_sd_original, os.path.join(save_dir, "adapter_model.safetensors"))

        # Copy adapter_config.json so the LoRA stays loadable
        shutil.copy(
            os.path.join(lora_path, "adapter_config.json"),
            os.path.join(save_dir, "adapter_config.json")
        )

        if verbose:
            print(f"Saved updated LoRA to {save_dir}")

        # --- Final evaluation ---
        final_reward = evaluate(save_dir, eval_type, dataset, gpu_id, seed=seed)
        log_string = (f"Initial evaluation reward: {initial_reward:.4f}\n"
                    f"Final evaluation reward:   {final_reward:.4f}")
        log_with_flush(log_string)
        if verbose:
            print(log_string)

    return final_reward, save_dir
