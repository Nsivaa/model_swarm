import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging
import numpy as np
import copy
import os
import argparse
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


logging.set_verbosity_error()
torch.backends.cuda.matmul.allow_tf32 = True

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='google/gemma-7b-it')
parser.add_argument('--hf_cache_dir', type=str, default='/scratch/a.dicembre/.hf_cache')
parser.add_argument('--precision', type=str, default='bf16')
parser.add_argument('--gpu_threads', type=int, default=4, help='Number of parallel threads per GPU')
parser.add_argument('--verbose', action='store_true', help='Print verbose logs')
parser.add_argument('--num_iterations', type=int, default=10, help='Number of ES iterations')
parser.add_argument('--population_size', type=int, default=30, help='Population size for ES')
parser.add_argument('--sigma', type=float, default=0.001, help='Standard deviation for weight perturbations')
parser.add_argument('--alpha', type=float, default=0.0005, help='Learning rate for ES')
parser.add_argument('--max_new_tokens', type=int, default=100, help='Maximum number of tokens to generate')
parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
args = parser.parse_args()


# Hyperparameters for ES
NUM_ITERATIONS = args.num_iterations             # Number of ES iterations (generations)
POPULATION_SIZE = args.population_size              # Population size (number of perturbations per iteration)
SIGMA = args.sigma                     # Standard deviation for weight perturbations (noise scale)
ALPHA = args.alpha                  # Learning rate
max_new_tokens = args.max_new_tokens              # Maximum number of tokens allowed to be generated
do_sample = False                 # Whether sampling is allowed in generating tokens, default to be not allowed (greedy decoding for ES)
initial_seed = args.seed                 # Initial random seed


# --- Dummy Dataset and Reward Function ---
# In practice, define a set of input reasoning tasks with desired targets.
dataset = [
    ("Solve: 3 + 5 =", "8"),
    ("If all birds can fly and penguins are birds, can penguins fly?", "No"),
]

def compute_reward(generated_text, target_text):
    # Negative absolute difference in length
    return -abs(len(generated_text) - len(target_text))

def force_memory_cleanup():
    """Force aggressive memory cleanup"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()


def process_seed(seed_args):
    """Function to process a single seed, used for thread pool"""
    seed_idx, seed, lora_path, accelerator, thread_id, verbose = seed_args

    if verbose:
        print(f"Process {accelerator.process_index} Thread {thread_id} processing seed {seed_idx} (value: {seed} lora_path: {lora_path})")

    sd_original = load_file(os.path.join(lora_path, "adapter_model.safetensors"), device="cpu")
    
    # copy LoRA weights for this particle
    sd = {k: v.clone() for k, v in sd_original.items()}

    gen = torch.Generator().manual_seed(int(seed))
    for name in sd:
        noise = torch.randn(sd[name].shape, generator=gen, dtype=sd[name].dtype)
        sd[name] += SIGMA * noise

    # Save as temporary LoRA
    tmp_path = f"/tmp/particle_{thread_id}_{seed_idx}.safetensors"
    save_file(sd, tmp_path)

    # Ensure weights are fully loaded before evaluation
    if torch.cuda.is_available():
        torch.cuda.synchronize(accelerator.device)
    
    reward = evaluate(tmp_path, eval_type, dataset, gpu_id, seed)

    try:
        os.remove(tmp_path)
    except FileNotFoundError:
        logging.warning(f"Temporary file {tmp_path} not found for removal.")

    if torch.cuda.is_available():
        torch.cuda.synchronize(accelerator.device)
    force_memory_cleanup()

    if verbose:
        print(f"Process {accelerator.process_index} Thread {thread_id} completed seed {seed_idx} with reward {reward:.4f}, lora_path: {lora_path}")

    return seed_idx, reward


# --- Main Evolution Strategies Loop ---
def main():
    accelerator = Accelerator()

    if accelerator.is_main_process:
        print(f"Total processes: {accelerator.num_processes}, GPU threads per process: {args.gpu_threads}")
        print(f"Population size: {POPULATION_SIZE}, Iterations: {NUM_ITERATIONS}")
        print(f"Sigma: {SIGMA}, Alpha: {ALPHA}")

    # Load model
    model_name = args.model_name
    hf_cache_dir = args.hf_cache_dir

    if accelerator.is_main_process:
        print(f"Loading model {model_name}...")
    

    # Load model
    model_list = []
    for model_index in range(args.gpu_threads):
        model_list.append(AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=hf_cache_dir,
            device_map={"": accelerator.process_index},  # Assign devices explicitly
            torch_dtype=torch.float16 if args.precision == 'fp16' else (torch.bfloat16 if args.precision == 'bf16' else torch.float32),
        ))
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, cache_dir=hf_cache_dir)

    if accelerator.is_main_process:
        print("Model loaded successfully")

    # Prepare model with accelerator
    for model in model_list:
        model.eval()  # Turn off dropout, etc.

    force_memory_cleanup()

    # Record total training start time
    training_start_time = time.time()

    np.random.seed(initial_seed)

    for iteration in tqdm(range(NUM_ITERATIONS)):
        # Record iteration start time
        iter_start_time = time.time()

        # Force garbage collection
        force_memory_cleanup()

        if args.verbose:
            print(f"Process {accelerator.process_index} starting iteration {iteration + 1}/{NUM_ITERATIONS}")

        # Generate seeds on main process only
        if accelerator.is_main_process:
            if args.verbose:
                print(f"Main process {accelerator.process_index} generating seeds")
            seeds = np.random.randint(0, 2**30, size=POPULATION_SIZE, dtype=np.int64).tolist()
            seeds_tensor = torch.tensor(seeds, device=accelerator.device)
        else:
            if args.verbose:
                print(f"Worker process {accelerator.process_index} waiting for seeds")
            seeds_tensor = torch.zeros(POPULATION_SIZE, dtype=torch.long, device=accelerator.device)

        # Broadcast seeds from main process to all processes
        if accelerator.num_processes>1:
            torch.distributed.broadcast(seeds_tensor, src=0)
        seeds = seeds_tensor.cpu().tolist()  # Convert back to list for all processes

        if args.verbose:
            print(f"Process {accelerator.process_index} received seeds")

        # Assign seeds to each process for processing
        local_seeds = []
        for seed_idx, seed in enumerate(seeds):
            # Simple task assignment: assign seeds by process ID
            if seed_idx % accelerator.num_processes == accelerator.process_index:
                local_seeds.append((seed_idx, seed))

        if args.verbose:
            print(f"Process {accelerator.process_index} assigned {len(local_seeds)} seeds: {[idx for idx, _ in local_seeds]}")

        # Process seeds in smaller batches to reduce memory pressure
        local_rewards = []
        batch_size = max(1, min(args.gpu_threads, len(local_seeds)))

        for batch_start in range(0, len(local_seeds), batch_size):
            batch_end = min(batch_start + batch_size, len(local_seeds))
            batch_seeds = local_seeds[batch_start:batch_end]

            with ThreadPoolExecutor(max_workers=len(batch_seeds)) as executor:
                # Prepare thread arguments
                thread_args = []
                for thread_id, (seed_idx, seed) in enumerate(batch_seeds):
                    # Pass verbose flag as argument to process_seed function
                    thread_args.append((seed_idx, seed, model_list[thread_id], tokenizer, accelerator, thread_id, args.verbose))

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
        if accelerator.num_processes>1:
            torch.distributed.all_reduce(all_rewards, op=torch.distributed.ReduceOp.SUM)

        # Convert aggregated rewards back to Python list
        rewards = all_rewards.cpu().tolist()
        # Clean up no longer needed tensor
        del all_rewards
        force_memory_cleanup()

        # Convert rewards to a tensor and normalize.
        rewards_tensor = np.array(rewards, dtype=np.float32)
        rewards_normalized = (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std() + 1e-8)

        # Aggregate perturbations and update model weights
        if args.verbose:
            print(f"Process {accelerator.process_index} updating model weights")
        original_model = model_list[0]
        for name, param in tqdm(original_model.named_parameters()):
            gen = torch.Generator(device=param.device)
            update = torch.zeros_like(param)
            for seed_idx in range(POPULATION_SIZE):
                r_norm = rewards_normalized[seed_idx]
                seed = seeds[seed_idx]
                gen.manual_seed(int(seed))

                noise = torch.randn(
                    param.shape,
                    generator=gen,
                    device=param.device,
                    dtype=param.dtype
                )
                noise.mul_(float(r_norm))
                update.add_(noise)
                del noise
            update.div_(POPULATION_SIZE)
            param.data.add_(ALPHA * update)
            torch.cuda.empty_cache()

        for model_idx in range(1, len(model_list)):
            original_model_tmp = model_list[model_idx]
            for name, param in original_model_tmp.named_parameters():
                param.data.copy_(original_model.get_parameter(name).data.clone())

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
            print(f"Iteration {iteration + 1}/{NUM_ITERATIONS}, Time: {iter_time:.2f}s, Mean: {mean_reward:.2f}, Min: {min_reward:.2f}, Max: {max_reward:.2f}")
            print(f"GPU Memory: {torch.cuda.memory_allocated() / 1024**2:.2f}MB allocated, {torch.cuda.max_memory_allocated() / 1024**2:.2f}MB peak")

    total_time = time.time() - training_start_time


    # Save the fine-tuned model weights.
    if accelerator.is_main_process:
        print(f"Training completed in {total_time:.2f}s ({total_time/60:.2f} minutes)")
        question_num = len(dataset)
        save_dir = f"finetuned_{model_name}_es_random_seed{initial_seed}_pop{POPULATION_SIZE}_iter{NUM_ITERATIONS}_sigma{SIGMA}_alpha{ALPHA}_{args.precision}_threads{args.gpu_threads}_question_num{question_num}_correct"
        print(f"Saving model to {save_dir}...")
        original_model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        print(f"Model saved successfully.")

if __name__ == "__main__":
    os.environ["PYTHONWARNINGS"] = "ignore"
    mp.set_start_method('spawn', force=True)
    main()
