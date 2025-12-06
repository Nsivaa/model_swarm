
def es_optimize(lora_name, pop_size = 30, sigma = 0.001, lr = 5e-4, seed = 42, eval_type, dataset, gpus):
    """
    
    Generate a new LoRA adapter by applying Evolution Strategies.
    It perturbs the LoRA adapter weights with Gaussian noise, then merges it to the base model, evaluates it
    and updates the original adapter weights based on the performance.

    Args:
        lora_name (str): Path to the LoRA adapter.
        pop_size (int): Population size.
        sigma (float): Standard deviation of the Gaussian noise.
        lr (float): Learning rate for updating the adapter.
        seed (int): Seed.

    Returns:
        new_lora_state_dict: The state dict of the new LoRA adapter.
    """

    # seed everything
    torch.manual_seed(seed)
    random.seed(seed)
    perturbation_seeds = []
    rewards = []
    # generate population of perturbed adapters
    for _ in range(pop_size):
        # load the original LoRA adapter
        state_dict = load_file(os.path.join(lora_name, "adapter_model.safetensors"), device="cpu")
        pert_seed = random.randint(0, 1e6)
        perturbation_seeds.append(pert_seed)
        # seed the perturbation
        random.seed(pert_seed)
        # apply the perturbation in place
        for v in state_dict.values():
            noise = torch.randn_like(v) * sigma
            v += noise
        # evaluate the perturbed adapter
        set_peft_model_state_dict(model, state_dict) # 
        eval_args = [lora_name, eval_type, dataset, gpu_id]
        pool = Pool(processes=len(gpus))
        reward = pool.starmap(evaluate, eval_args, chunksize=math.ceil(len(particle_paths)/len(gpus)))
        pool.close()
        pool.join()
        rewards.append(reward)
        # reload the original adapter for the next perturbation
        state_dict = load_file(os.path.join(lora_name, "adapter_model.safetensors"), device="cpu")

    for i in range(pop_size):
    # generate the optimized adapter applying the weights
    final_state_dict = {}
    for i in range(len(lora_state_dict_list)):
        if i == 0:
            for key in lora_state_dict_list[i].keys():
                final_state_dict[key] = weights[i] * lora_state_dict_list[i][key]
        else:
            for key in lora_state_dict_list[i].keys():
                assert key in final_state_dict.keys()
                final_state_dict[key] += weights[i] * lora_state_dict_list[i][key]
    
    if not os.path.exists(output_name):
        os.mkdir(output_name)
    save_file(final_state_dict, os.path.join(output_name, "adapter_model.safetensors"))

    return final_state_dict