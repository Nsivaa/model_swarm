
def es_optimize(model, lora_name, pop_size = 30, sigma = 0.001, lr = 5e-4, seed = 42):
    """
    
    Generate a new LoRA adapter by applying Evolution Strategies.
    It perturbs the LoRA adapter weights with Gaussian noise, then merges it to the base model, evaluates it
    and updates the original adapter weights based on the performance.

    Args:
        model: The base model to apply the LoRA adapter to.
        lora_name (str): Path to the LoRA adapter.
        n (int): Number of perturbations.
        sigma (float): Standard deviation of the Gaussian noise.
        alpha (float): Learning rate for updating the adapter.
    Returns:
        new_lora_state_dict: The state dict of the new LoRA adapter.
    """

    # seed everything
    torch.manual_seed(seed)
    random.seed(seed)

    # load the original LoRA adapter
    original_state_dict = load_file(os.path.join(lora_name, "adapter_model.safetensors"), device="cpu")
    perturbation_seeds = []
    # generate population of perturbed adapters
    for i in range(pop_size):
        pert_seed = random.randint(0, 1e6)
        perturbation_seeds.append(pert_seed)
        perturbed_state_dicts = []
        # seed the perturbation
        random.seed(pert_seed)
        for key in original_state_dict.keys():
            noise = torch.randn_like(original_state_dict[key]) * sigma
            perturbed_state_dicts.append(original_state_dict[key] + noise)
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