import os
from transformers import AutoModelForCausalLM

global CACHE_DIR

for model_name in ["code_alpaca", "cot", "flan_v2", "gemini_alpaca", "lima", "oasst1", "open_orca", "science", "sharegpt", "wizardlm"]:
    if os.path.exists(model_name):
        continue
    model = AutoModelForCausalLM.from_pretrained("bunsenfeng/"+model_name, cache_dir=CACHE_DIR)
    model.save_pretrained(model_name)