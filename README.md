# Model Swarms

Repository for [Model Swarms: Collaborative Search to Adapt LLM Experts via Swarm Intelligence](https://arxiv.org/abs/2410.11163). A fork of [link](https://github.com/google-research/google-research/tree/master/model_swarm) but more frequently updated here.

## Quick Start

#### Initialization

Create a conda environment for Model Swarms.
```
conda env create -f swarm.yml
conda activate swarm
```

Log into huggingface (for gemma access).
```
huggingface-cli login
```

Download initial experts.
```
cd initial_experts
python initial_experts.py
cd ..
```

#### Execute your first Model Swarms search.

Let's run Model Swarms on the [NLGraph](https://arxiv.org/abs/2305.10037) dataset focusing on LLM graph reasoning. `search_nlgraph.sh` provides the starter script for this.

Before running, how many GPUs do you have (and what are the GPU ids?). Change `-g` in line 23 of `search_nlgraph.sh`: by default five GPUs with ids `0,1,2,3,4`, but you can change to `0`, `0,1`, `0,2,4,5,6` or any combination you'd like.

Run it!
```
bash search_nlgraph.sh
```

You might be prompted to log into WandB and it is highly recommended. There will be a directory in `search/` that starts with `nlgraph_...`, all the logs, models, and results will be stored there. You can check out `search/nlgraph_.../log.txt` to see current progress.

[My GPUs are not so powerful and it runs kind of slow?] change `--populate_initial_experts` from 1 to 0, change `--initial_experts_num` from 20 to 10, add `--dropK 0.5`, `--dropN 0.5`, `-p 5`, `-m 20` together for a toned-down version.

End of search: the best-found model will be in `search/nlgraph_.../global_best` and performance metrics will be in `search/nlgraph_.../utility_scratchpad.json`. The json file presents the change in utility values in the search process: for global best, global worst, and individual models. The `log.txt` will contain in the end several overall metrics. Watch for `starting_best_validation_utility`, the best initial expert's utility value on the dev set; `starting_best_single_test_accuracy`, this initial expert's utility on the test set; `ending_best_validation_utility`, the ending global best's utility on the dev set; `ending_best_single_test_accuracy`, the ending global best's utility on the test set.

## Other Objectives

For Objective 1: single task, follow `search_nlgraph.sh` and change `-e` and `-d` to combinations of `(multiple_choice, mmlu)`, `(multiple_choice, mmlu_pro)`, `(multiple_choice, knowledge_crosswords)`, `(multiple_choice, hellaswag)`, `(exact_match, gsm8k)`, `(exact_match, nlgraph)`, `(multiple_choice, truthfulqa)`, `(external_api, realtoxicityprompts)`, `(AbstainQA, mmlu)`. Make sure to change `-n` to include the dataset name in the search directory name. For RealToxicityPrompts you will be prompted to setup Perspective API through google cloud.

For Objective 2: multi-task domains, follow `search_legal.sh` and change `-d` to `legal`, `medical`, `science`, and `culture`. Make sure to change `-n` as well.

For Objective 3: reward models, follow `search_concise.sh` and change `-e` to `rm_default`, `rm_verbose`, and `rm_concise`. Make sure to change `-n` as well.

For Objective 4: human interests, follow `search_phd_application.sh` and change line 8 `domain_list` into the `human_<name>.json` in `data/eval/`. Make sure to change `-n` as well. You will be prompted to setup Vertex AI API to access Gemini APIs through google cloud.

Adding your data: just follow the format of existing data files in `data/eval/` and then change `-n` and use a respective `-e`. For example, for multiple-choice, follow the format of `data/eval/hellaswag.json` and use `-e` as `multiple_choice`; for answer match, follow the format of `data/eval/nlgraph.json` and use `-e` as `exact_match`; for optimizing reward model scores (objective 3 or 4), follow `data/eval/rm.json` and use `-e` as either `rm_default` for a local reward model or `human` for gemini-as-a-judge.

Adding your models: look at `initial_experts/`: it is essentially a folder of 10 LoRA adapters of Gemma-7B. Create your own folder of models with the same architecture like `initial_experts` and change add the argument `-i <path_to_folder>` (see search.py). If the models are adapters/only have 1 shard, you don't need to change anything else. If they are full models/multiple shards, change `--fast_merge` from `1` to `0`.

Adding your evaluation: well that will be a bit more workload. Go to `evaluate.py`: there's a `evaluate()` for valiation set and `evaluate_test()` for test set (intentionally kept separate), essentially you specify path to a model and these functions give you a scalar score. In both of them there are `if eval_type == ...` clauses: name your evaluation type, open a new if clause `if eval_type == <your_eval>`, implement how you load the data and get a scalar score. You need to do it for both `evaluate()` and `evaluate_test()`. `batch_generate()` and `batch_generate_chat_template()` provide two helper functions to generate text from a model and please use them for model output. You could refer to `eval_type == multiple_choice` or `eval_type == exact_match` as examples.

## Changing Hyperparameters and Settings

Do `python search.py -h` to see a list of all possible hyperparameters and settings. Additionally look at the comments for hyperparameters in `search.py`. We already included the default settings in the four `search_<name>.sh` starter scripts, but feel free to play around different settings.

## Adpated Models

We provide best-found model checkpoints for the objectives in `link pending`.

## Citation

If Model Swarms is helpful to you:

```
@article{feng2024model,
  title={Model Swarms: Collaborative Search to Adapt LLM Experts via Swarm Intelligence},
  author={Feng, Shangbin and Wang, Zifeng and Wang, Yike and Ebrahimi, Sayna and Palangi, Hamid and Miculicich, Lesly and Kulshrestha, Achin and Rauschmayr, Nathalie and Choi, Yejin and Tsvetkov, Yulia and others},
  journal={arXiv preprint arXiv:2410.11163},
  year={2024}
}
```


# es-fine-tuning-paper
This repo contains the source code for the paper "Evolution Strategies at Scale: LLM Fine-Tuning Beyond Reinforcement Learning" (https://arxiv.org/abs/2509.24372). Evolution strategies (ES) is used to directly optimize billions of parameters of large language models (LLMs).

Feel free to join the ES fine-tuning forum in [Discussions](https://github.com/VsonicV/es-fine-tuning-paper/discussions).

### News
10/27/2025: :fire::fire::fire: An accelerated version with **10X+ speed-up** in running time is added to the repo!  See [here](https://github.com/VsonicV/es-fine-tuning-paper?tab=readme-ov-file#accelerated-version-10x-speed-up). :rocket::rocket::rocket:

Note: we are still actively adding more experimental codes into this repo. We expect breaking change to the accelerated implementations.

## Setup
Create a virtual environment with python version >= 3.10 and activate it
```bash
python -m venv es
source es/bin/activate
```

From the root of the repository run following command to install all the relevant python packages
```bash
pip install -r requirement.txt
```


## Usage
For running the main ES code on conciseness fine-tuning

```bash
accelerate launch \
    --num_processes 2 \
    --num_machines 1 \
    --machine_rank 0 \
    es_fine-tuning_conciseness.py \
    --gpu_threads=1 \
    --model_name=Qwen/Qwen2.5-7B-Instruct
```

`--num_processes` specifies the number of GPUs to use and `--gpu_threads` specifies the number of threads inside each GPU. The total number of parallel evaluations is thereby equal to `num_processes`*`gpu_threads`.

For running the main ES code on the Countdown task
```bash
accelerate launch \
    --num_processes 4 \
    --num_machines 1\
    --machine_rank 0 \
    countdown/es_fine-tuning_countdown.py \
    --data_sample 200 \
    --model_name Qwen/Qwen2.5-3B-Instruct \
    --gpu_threads 1
```


### Other Parameters

- `--gpu_ids`: Specify which GPUs to use (CUDA device id), argument for `accelerate launch`
- `--model_name`: HuggingFace model to fine-tune
- `--hf_cache_dir`: Directory for HuggingFace cache
- `--precision`: Model precision, default to be `bf16`
- `--verbose`: Enable detailed logging if this argument is present in the command line

Note: The original implementation uses a partially correlated noise. To use complete i.i.d. noise, please use `es_fine-tuning_conciseness_iid.py` and `countdown/es_fine-tuning_countdown_iid.py` instead. See [here](https://github.com/VsonicV/es-fine-tuning-paper/discussions/7) for more details.

## Accelerated Version (10X+ Speed-up)

If you are using the latest accelerated version `es-fine-tuning_countdown_accl.py`, please also install the `vllm` and `tensorboard` by:
```bash
pip install vllm==0.11.0
pip install tensorboard
```

For running the accelerated version on the Countdown task:
```bash
# Single-GPU quickstart
python es_fine-tuning_countdown_accl.py \
  --model_name Qwen/Qwen2.5-3B-Instruct \
  --cuda_devices 0 \
  --num_engines 1 \
  --population_size 30 \
  --num_iterations 1000

# Multi-GPU run (one vLLM engine per GPU)
python es_fine-tuning_countdown_accl.py \
  --model_name Qwen/Qwen2.5-3B-Instruct \
  --cuda_devices 0,1,2,3 \
  --num_engines 4 \
  --population_size 30 \
  --num_iterations 1000 \
  --sigma 0.001 \
  --alpha 0.0005 \
  --experiment_dir es-ft-experiment
```

On preliminary 4xH100 setting, accelerated version achieves ~10 times speed-up with similar convergence rate.

## Citation

If you find this work helpful in your research, please cite:

```bibtex
@misc{qiu2025evolutionstrategiesscalellm,
      title={Evolution Strategies at Scale: LLM Fine-Tuning Beyond Reinforcement Learning}, 
      author={Xin Qiu and Yulu Gan and Conor F. Hayes and Qiyao Liang and Elliot Meyerson and Babak Hodjat and Risto Miikkulainen},
      year={2025},
      eprint={2509.24372},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2509.24372}, 
}
```
