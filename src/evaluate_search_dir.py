# src/evaluate_search_dir.py
import argparse
from sklearn.metrics import accuracy_score
from evaluate import evaluate, evaluate_test
import os
from search import assign_gpu
import json

def evaluate_search_dir_metrics(
    search_pass_name,
    dataset,
    eval_type="multiple_choice",
    gpus=[0],
    base_model="google/gemma-7b-it",
    top_k=10,
    skip_ensemble=True,
    seed=42
):
    search_dir = os.path.join("search", search_pass_name)

    # infer number of particles (index-based, reseed-safe)
    num_particles = 0
    while os.path.isdir(os.path.join(search_dir, f"particle_{num_particles}")):
        num_particles += 1

    assert num_particles > 0, "No particles found"

    ending_utility = []

    # ---------------- DEV EVAL ----------------
    for i in range(num_particles):
        particle_now = os.path.join(search_dir, f"particle_{i}", "now")
        gpu_id = gpus[assign_gpu(len(gpus), i, num_particles)]

        dev_acc = evaluate(
            particle_now,
            eval_type,
            dataset,
            gpu_id,
            base_model,
            seed=seed
        )
        ending_utility.append(dev_acc)

    best_dev_idx = ending_utility.index(max(ending_utility))

    print("----- DEV METRICS -----")
    print(f"dataset: {dataset}")
    print("best particle on dev:", best_dev_idx)
    print("best dev accuracy:", ending_utility[best_dev_idx])

    # ---------------- TEST EVAL ----------------
    test_preds = []
    test_accs = []
    golds = None

    for i in range(num_particles):
        particle_now = os.path.join(search_dir, f"particle_{i}", "now")
        gpu_id = gpus[assign_gpu(len(gpus), i, num_particles)]

        test_acc = evaluate_test(
            particle_now,
            eval_type,
            dataset,
            gpu_id,
            base_model,
            seed=seed
        )
        test_accs.append(test_acc)

        with open(os.path.join(particle_now, "preds.json")) as f:
            test_preds.append(json.load(f))

        with open(os.path.join(particle_now, "golds.json")) as f:
            g = json.load(f)
            if golds is None:
                golds = g
            else:
                assert golds == g

    print("----- TEST METRICS -----")
    print(f"dataset: {dataset}")
    print(
        "best single test accuracy (best-dev particle):",
        test_accs[best_dev_idx]
    )

    if not skip_ensemble:
        final_test_preds = ensemble_based_on_utility(
            test_preds, ending_utility, top_k
        )
        print(
            "top-k ensemble test accuracy:",
            accuracy_score(golds, final_test_preds)
        )

    return {
        "ending_best_validation_utility": max(ending_utility),
        "ending_best_particle_on_validation": best_dev_idx,
        "ending_best_single_test_accuracy": test_accs[best_dev_idx],
        "ending_top-k_ensemble_test_accuracy":
            accuracy_score(golds, final_test_preds) if not skip_ensemble else None,
    }



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--search_pass_name", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--eval_type", type=str, default="multiple_choice")
    parser.add_argument("--gpus", type=str, default="0")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--skip_ensemble", action="store_true")
    parser.add_argument("--seed", type=int, default=None)

    args = parser.parse_args()

    gpus = [int(g) for g in args.gpus.split(",")]

    evaluate_search_dir_metrics(
        search_pass_name=args.search_pass_name,
        dataset=args.dataset,
        eval_type=args.eval_type,
        gpus=gpus,
        top_k=args.top_k,
        skip_ensemble=args.skip_ensemble,
        seed=args.seed,
    )

if __name__ == "__main__":
    main()
