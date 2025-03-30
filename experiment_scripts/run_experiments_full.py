import os
import sys
import argparse
import json
import time
sys.path.append('../caif_negotiation/')
import pandas as pd
import concurrent.futures
from utils.negotiation_game import run_game

def main():
    parser = argparse.ArgumentParser(description="Run negotiation experiments for a specific model against all models in game matrix.")
    parser.add_argument("--model", type=str, required=True,
                        help="Base model name, e.g., 'anthropic_sonnet_3.7_reasoning_2025-02-19'")
    parser.add_argument("--circle", type=int, required=True,
                        help="Circle value for the model")
    parser.add_argument("--date", type=str, required=False, default=time.strftime("%m_%d_%Y"),
                        help="Date string for output naming")
    parser.add_argument("--max_rounds", type=int, required=False, default=3,
                        help="Maximum number of negotiation rounds")
    parser.add_argument("--games", type=int, required=False, default=20,
                        help="Number of games to run per pairing")
    parser.add_argument("--parallel", type=bool, required=False, default=True,
                        help="Run experiments in parallel")
    parser.add_argument("--discount", type=float, required=False, default=.98, 
                        help="Discount rate for game (between 0 and 1)")
    args = parser.parse_args()

    # Define valid model-circle combinations from the performance matrix
    model_circles = {
        "anthropic_3.7_sonnet_2025-02-19": [5, 6],
        "anthropic_sonnet_3.7_reasoning_2025-02-19": [0],
        "gemini_2.0_flash": [2, 5],
        "openai_4o_2024-08-06": [4, 5, 6],
        "openai_o3_mini_2025-01-31": [0]
    }

    if args.model not in model_circles:
        raise ValueError(f"Invalid model: {args.model}. Valid models are: {list(model_circles.keys())}")
    
    if args.circle not in model_circles[args.model]:
        raise ValueError(f"Circle {args.circle} is not valid for model {args.model}. Valid circles are: {model_circles[args.model]}")

    # Build the list of all valid model-circle combinations
    all_combinations = []
    for model_name, circles in model_circles.items():
        for circle in circles:
            all_combinations.append((model_name, circle))

    # Create list of experiments to run
    experiments = []
    
    # The input model as player 1 against all others as player 2
    for opponent_model, opponent_circle in all_combinations:
        prompt_style = f"{args.model}_circle_{args.circle}_vs_{opponent_model}_circle_{opponent_circle}"
        experiments.append({
            "p1_model": args.model,
            "p1_circle": args.circle,
            "p2_model": opponent_model,
            "p2_circle": opponent_circle,
            "prompt_style": prompt_style
        })
    
    # The input model as player 2 against all others as player 1
    for opponent_model, opponent_circle in all_combinations:
        prompt_style = f"{opponent_model}_circle_{opponent_circle}_vs_{args.model}_circle_{args.circle}"
        experiments.append({
            "p1_model": opponent_model,
            "p1_circle": opponent_circle,
            "p2_model": args.model,
            "p2_circle": args.circle,
            "prompt_style": prompt_style
        })

    print(f"[INFO] Starting experiments for {args.model} with circle {args.circle}")
    print(f"  Total experiments to run: {len(experiments)}")
    print(f"  Date: {args.date}")
    print(f"  Max rounds: {args.max_rounds}")
    print(f"  Games per pairing: {args.games}")
    print(f"  Discount: {args.discount}")
    print("--------------------------------------------------")

    # Run the experiments
    if args.parallel:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = []
            for exp in experiments:
                futures.append(executor.submit(
                    run_game,
                    exp["p1_circle"],
                    exp["p2_circle"],
                    args.games,
                    args.max_rounds,
                    args.date,
                    exp["prompt_style"],
                    exp["p1_model"],
                    exp["p2_model"],
                    args.discount
                ))
            
            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                exp = experiments[i]
                try:
                    future.result()
                    print(f"[INFO] Completed: {exp['prompt_style']}")
                except Exception as exc:
                    print(f"[ERROR] {exp['prompt_style']} generated exception: {exc}")
    else:
        for exp in experiments:
            try:
                run_game(
                    exp["p1_circle"],
                    exp["p2_circle"],
                    args.games,
                    args.max_rounds,
                    args.date,
                    exp["prompt_style"],
                    exp["p1_model"],
                    exp["p2_model"],
                    args.discount
                )
                print(f"[INFO] Completed: {exp['prompt_style']}")
            except Exception as exc:
                print(f"[ERROR] {exp['prompt_style']} generated exception: {exc}")

    print("[INFO] All experiment runs completed.")

if __name__ == "__main__":
    main()