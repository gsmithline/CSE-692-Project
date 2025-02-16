import os
import sys
import argparse
import json
import time

import sys
import pandas as pd
#sys.path.append('../')
sys.path.append('../caif_negotiation/')
import concurrent.futures
pathology_results = pd.DataFrame()  
envy_results_history = {}
from utils.negotiation_game import run_game


def main():
    parser = argparse.ArgumentParser(description="Script to run negotiation experiments with concurrency.")
    parser.add_argument("--prompt_style", type=str, required=False, default="o3_mini_vs_4o",
                        help="Prompt style identifier, e.g., 'o3_mini_vs_4o'")
    parser.add_argument("--llm_model_p1", type=str, required=False, default="openai_4o",
                        help="LLM model identifier, e.g., 'openai_4o'")
    parser.add_argument("--llm_model_p2", type=str, required=False, default="openai_o3_mini",
                        help="LLM model identifier, e.g., 'openai_o3_mini'")
    parser.add_argument("--date", type=str, required=False, default="2_10_2025",
                        help="Date string for output naming, e.g., '2_10_2025'")
    parser.add_argument("--max_rounds", type=int, required=False, default=3,
                        help="Maximum number of negotiation rounds.")
    parser.add_argument("--games", type=int, required=False, default=10,
                        help="Number of games (simulations) to run per circle.")
    parser.add_argument("--circles", type=int, nargs='+', default=[0, 1, 2, 3, 4, 5, 6],
                        help="List of integer circle values to iterate over.")
    parser.add_argument("--parallel", type=bool, required=False, default=False,
                        help="Whether to run the experiments in parallel.")
    args = parser.parse_args()

    print("[INFO] Starting experiment...")
    print(f"  prompt_style: {args.prompt_style}")
    print(f"  llm_model_p1:     {args.llm_model_p1}")
    print(f"  llm_model_p2:     {args.llm_model_p2}")
    print(f"  date:         {args.date}")
    print(f"  max_rounds:   {args.max_rounds}")
    print(f"  games:        {args.games}")
    print(f"  circles:      {args.circles}")
    print("--------------------------------------------------")
    #checking if the models are valid
    valid_models = ["openai_4o", "openai_o3_mini", "anthropic_3.5_sonnet", "gemini_2.0_flash", "llama3.3-70b", "llama3.3-8b", "llama3.3-4050"]
    if args.llm_model_p1 not in valid_models:
        raise ValueError(f"Invalid model: {args.llm_model_p1}")
    if args.llm_model_p2 not in valid_models:
        raise ValueError(f"Invalid model: {args.llm_model_p2}")

    if args.parallel:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            future_to_circle = {
                executor.submit(
                    run_game, 
                    circle, 
                    args.games, 
                    args.max_rounds, 
                    args.date, 
                    args.prompt_style, 
                    args.llm_model_p1,
                    args.llm_model_p2
                ): circle
                for circle in args.circles
            }
            for future in concurrent.futures.as_completed(future_to_circle):
                circle_val = future_to_circle[future]
                try:
                    future.result()
                    print(f"[INFO] circle={circle_val} run finished successfully.")
                except Exception as exc:
                    print(f"[ERROR] circle={circle_val} generated an exception: {exc}")

        print("[INFO] All experiment runs completed.")
    else:
        for circle in args.circles:
            run_game(circle, args.games, args.max_rounds, args.date, args.prompt_style, args.llm_model_p1, args.llm_model_p2)


if __name__ == "__main__":
    main()