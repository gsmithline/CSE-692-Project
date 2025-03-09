import os
import sys
import argparse
import json
import time
sys.path.append('../caif_negotiation/')
import sys
import pandas as pd
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
    parser.add_argument("--games", type=int, required=False, default=3,
                        help="Number of games (simulations) to run per circle.")
    parser.add_argument("--p1_circles", type=int, nargs='+', default=[0, 1, 2, 3, 4, 5, 6],
                        help="List of integer circle values to iterate over.")
    parser.add_argument("--p2_circles", type=int, nargs='+', default=[0, 1, 2, 3, 4, 5, 6],
                        help="List of integer circle values to iterate over.")
    parser.add_argument("--parallel", type=bool, required=False, default=True,
                        help="Whether to run the experiments in parallel.")
    parser.add_argument("--discount", type=float, required=False, default=.9, help="discount rate for game ranging between [0, 1].")
    args = parser.parse_args()

    print("[INFO] Starting experiment...")
    print(f"  prompt_style: {args.prompt_style}")
    print(f"  llm_model_p1:     {args.llm_model_p1}")
    print(f"  llm_model_p2:     {args.llm_model_p2}")
    print(f"  date:         {args.date}")
    print(f"  max_rounds:   {args.max_rounds}")
    print(f"  games:        {args.games}")
    print(f"  p1 circles:      {args.p1_circles}")
    print(f"  p2 circles:      {args.p2_circles}")
    print(f"  discount:         {args.discount}")
    print("--------------------------------------------------")
    
    valid_models = ["openai_4o_2024-08-06", "openai_4o_2024-11-20", "openai_o3_mini_2025-01-31", "anthropic_3.5_sonnet_2024-10-22", "anthropic_3.7_sonnet_2025-02-19", 
                    "gemini_2.0_flash", "llama3.3-70b", "llama3.3-8b", "llama3.3-4050", "openai_o1_mini_2024-09-12", "openai_o1_preview_2024-09-12", "openai_o1_2024-12-17",
                    "anthropic_sonnet_3.7_reasoning_2025-02-19", "deepseek_reasoner"]
    if args.llm_model_p1 not in valid_models:
        raise ValueError(f"Invalid model: {args.llm_model_p1}")
    if args.llm_model_p2 not in valid_models:
        raise ValueError(f"Invalid model: {args.llm_model_p2}")
    if args.discount > 1 or args.discount < 0:
        raise ValueError(f"Invalid Discount: {args.llm_model_p2}")

    if args.parallel:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            future_to_circle = {
                executor.submit(
                    run_game, 
                    circle1, 
                    circle2, 
                    args.games, 
                    args.max_rounds, 
                    args.date, 
                    args.prompt_style, 
                    args.llm_model_p1,
                    args.llm_model_p2,
                    args.discount
                ): (circle1, circle2)
                for (circle1, circle2) in zip(args.p1_circles, args.p2_circles)
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
        for circle1, circle2 in zip(args.p1_circles, args.p2_circles):
            run_game(circle1, circle2, args.games, args.max_rounds, args.date, args.prompt_style, args.llm_model_p1, args.llm_model_p2, args.discount)


if __name__ == "__main__":
    main()