import sys
sys.path.append('../')

from game_runner import NegotitaionGame
from eval.game_evaluator import GameEvaluator
import agents.simple_agent as simple_agent
import agents.llm_agent as llm_agent
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from dataclasses import dataclass, field
from math import prod, sqrt
sys.path.append('../caif_negotiation/')


import import_ipynb
from IPython import get_ipython
import runpy
import os

ipython = get_ipython()
if ipython is not None:
    ipython.run_line_magic('run', '../test_game_eval.ipynb')
else:
    py_file = os.path.join('..', 'test_game_eval.py')
    if os.path.isfile(py_file):
        runpy.run_path(py_file)
    else:
        print(
            "ERROR: Not in an IPython environment, and ../test_game_eval.ipynb "
            "hasn’t been converted to ../test_game_eval.py. Please convert it:\n"
            "  jupyter nbconvert --to python ../test_game_eval.ipynb\n"
            "…and then try again."
        )
from test_game_eval import *
import torch
from utils.offer import Offer

from prompts.make_prompt import make_prompt
from prompts.make_prompt_bargain import make_prompt_bargain
from metrics.visualizations import (
    plot_discounted_values,
    plot_offer_evolution,
    plot_negotiation_gap,
    plot_fairness
)

pathology_results = pd.DataFrame()  
import itertools
envy_results_history = {}
from eval.metrics import *
import time
import pandas as pd
import torch
import numpy as np
from math import sqrt, prod
from utils.helpers import *
import time
import numpy as np
import pandas as pd
import torch
from math import sqrt, prod
from eval.game_data import GameData  # Importing GameData from game_data.py
import pickle
import json


def run_game(circle: int, games: int, max_rounds: int, date: str, game_title: str, llm_type: str):
    """
    Runs a series of negotiation games for a specific circle, tracking comprehensive metrics.

    Args:
        circle (int): The circle parameter influencing allocation strategies.
        games (int): Number of games to simulate.
        max_rounds (int): Maximum number of rounds per game.
        date (str): Date identifier for result files.
        game_title (str): Title identifier for the game series.
        llm_type (str): Type of LLM agent being used (e.g., "openai").
    """
    # Initialize a list to store all GameData instances
    all_game_data = []

    for i in range(games):
        # --------------------------------------------------------------------
        # 1) Per-Game Setup
        # --------------------------------------------------------------------
        # Rate-limit every 10 games to avoid API overuse
        if (i + 1) % 10 == 0:
            print(f"Game {i + 1} of {games}")
            sleep_duration = 2 * np.random.randint(55, 60)  # Sleep for ~2 minutes
            print(f"Sleeping for {sleep_duration} seconds to respect rate limits.")
            time.sleep(sleep_duration)

        # Reset per-game trackers
        p1_values = []
        p2_values = []
        p1_offers = []
        p2_offers = []
        p1_offers_utility = []
        p2_offers_utility = []
        p1_kept = None
        p2_kept = None

        # --------------------------------------------------------------------
        # 2) Initialize a Single Negotiation Game
        # --------------------------------------------------------------------
        game = NegotitaionGame(
            player1_agent=llm_agent.LLMAgent(llm_type=llm_type, player_num=0),
            player2_agent=llm_agent.LLMAgent(llm_type=llm_type, player_num=1),
            num_items=5,
            item_value_range=[1, 101],
            gamma=0.9,
            max_rounds=max_rounds,
            circle=circle
        )

        # Compute Pareto frontier for reference
        pareto_front = compute_pareto_frontier(
            game.player_values[0],
            game.player_values[1],
            game.num_items,
            game.items,
            game.outside_offer_values
        )

        # --------------------------------------------------------------------
        # 3) Optional: Find Allocations with Utility < Outside Offer (Circles 5 & 6)
        # --------------------------------------------------------------------
        allocations_less_than_outside_offer = None
        if circle in (5, 6):
            allocations_less_than_outside_offer = []

            # Find allocations where Player 1's utility is less than their outside offer
            allocation_p1 = find_allocation_less_than_outside_offer_dp(
                items=game.items,
                player_valuations=game.player_values[0],
                outside_offer=game.outside_offer_values[0],
                player_num=1
            )
            if allocation_p1:
                allocations_less_than_outside_offer.append({
                    'player': 1,
                    'allocation': list(allocation_p1.values())
                })
            else:
                allocations_less_than_outside_offer.append({
                    'player': 1,
                    'allocation': [0] * game.num_items
                })
                print(f"[INFO] No feasible < outside_offer allocation for Player 1 in Game {i + 1}.")

            # Find allocations where Player 2's utility is less than their outside offer
            allocation_p2 = find_allocation_less_than_outside_offer_dp(
                items=game.items,
                player_valuations=game.player_values[1],
                outside_offer=game.outside_offer_values[1],
                player_num=2
            )
            if allocation_p2:
                allocations_less_than_outside_offer.append({
                    'player': 2,
                    'allocation': list(allocation_p2.values())
                })
            else:
                allocations_less_than_outside_offer.append({
                    'player': 2,
                    'allocation': [0] * game.num_items
                })
                print(f"[INFO] No feasible < outside_offer allocation for Player 2 in Game {i + 1}.")

            print(f"[DEBUG] Game {i + 1} allocations_less_than_outside_offer: {allocations_less_than_outside_offer}")

        print(f"[DEBUG] game.items: {game.items}")
        print(f"[DEBUG] allocations_less_than_outside_offer: {allocations_less_than_outside_offer}")

        # --------------------------------------------------------------------
        # 4) Initialize Game History
        # --------------------------------------------------------------------
        game_history = GameHistory(
            agent_1_name="Agent1",
            agent_2_name="Agent2",
            num_items=game.num_items,
            items=torch.tensor(game.items),
            agent_1_values=torch.tensor(game.player_values[0]),
            agent_2_values=torch.tensor(game.player_values[1]),
            agent_1_outside_value=game.outside_offer_values[0],
            agent_2_outside_value=game.outside_offer_values[1]
        )
        game_history.agent_1_offers = []
        game_history.agent_2_offers = []

        # --------------------------------------------------------------------
        # 5) Initialize GameData Instance
        # --------------------------------------------------------------------
        game_data = GameData(
            circle=circle,
            date=date,
            agent1="Agent1",
            agent2="Agent2"
        )

        print(f"[INFO] Starting Game {i + 1} of {games} for Circle {circle}.")

        # --------------------------------------------------------------------
        # 6) (Optional) Verify Previous Game Results
        # --------------------------------------------------------------------
        if i > 0:
            # Not necessary when using GameData, but kept for consistency
            print(f"[DEBUG] Processing Game {i} completed.")

        # --------------------------------------------------------------------
        # 7) Main Negotiation Loop
        # --------------------------------------------------------------------
        while game.in_progress:
            # Sleep to simulate thinking time and rate-limit API calls
            sleep_duration = 5  # Adjust based on desired rate-limiting
            print(f"[DEBUG] Sleeping for {sleep_duration} seconds before next step.")
            time.sleep(sleep_duration)

            # Determine current step, round, and player
            current_step = len(game.history[0]) + len(game.history[1]) + 1
            current_round = (current_step - 1) // 2 + 1
            current_player = 1 if current_step % 2 == 1 else 2
            game.current_round = current_round

            # Initialize pathology flags for this step
            making_offer_worse_than_previous = False
            making_offer_worse_than_outside_offer = False
            offer_no_items_or_all_items = False
            accepting_an_offer_worse_than_outside_offer = False
            walking_away_from_an_offer_better_than_outside_offer = False
            walk_away_type = None
            your_side_of_current_offer = None

            print("\n" + "=" * 80)
            print(f"Game {i + 1}, Round {current_round}, Player {current_player}'s turn (Step {current_step})")
            print("=" * 80)

            # Capture the previous offer before the player acts
            prev_offer = game.current_offer

            # Provide example allocation if circle is 5 or 6
            current_allocation_example = None
            if circle in (5, 6) and allocations_less_than_outside_offer is not None:
                if current_player == 1:
                    current_allocation_example = allocations_less_than_outside_offer[0]['allocation']
                elif current_player == 2:
                    current_allocation_example = allocations_less_than_outside_offer[1]['allocation']

            print(f"[DEBUG] Current allocation example type: {type(current_allocation_example)}")

            # Let the current player make a move (ACCEPT, WALK, COUNTEROFFER)
            # Capture the prompt and response
            response = game.step(example_offer_less_than_outside_offer_self=current_allocation_example)

            # Capture the action played by the current player
            action_played = game.players[current_player - 1].action.upper()

            # Check pathologies #4 (accepting worse) and #5 (walking away from better)
            walking_away_from_an_offer_better_than_outside_offer = check_walking_away_from_better(
                current_player, p1_kept, p2_kept, game
            )
            accepting_an_offer_worse_than_outside_offer = check_accepting_worse_than_outside(
                current_player, p1_kept, p2_kept, game
            )

            # Determine walk away type (valid/invalid/no walk)
            walk_away_type = determine_walk_away_type(current_player, game)

            # ------------------------------------------------------------
            # Check if the current allocation is on the Pareto Frontier
            # ------------------------------------------------------------
            on_pareto_frontier = False
            for vals in pareto_front:
                if vals["type"] == "outside_offer" and action_played == "WALK":
                    on_pareto_frontier = True
                    break
                elif vals["type"] == "allocation":
                    if (np.array_equal(vals["agent1"], p1_kept) and 
                        np.array_equal(vals["agent2"], p2_kept)):
                        on_pareto_frontier = True
                        break

            # ------------------------------------------------------------
            # 8) Compute Metrics if an Offer Exists
            # ------------------------------------------------------------
            metrics = {}
            if game.current_offer:
                # Determine realization round based on action
                if action_played == "COUNTEROFFER":
                    realization_round_p1 = current_round
                    realization_round_p2 = current_round
                else:
                    realization_round_p1 = current_round
                    realization_round_p2 = current_round

                # Update allocations based on the offer
                p1_kept, p2_kept = update_kept_portions(current_player, game, p1_kept, p2_kept)

                # Calculate undiscounted utilities
                p1_undisc_val = float(np.dot(game.player_values[0], p1_kept))
                p2_undisc_val = float(np.dot(game.player_values[1], p2_kept))
                p1_offers_utility.append(p1_undisc_val)
                p2_offers_utility.append(p2_undisc_val)

                # Calculate discounted utilities
                p1_value = calculate_discounted_value(
                    p1_kept, game.player_values[0], game.gamma, realization_round_p1
                )
                p2_value = calculate_discounted_value(
                    p2_kept, game.player_values[1], game.gamma, realization_round_p2
                )

                # Track values across rounds
                p1_values.append(p1_value)
                p2_values.append(p2_value)

                # Log which player's offer it is
                if current_player == 1:
                    p1_offers.append(list(game.current_offer.offer))
                else:
                    p2_offers.append(list(game.current_offer.offer))
                    print(f"\n[INFO] Round {current_round} Values:")
                    print(f"Player 1: {p1_value:.2f}")
                    print(f"Player 2: {p2_value:.2f}")

                # ------------------------------
                # Pathology #1: Offer is worse than the immediate previous offer
                # ------------------------------
                if action_played == "COUNTEROFFER":
                    if (current_round > 1) or (current_round == 1 and current_player == 2):
                        if current_player == 1:
                            previous_offer_value = p1_values[-2]
                            current_offer_value = p1_values[-1]
                        else:
                            previous_offer_value = p2_values[-2]
                            current_offer_value = p2_values[-1]
                        making_offer_worse_than_previous = (
                            previous_offer_value > current_offer_value
                        )

                    # ------------------------------
                    # Pathology #2: Offer is worse than your outside offer
                    # ------------------------------
                    if current_player == 1:
                        outside_offer = float(game.outside_offer_values[0]) * (game.gamma ** (current_round - 1))
                        making_offer_worse_than_outside_offer = (p1_value < outside_offer)
                    else:
                        outside_offer = float(game.outside_offer_values[1]) * (game.gamma ** current_round)
                        making_offer_worse_than_outside_offer = (p2_value < outside_offer)
                else:
                    making_offer_worse_than_outside_offer = False

                # ------------------------------
                # Pathology #3: Offer has no items or all items
                # ------------------------------
                if action_played == "COUNTEROFFER":
                    if (np.array_equal(game.current_offer.offer, [0] * game.num_items) or
                            np.array_equal(game.current_offer.offer, game.items)):
                        offer_no_items_or_all_items = True

                # ------------------------------
                # Compute Nash Welfare
                # ------------------------------
                if action_played in ("ACCEPT", "COUNTEROFFER"):
                    nash_welfare = sqrt(prod([
                        np.dot(game.player_values[0], p1_kept),
                        np.dot(game.player_values[1], p2_kept)
                    ]))
                else:
                    nash_welfare = sqrt(prod([
                        game.outside_offer_values[0],
                        game.outside_offer_values[1]
                    ]))

                # ------------------------------------------------------------
                # Utility Proportion for the Round (x1_prop, x2_prop)
                # ------------------------------------------------------------
                if action_played == "WALK":
                    x1_prop = float(game.outside_offer_values[0])
                    x2_prop = float(game.outside_offer_values[1])
                else:
                    total_val_p1_kept = float(np.dot(game.player_values[0], p1_kept))
                    total_val_p2_kept = float(np.dot(game.player_values[1], p2_kept))
                    total_val = total_val_p1_kept + total_val_p2_kept
                    if total_val > 0:
                        x1_prop = total_val_p1_kept / total_val
                        x2_prop = total_val_p2_kept / total_val
                    else:
                        x1_prop, x2_prop = 0.0, 0.0

                # ------------------------------------------------------------
                # Concession Size
                # ------------------------------------------------------------
                concession_size = None
                if action_played == "COUNTEROFFER" and current_round > 1:
                    if current_player == 1 and len(p1_offers_utility) >= 2:
                        old_value = p1_offers_utility[-2]
                        new_value = p1_offers_utility[-1]
                        concession_size = max(0, old_value - new_value)
                    elif current_player == 2 and len(p2_offers_utility) >= 2:
                        old_value = p2_offers_utility[-2]
                        new_value = p2_offers_utility[-1]
                        concession_size = max(0, old_value - new_value)
                else:
                    # Initialize utility if first COUNTEROFFER
                    if action_played == "COUNTEROFFER":
                        if current_player == 1 and not p1_offers_utility:
                            p1_offers_utility.append(p1_undisc_val)
                        elif current_player == 2 and not p2_offers_utility:
                            p2_offers_utility.append(p2_undisc_val)

                # ------------------------------------------------------------
                # Proportion of Collective Value for Each Player
                # ------------------------------------------------------------
                proportion_of_value_p1 = None
                proportion_of_value_p2 = None

                if action_played in ("ACCEPT", "COUNTEROFFER"):
                    denom = float(np.dot(game.player_values[0], p1_kept) + np.dot(game.player_values[1], p2_kept))
                    if denom > 0:
                        proportion_of_value_p1 = float(np.dot(game.player_values[0], p1_kept)) / denom
                        proportion_of_value_p2 = float(np.dot(game.player_values[1], p2_kept)) / denom
                    else:
                        proportion_of_value_p1 = 0.0
                        proportion_of_value_p2 = 0.0

                # ------------------------------------------------------------
                # Proportional Share of the Maximum Available Utility
                # ------------------------------------------------------------
                proposal_proportion_player_1_available_items = None
                proposal_proportion_player_2_available_items = None
                if action_played in ("ACCEPT", "COUNTEROFFER"):
                    total_p1_available_items = float(np.dot(game.player_values[0], game.items))
                    total_p2_available_items = float(np.dot(game.player_values[1], game.items))

                    proposal_proportion_player_1_available_items = (
                        float(np.dot(game.player_values[0], p1_kept)) / total_p1_available_items
                    ) if total_p1_available_items > 0 else 0.0

                    proposal_proportion_player_2_available_items = (
                        float(np.dot(game.player_values[1], p2_kept)) / total_p2_available_items
                    ) if total_p2_available_items > 0 else 0.0

                # ------------------------------------------------------------
                # Utilitarian & Rawlsian Welfare
                # ------------------------------------------------------------
                if action_played == "WALK":
                    utilitarian_welfare = float(sum(game.outside_offer_values))
                    rawlsian_welfare = float(min(game.outside_offer_values))
                else:
                    utilitarian_welfare = float(np.dot(game.player_values[0], p1_kept) + np.dot(game.player_values[1], p2_kept))
                    rawlsian_welfare = float(min(np.dot(game.player_values[0], p1_kept),
                                                 np.dot(game.player_values[1], p2_kept)))

                # ------------------------------------------------------------
                # Gini Coefficient (n=2)
                # ------------------------------------------------------------
                if action_played == "WALK":
                    wealth_distribution = np.array([
                        float(game.outside_offer_values[0]),
                        float(game.outside_offer_values[1])
                    ], dtype=float)
                else:
                    wealth_distribution = np.array([
                        float(np.dot(game.player_values[0], p1_kept)),
                        float(np.dot(game.player_values[1], p2_kept))
                    ], dtype=float)

                total_wealth = wealth_distribution.sum()
                if total_wealth > 0:
                    x1, x2 = wealth_distribution
                    gini_coefficient = abs(x1 - x2) / (4.0 * total_wealth)
                else:
                    gini_coefficient = 0.0

                # Jain's Fairness Index Calculation
                if action_played == "WALK":
                    x1 = float(game.outside_offer_values[0])
                    x2 = float(game.outside_offer_values[1])
                else:
                    x1 = float(np.dot(game.player_values[0], p1_kept))
                    x2 = float(np.dot(game.player_values[1], p2_kept))

                # Mean Utility
                mean_utility = (x1 + x2) / 2.0

                # Variance
                variance_value = (x1**2 + x2**2) / 2.0 - mean_utility**2
                variance_value = max(variance_value, 0.0)  # Correct for negative variance due to precision

                # Coefficient of Variation
                if mean_utility != 0:
                    coefficient_of_variation = np.sqrt(variance_value) / mean_utility
                else:
                    coefficient_of_variation = 0.0  # Avoid division by zero

                # Jain's Fairness Index
                jain_fairness_index = 1 / (1 + coefficient_of_variation ** 2)

                # ------------------------------------------------------------
                # Security Level
                # ------------------------------------------------------------
                if action_played == "WALK":
                    security_level_player_1 = 0.0
                    security_level_player_2 = 0.0
                else:
                    security_level_player_1 = max(
                        0.0, float(game.outside_offer_values[0] - np.dot(game.player_values[0], p1_kept))
                    )
                    security_level_player_2 = max(
                        0.0, float(game.outside_offer_values[1] - np.dot(game.player_values[1], p2_kept))
                    )

                # ------------------------------------------------------------
                # Mean Absolute Difference (MAD)
                # ------------------------------------------------------------
                if action_played == "WALK":
                    x1 = float(game.outside_offer_values[0])
                    x2 = float(game.outside_offer_values[1])
                else:
                    x1 = float(np.dot(game.player_values[0], p1_kept))
                    x2 = float(np.dot(game.player_values[1], p2_kept))

                if x1 == 0.0 and x2 == 0.0:
                    mean_absolute_difference = 0.0
                else:
                    mean_absolute_difference = abs(x1 - x2) / 2.0

                # ------------------------------------------------------------
                # Compile Metrics
                # ------------------------------------------------------------
                metrics = {
                    "game_num": i,
                    "step_num": current_step,
                    "round_num": current_round,
                    "player": current_player,
                    "action_played": action_played,
                    "discount_rate": game.gamma ** (current_round - 1),
                    "offer": list(game.current_offer.offer) if game.current_offer else [],
                    "value": p1_value if current_player == 1 else p2_value,
                    "undiscounted_value": (
                        p1_value / (game.gamma ** (realization_round_p1 - 1)) if current_player == 1 else
                        p2_value / (game.gamma ** (realization_round_p2 - 1))
                    ),
                    "p1_valuations": list(game.player_values[0]),
                    "p2_valuations": list(game.player_values[1]),
                    "p1_kept_allocation": list(p1_kept) if p1_kept is not None else None,
                    "p2_kept_allocation": list(p2_kept) if p2_kept is not None else None,
                    "p1_final_value": p1_value,
                    "p2_final_value": p2_value,
                    "items": list(game.items),
                    "your_side_of_current_offer": your_side_of_current_offer,
                    "outside_offer": game.outside_offer_values[current_player - 1] * (game.gamma ** (current_round - 1)),  
                    "outside_offer_undiscounted": game.outside_offer_values[current_player - 1], 
                    "accepting_an_offer_worse_than_outside_offer": accepting_an_offer_worse_than_outside_offer,
                    "making_an_offer_worse_for_you_than_your_outside_offer": making_offer_worse_than_outside_offer,
                    "walking_away_from_an_offer_better_than_your_outside_offer": walking_away_from_an_offer_better_than_outside_offer,
                    "offer_no_items_or_all_items": offer_no_items_or_all_items,
                    "making_offer_worse_than_previous": making_offer_worse_than_previous,
                    "nash_welfare": nash_welfare,
                    "proposal_proportion_player_1": x1_prop,
                    "proposal_proportion_player_2": x2_prop,
                    "concession_size": concession_size,
                    "security_level_player_1": security_level_player_1,
                    "security_level_player_2": security_level_player_2,
                    "average_concession_size": None,  # To be computed post-game if needed
                    "rawlsian_welfare": rawlsian_welfare,
                    "gini_coefficient": gini_coefficient,
                    "utilitarian_welfare": utilitarian_welfare,
                    "jain_fairness_index": jain_fairness_index,
                    "on_pareto_frontier": on_pareto_frontier,
                    "mean_absolute_difference": mean_absolute_difference,
                    "walk_type": walk_away_type
                }

                # ------------------------------------------------------------
                # Add Round Data to GameData
                # ------------------------------------------------------------
                game_data.add_round_data(
                    prompt=game.players[current_player - 1].current_prompt,
                    response=game.players[current_player - 1].current_response,  # Assuming response includes the agent's textual response
                    action=action_played,
                    game_metrics=metrics,
                    envy_free_metrics=game_history.to_dict()
                )

                if "WALK" or "ACCEPT" in action_played:
                    game_data.set_outcome(metrics)

            # ---------------------------------------------------------
            # 9) Add the Current Offer to Game History if Complete
            # ---------------------------------------------------------
            if game.current_offer is not None and len(game.current_offer.offer) == game.num_items:
                offer_items = list(game.current_offer.offer) if isinstance(game.current_offer.offer, torch.Tensor) else game.current_offer.offer
                game_history.add_offer(
                    game.current_player,
                    Offer(game.current_player, offer=offer_items)
                )

            # ---------------------------------------------------------
            # 10) Check Final-Round Logic (Player 2's Turn on Last Round)
            # ---------------------------------------------------------
            if current_round == game.max_rounds and current_player == 2:
                # Handle acceptance/walk/counter in final round
                final_metrics = handle_final_round(
                    game_num=i,
                    current_round=current_round,
                    current_player=current_player,
                    game=game,
                    prev_offer=prev_offer,
                    p1_kept=p1_kept,
                    p2_kept=p2_kept,
                    p1_values=p1_values,
                    p2_values=p2_values,
                    p1_offers=p1_offers,
                    accepting_an_offer_worse_than_outside_offer=accepting_an_offer_worse_than_outside_offer,
                    pareto_front=pareto_front
                )

                # Add the final round data to GameData
                game_data.add_round_data(
                    prompt=game.players[current_player - 1].current_prompt,
                    response=game.players[current_player - 1].current_response,  # Assuming response attribute exists
                    action=final_metrics["action_played"],
                    game_metrics=final_metrics,
                    envy_free_metrics=game_history.to_dict()
                )

                # Set the final outcome in GameData
                game_data.set_outcome(final_metrics)

                # End the game loop
                break

            # ---------------------------------------------------------
            # 11) Handle Walk Away Scenarios
            # ---------------------------------------------------------
            elif (game.current_offer is None
                  and "WALK" in action_played):
                print(f"[INFO] Player {current_player} walked away.")

                # Compile walk-away metrics
                walk_metrics = {
                    "game_num": i,
                    "step_num": current_step,
                    "round_num": current_round,
                    "player": current_player,
                    "action_played": action_played,
                    "discount_rate": game.gamma ** (current_round - 1),
                    "offer": [],
                    "value": None,  # No value since walk occurred
                    "undiscounted_value": game.outside_offer_values[current_player - 1],
                    "p1_valuations": list(game.player_values[0]),
                    "p2_valuations": list(game.player_values[1]),
                    "p1_kept_allocation": None,  # No allocation
                    "p2_kept_allocation": None,  # No allocation
                    "p1_final_value": float(game.outside_offer_values[0]),
                    "p2_final_value": float(game.outside_offer_values[1]),
                    "items": list(game.items),
                    "your_side_of_current_offer": None,
                    "outside_offer": game.outside_offer_values[current_player - 1] * (game.gamma ** (current_round - 1)),
                    "outside_offer_undiscounted": game.outside_offer_values[current_player - 1],
                    "accepting_an_offer_worse_than_outside_offer": accepting_an_offer_worse_than_outside_offer,
                    "making_an_offer_worse_for_you_than_your_outside_offer": making_offer_worse_than_outside_offer,
                    "walking_away_from_an_offer_better_than_your_outside_offer": walking_away_from_an_offer_better_than_outside_offer,
                    "offer_no_items_or_all_items": offer_no_items_or_all_items,
                    "making_offer_worse_than_previous": making_offer_worse_than_previous,
                    "nash_welfare": sqrt(prod([
                        game.outside_offer_values[0],
                        game.outside_offer_values[1]
                    ])),
                    "proposal_proportion_player_1": None,
                    "proposal_proportion_player_2": None,
                    "concession_size": None,
                    "security_level_player_1": 0.0,
                    "security_level_player_2": 0.0,
                    "average_concession_size": None,  # To be computed post-game
                    "rawlsian_welfare": float(min(game.outside_offer_values[0], game.outside_offer_values[1])),
                    "gini_coefficient": (
                        abs(game.outside_offer_values[0] - game.outside_offer_values[1]) /
                        (4.0 * sum(game.outside_offer_values))
                    ) if sum(game.outside_offer_values) > 0 else 0.0,
                    "utilitarian_welfare": float(sum(game.outside_offer_values)),
                    "jain_fairness_index": (
                        1 / (1 + (abs(game.outside_offer_values[0] - game.outside_offer_values[1]) / sum(game.outside_offer_values))**2)
                        if sum(game.outside_offer_values) > 0 else 0.0
                    ),
                    "on_pareto_frontier": on_pareto_frontier, 
                    "mean_absolute_difference": abs(
                        game.outside_offer_values[0] - game.outside_offer_values[1]
                    ) / 2.0,
                    "walk_type": walk_away_type
                }

                # Add the walk-away data to GameData
                game_data.add_round_data(
                    prompt=game.players[current_player - 1].current_prompt,
                    response=game.players[current_player - 1].current_response,
                    action=game.players[current_player - 1].action,
                    game_metrics=walk_metrics,
                    envy_free_metrics=game_history.to_dict()
                )

                # Set the final outcome in GameData
                game_data.set_outcome(walk_metrics)

                # End the game loop
                game.in_progress = False

        # --------------------------------------------------------------------
        # 12) After the Game Loop Ends, Save GameData
        # --------------------------------------------------------------------
        all_game_data.append(game_data)
        #UNCOMMENT THESE TO SAVE EACH GAME'S DATA SEPERATELY 
        # Optionally, save each game's data immediately
        # Filename can include game number, circle, date, etc.
        #filename = f'game_data_{date}_game_{i + 1}_circle_{circle}.json'
        #game_data.save_to_json(filename)
        #print(f"[INFO] Saved GameData to {filename}.")
        #save to pickle
        #filename_pkl = f'game_data_{date}_game_{i + 1}_circle_{circle}.pkl'
        #with open(filename_pkl, "wb") as pf:
        #    pickle.dump(game_data, pf)
        #print(f"[INFO] Saved GameData to {filename_pkl}.")

    # ------------------------------------------------------------------------
    # 13) Save All Games' Data (Optional)
    # ------------------------------------------------------------------------
    # If you prefer to save all games' data collectively, you can serialize the list
    # For example, save as a list of dictionaries
    print("HERE IS THE DATA")
    all_data = {
        "date": date,
        "circle": circle,
        "games": games,
        "game_title": game_title,
        "all_game_data": [gd.to_dict() for gd in all_game_data]
    }
    all_games_filename = f'all_game_data_{date}_{games}_{game_title}_circle_{circle}.json'
    with open(all_games_filename, "w") as f:
        json.dump(all_data, f, indent=4)
        #json.pickle(all_data, f)
    print(f"[INFO] Saved all GameData to JSON file: {all_games_filename}.")

    #save to pickle optinally
    all_games_filename_pkl = f'all_game_data_{date}_{games}_{game_title}_circle_{circle}.pkl'
    with open(all_games_filename_pkl, "wb") as pf:
        pickle.dump(all_data, pf)
    print(f"[INFO] Saved all GameData as a pickle to {all_games_filename_pkl}.")

