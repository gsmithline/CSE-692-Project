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

# Execute the notebook - place this before any code that needs GameHistory
ipython = get_ipython()
if ipython is not None:
    # If in IPython/Jupyter, we can safely run the .ipynb directly
    ipython.run_line_magic('run', '../test_game_eval.ipynb')
else:
    # Fallback if not in IPython/Jupyter environment: run the converted .py file
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
import concurrent.futures



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




def run_game(circle: int, games: int, max_rounds: int, date: str, game_title: str, llm_type: str):
    rounds = []
    p1_values = []
    p2_values = []
    pathology_results = pd.DataFrame()
    envy_results_history = {}

    for i in range(games):
        # Print progress / rate-limit
        if (i + 1) % 10 == 0:
            print(f"Game {i + 1} of {games}")
            time.sleep(2 * np.random.randint(55, 60))  # Sleep for 2 minutes
        
        # Reset per-game trackers
        rounds = []
        p1_values = []
        p2_values = []
        p1_offers = []
        p2_offers = []
        p1_offers_utility = []
        p2_offers_utility = []
        p1_kept = None
        p2_kept = None

        # Initialize game & game history
        game = NegotitaionGame(
            player1_agent=llm_agent.LLMAgent(llm_type=llm_type, player_num=0),
            player2_agent=llm_agent.LLMAgent(llm_type=llm_type, player_num=1),
            num_items=5,
            item_value_range=[1, 101],
            gamma=0.9,
            max_rounds=max_rounds,
            circle=circle
        )

        pareto_front = compute_pareto_frontier(
            game.player_values[0],
            game.player_values[1],
            game.num_items,
            game.items,
            game.outside_offer_values
        )

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
                    'allocation': [value for value in allocation_p1.values()]
                })
            else:
                allocations_less_than_outside_offer.append({
                    'player': 1,
                    'allocation': [0, 0, 0, 0, 0]
                })
                print(f"No feasible allocation found for Player 1 in Game {i + 1} where utility < outside offer.")

            # Find allocations where Player 2's utility is less than their outside offer
            print(f"game.items: {game.items}")
            
            allocation_p2 = find_allocation_less_than_outside_offer_dp(
                items=game.items,
                player_valuations=game.player_values[1],
                outside_offer=game.outside_offer_values[1],
                player_num=2
            )
            if allocation_p2:
                allocations_less_than_outside_offer.append({
                    'player': 2,
                    'allocation': [value for value in allocation_p2.values()]
                })
            else:
                print(f"No feasible allocation found for Player 2 in Game {i + 1} where utility < outside offer.")
                allocations_less_than_outside_offer.append({
                    'player': 2,
                    'allocation': [0, 0, 0, 0, 0]
                })

        print(f"game.items: {game.items}")
        print(f"allocations_less_than_outside_offer: {allocations_less_than_outside_offer}")

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

        # Prepare a DataFrame to record step-by-step results
        game.game_results = pd.DataFrame(columns=[
            "game_num", "round_num", "player", "discount_rate", "offer", "value",
            "undiscounted_value", "items", "your_side_of_current_offer",
            "outside_offer", "outside_offer_undiscounted",
            "accepting_an_offer_worse_than_outside_offer",
            "making_an_offer_worse_for_you_than_your_outside_offer",
            "walking_away_from_an_offer_better_than_your_outside_offer",
            "offer_no_items_or_all_items", "making_offer_worse_than_previous",
            "nash_welfare", "proposal_proportion_player_1",
            "proposal_proportion_player_2", "concession_size",
            "security_level_player_1", "security_level_player_2",
            "average_concession_size", "rawlsian_welfare", "gini_coefficient",
            "utilitarian_welfare", "jain_fairness_index",
            "on_pareto_frontier", "mean_absolute_difference",  # <-- fixed column name
            "walk_type"
        ])

        print(f"PLAYING GAME_NUM: {i}")

        # (Optional) check whether previous game results exist
        # Commenting out or adjusting this logic if it causes breaks
        if i > 0:
            print("Unique games in pathology_results:", pathology_results['game_num'].unique())
            if i - 1 in pathology_results['game_num'].unique():
                print(f"Game {i - 1} exists in pathology_results")
            else:
                print(f"Game {i - 1} does not exist in pathology_results")
                # If you do NOT want to break, just comment this out:
                # break

        # --------------------------------------------------------------------
        # Main negotiation loop
        # -------------------------------------------------------------------- 
        while game.in_progress:
            time.sleep(circle + 0.5)  # Sleep 5s to simulate thinking or rate-limit
            current_step = len(game.history[0]) + len(game.history[1]) + 1
            current_round = (current_step - 1) // 2 + 1
            current_player = 1 if current_step % 2 == 1 else 2
            game.current_round = current_round

            # Pathology flags
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

            # Remember the previous offer, let the current player move
            prev_offer = game.current_offer
            current_allocation_example = None

            if circle in (5, 6):
                if current_player == 1:
                    current_allocation_example = allocations_less_than_outside_offer[0]['allocation']
                elif current_player == 2:
                    current_allocation_example = allocations_less_than_outside_offer[1]['allocation']

            print(f"type of current_allocation_example: {type(current_allocation_example)}")
            game.step(example_offer_less_than_outside_offer_self=current_allocation_example)

            # Check pathologies #4 (accepting worse) and #5 (walking away from better)
            walking_away_from_an_offer_better_than_outside_offer = check_walking_away_from_better(
                current_player, p1_kept, p2_kept, game
            )
            accepting_an_offer_worse_than_outside_offer = check_accepting_worse_than_outside(
                current_player, p1_kept, p2_kept, game
            )

            # Check for walk away type (valid/invalid/no walk)
            walk_away_type = determine_walk_away_type(current_player, game)

            # If there's a current offer, compute new values and pathologies
            if game.current_offer:
                # Determine realization round for each player
                action = game.players[current_player - 1].action
                if action == "COUNTEROFFER":
                    if current_player == 1:
                        realization_round_p1 = current_round
                        realization_round_p2 = current_round
                    else:
                        realization_round_p1 = current_round + 1
                        realization_round_p2 = current_round + 1
                else:
                    # For ACCEPT or other actions, realization is the current round
                    realization_round_p1 = current_round
                    realization_round_p2 = current_round

                # Update p1_kept, p2_kept if the current player made a counteroffer
                p1_kept, p2_kept = update_kept_portions(current_player, game, p1_kept, p2_kept)

                # Track undiscounted utilities
                p1_offers_utility.append(np.dot(game.player_values[0], p1_kept))
                p2_offers_utility.append(np.dot(game.player_values[1], p2_kept))

                # Discounted values
                p1_value = calculate_discounted_value(
                    p1_kept, game.player_values[0], game.gamma, realization_round_p1
                )
                p2_value = calculate_discounted_value(
                    p2_kept, game.player_values[1], game.gamma, realization_round_p2
                )

                rounds.append(current_round)
                p1_values.append(p1_value)
                p2_values.append(p2_value)

                # Track which player's offer it is
                if current_player == 1:
                    p1_offers.append(game.current_offer.offer)
                else:
                    p2_offers.append(game.current_offer.offer)
                    print(f"\nRound {current_round} Values:")
                    print(f"Player 1: {p1_value:.2f}")
                    print(f"Player 2: {p2_value:.2f}")

                # ------------------------------
                # Pathology #1: Offer is worse than the immediate previous offer
                # ------------------------------
                if action == "COUNTEROFFER":
                    # We can only compare if there's a previous offer for the same player
                    if (current_round == 1 and current_player == 2) or (current_round > 1):
                        if current_player == 1:
                            your_side_of_previous_offer = p1_values[-2]
                            your_side_of_current_offer = p1_values[-1]
                        else:
                            your_side_of_previous_offer = p2_values[-2]
                            your_side_of_current_offer = p2_values[-1]

                        making_offer_worse_than_previous = (
                            your_side_of_previous_offer > your_side_of_current_offer
                        )

                    # ------------------------------
                    # Pathology #2: Offer is worse than your outside offer
                    # ------------------------------
                    if current_player == 1:
                        outside_offer = game.outside_offer_values[0] * (game.gamma ** (current_round - 1))
                        making_offer_worse_than_outside_offer = (p1_value < outside_offer)
                    else:  # current_player == 2
                        outside_offer = game.outside_offer_values[1] * (game.gamma ** current_round)
                        making_offer_worse_than_outside_offer = (p2_value < outside_offer)
                else:
                    making_offer_worse_than_outside_offer = False

                # ------------------------------
                # Pathology #3: Offer has no items or all items
                # ------------------------------
                if action == "COUNTEROFFER":
                    if (np.array_equal(game.current_offer.offer, [0] * game.num_items) or
                            np.array_equal(game.current_offer.offer, game.items)):
                        offer_no_items_or_all_items = True

                # ------------------------------
                # Compute Nash welfare
                # ------------------------------
                if action in ("ACCEPT", "COUNTEROFFER"):
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
                # Utility proportion for the round (x1_prop, x2_prop)
                # ------------------------------------------------------------
                if "WALK" in action:
                    x1_prop = game.outside_offer_values[0]
                    x2_prop = game.outside_offer_values[1]
                else:
                    # Normalize by total utility for that player + item draws
                    x1_prop = (
                        np.dot(game.player_values[0], p1_kept)
                        / sum([np.dot(game.player_values[0], p1_kept),
                            np.dot(game.player_values[0], game.num_items)])
                    )
                    x2_prop = (
                        np.dot(game.player_values[1], p2_kept)
                        / sum([np.dot(game.player_values[1], p2_kept),
                            np.dot(game.player_values[1], game.num_items)])
                    )

                # ------------------------------------------------------------
                # Concession size
                # ------------------------------------------------------------
                concession_size = None
                if action == "COUNTEROFFER" and game.current_round > 1:
                    if current_player == 1:
                        old_value = p1_offers_utility[-1] if p1_offers_utility else np.dot(game.player_values[0], p1_kept)
                        new_value = p1_offers_utility[-1]
                        concession_size = max(0, old_value - new_value)
                        p1_offers_utility.append(new_value)
                    else:
                        old_value = p2_offers_utility[-1] if p2_offers_utility else np.dot(game.player_values[1], p2_kept)
                        new_value = np.dot(game.player_values[1], p2_kept)
                        concession_size = max(0, old_value - new_value)
                        p2_offers_utility.append(new_value)
                else:
                    if action == "COUNTEROFFER":
                        if current_player == 1 and not p1_offers_utility:
                            p1_offers_utility.append(np.dot(game.player_values[0], p1_kept))
                        elif current_player == 2 and not p2_offers_utility:
                            p2_offers_utility.append(np.dot(game.player_values[1], p2_kept))

                # ------------------------------------------------------------
                # Proportion of collective value for each player
                # ------------------------------------------------------------
                if "WALK" in action:
                    proportion_of_value_p1 = None
                    proportion_of_value_p2 = None
                else:
                    denom = (np.dot(game.player_values[0], p1_kept)
                            + np.dot(game.player_values[1], p2_kept))
                    if denom > 0:
                        proportion_of_value_p1 = np.dot(game.player_values[0], p1_kept) / denom
                        proportion_of_value_p2 = np.dot(game.player_values[1], p2_kept) / denom
                    else:
                        proportion_of_value_p1 = 0.0
                        proportion_of_value_p2 = 0.0

                # ------------------------------------------------------------
                # Proportional share of the maximum available utility
                # ------------------------------------------------------------
                if "WALK" in action:
                    proposal_proportion_player_1_available_items = None
                    proposal_proportion_player_2_available_items = None
                else:
                    total_p1_available_items = np.dot(game.player_values[0], game.items)
                    total_p2_available_items = np.dot(game.player_values[1], game.items)

                    proposal_proportion_player_1_available_items = (
                        np.dot(game.player_values[0], p1_kept) / total_p1_available_items
                    ) if total_p1_available_items else 0.0

                    proposal_proportion_player_2_available_items = (
                        np.dot(game.player_values[1], p2_kept) / total_p2_available_items
                    ) if total_p2_available_items else 0.0

                # ------------------------------------------------------------
                # Utilitarian & Rawlsian welfare
                # ------------------------------------------------------------
                if "WALK" in action:
                    utilitarian_welfare = sum(game.outside_offer_values)
                    rawlsian_welfare = min(game.outside_offer_values[0], game.outside_offer_values[1])
                else:
                    utilitarian_welfare = (
                        np.dot(game.player_values[0], p1_kept)
                        + np.dot(game.player_values[1], p2_kept)
                    )
                    rawlsian_welfare = min(
                        np.dot(game.player_values[0], p1_kept),
                        np.dot(game.player_values[1], p2_kept)
                    )

                # ------------------------------------------------------------
                # Gini Coefficient (n=2)
                # ------------------------------------------------------------
                if "WALK" in action:
                    wealth_distribution = np.array([
                        game.outside_offer_values[0],
                        game.outside_offer_values[1]
                    ], dtype=float)
                else:
                    wealth_distribution = np.array([
                        np.dot(p1_kept, game.player_values[0]),
                        np.dot(p2_kept, game.player_values[1])
                    ], dtype=float)

                total_wealth = wealth_distribution.sum()
                jain_fairness_index = None 
                if total_wealth > 0:
                    x1, x2 = wealth_distribution
                    numerator = abs(x1 - x2)
                    denominator = 4.0 * total_wealth
                    gini_coefficient = numerator / denominator
                else:
                    gini_coefficient = 0.0

                    # ------------------------------------------------------------
                    # Jain's Fairness Index Calculation
                    # ------------------------------------------------------------

                    # Calculate x1 and x2
                    if "WALK" in action:
                        x1 = game.outside_offer_values[0]
                        x2 = game.outside_offer_values[1]
                    else:
                        x1 = np.dot(p1_kept, game.player_values[0])
                        x2 = np.dot(p2_kept, game.player_values[1])

                    # Mean Utility
                    mean_utility = (x1 + x2) / 2.0

                    # Variance
                    variance_value = (x1**2 + x2**2) / 2.0 - mean_utility**2
                    if variance_value < 0:
                        variance_value = 0.0  # Correct for possible negative variance due to precision

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
                if "WALK" in action:
                    security_level_player_1 = 0
                    security_level_player_2 = 0
                else:
                    security_level_player_1 = max(
                        0, game.outside_offer_values[0] - np.dot(game.player_values[0], p1_kept)
                    )
                    security_level_player_2 = max(
                        0, game.outside_offer_values[1] - np.dot(game.player_values[1], p2_kept)
                    )

                # ------------------------------------------------------------
                # On Pareto Frontier?
                # ------------------------------------------------------------
                on_pareto_frontier = False
                for vals in pareto_front:
                    if vals["type"] == "outside_offer":
                        # A WALK was played and the outside offers lie on the frontier
                        if "WALK" in game.players[current_player - 1].action:
                            on_pareto_frontier = True
                            break
                    elif vals["type"] == "allocation":
                        if (np.array_equal(vals["agent1"], p1_kept)
                            and np.array_equal(vals["agent2"], p2_kept)):
                            on_pareto_frontier = True
                            break

                # ------------------------------------------------------------
                # Mean Absolute Difference (MAD)
                # ------------------------------------------------------------
                if "WALK" in action:
                    x1 = game.outside_offer_values[0]
                    x2 = game.outside_offer_values[1]
                else:
                    x1 = np.dot(game.player_values[0], p1_kept)
                    x2 = np.dot(game.player_values[1], p2_kept)

                if x1 == 0.0 and x2 == 0.0:
                    mean_absolute_difference = 0.0
                else:
                    mean_absolute_difference = abs(x1 - x2) / 2.0

                # ------------------------------------------------------------
                # Record step in the DataFrame
                # ------------------------------------------------------------
                new_row = pd.DataFrame([{
                    "game_num": i,
                    "round_num": current_round,
                    "player": current_player,
                    "discount_rate": game.gamma ** (current_round - 1),
                    "offer": game.current_offer.offer,
                    "value": (p1_value if current_player == 1 else p2_value),
                    "undiscounted_value": (
                        p1_value / (game.gamma ** (current_round - 1))
                        if current_player == 1
                        else p2_value / (game.gamma ** (current_round - 1))
                    ),
                    "items": game.items,
                    "your_side_of_current_offer": your_side_of_current_offer,
                    "outside_offer": (
                        outside_offer if action == "COUNTEROFFER" else None
                    ),
                    "outside_offer_undiscounted": game.outside_offer_values[current_player - 1],
                    "accepting_an_offer_worse_than_outside_offer":
                        accepting_an_offer_worse_than_outside_offer,
                    "making_an_offer_worse_for_you_than_your_outside_offer":
                        making_offer_worse_than_outside_offer,
                    "walking_away_from_an_offer_better_than_your_outside_offer":
                        walking_away_from_an_offer_better_than_outside_offer,
                    "offer_no_items_or_all_items": offer_no_items_or_all_items,
                    "making_offer_worse_than_previous": making_offer_worse_than_previous,
                    "nash_welfare": nash_welfare,
                    "proposal_proportion_player_1": x1_prop,
                    "proposal_proportion_player_2": x2_prop,
                    "concession_size": concession_size,
                    "security_level_player_1": security_level_player_1,
                    "security_level_player_2": security_level_player_2,
                    "average_concession_size": None,  # TODO: compute post-game
                    "rawlsian_welfare": rawlsian_welfare,
                    "gini_coefficient": gini_coefficient,
                    "utilitarian_welfare": utilitarian_welfare,
                    "jain_fairness_index": jain_fairness_index,
                    "on_pareto_frontier": on_pareto_frontier,
                    "mean_absolute_difference": mean_absolute_difference,
                    "walk_type": walk_away_type
                }])
                game.game_results = pd.concat([game.game_results, new_row], ignore_index=True)

            # ---------------------------------------------------------
            # Add the current offer to the game history if complete
            # ---------------------------------------------------------
            if game.current_offer is not None and len(game.current_offer.offer) == game.num_items:
                game_history.add_offer(
                    game.current_player,
                    Offer(game.current_player, offer=torch.tensor(game.current_offer.offer))
                )

            # ---------------------------------------------------------
            # Check final-round logic (Player 2's turn on last round)
            # ---------------------------------------------------------
            if current_round == game.max_rounds and current_player == 2:
                # Handle acceptance/walk/counter in final round
                p1_kept, p2_kept, p1_values, p2_values, accepting_an_offer_worse_than_outside_offer = (
                    handle_final_round(
                        i,
                        current_round,
                        current_player,
                        game,
                        prev_offer,
                        p1_kept,
                        p2_kept,
                        p1_values,
                        p2_values,
                        p1_offers,
                        accepting_an_offer_worse_than_outside_offer
                    )
                )
                break

            # ---------------------------------------------------------
            # Check if Player 1 walked away (offer is None + "WALK")
            # ---------------------------------------------------------
            elif (game.current_offer is None
                and current_player == 1
                and "WALK" in game.players[current_player - 1].action):
                print("Player 1 walked")
                # Record a final row for the walk
                new_row = pd.DataFrame([{
                    "game_num": i,
                    "round_num": current_round,
                    "player": current_player,
                    "discount_rate": game.gamma ** (current_round - 1),
                    "offer": [],
                    "value": None,
                    "items": game.items,
                    "your_side_of_current_offer": None,
                    "outside_offer": None,
                    "accepting_an_offer_worse_than_outside_offer":
                        accepting_an_offer_worse_than_outside_offer,
                    "making_an_offer_worse_for_you_than_your_outside_offer":
                        making_offer_worse_than_outside_offer,
                    "walking_away_from_an_offer_better_than_your_outside_offer":
                        walking_away_from_an_offer_better_than_outside_offer,
                    "offer_no_items_or_all_items": offer_no_items_or_all_items,
                    "making_offer_worse_than_previous": making_offer_worse_than_previous,
                    "security_level_player_1": 0,
                    "security_level_player_2": 0,
                    "average_concession_size": None,
                    "nash_welfare": sqrt(prod([
                        game.outside_offer_values[0],
                        game.outside_offer_values[1]
                    ])),
                    "utilitarian_welfare":
                        game.outside_offer_values[0] + game.outside_offer_values[1],
                    "rawlsian_welfare":
                        min(game.outside_offer_values[0], game.outside_offer_values[1]),
                    "gini_coefficient": (
                        abs(game.outside_offer_values[0] - game.outside_offer_values[1]) /
                        (4.0 * sum([game.outside_offer_values[0], game.outside_offer_values[1]]))
                    ) if sum(game.outside_offer_values) > 0 else 0.0,
                    "security_level_player_1": 0,
                    "security_level_player_2": 0,
                    "jain_fairness_index": None,
                    "proposal_proportion_player_1": None,
                    "proposal_proportion_player_2": None,
                    "concession_size": None,
                    "on_pareto_frontier": False,
                    "mean_absolute_difference":
                        abs(game.outside_offer_values[0] - game.outside_offer_values[1]) / 2.0,
                    "walk_type": walk_away_type
                }])
                game.game_results = pd.concat([game.game_results, new_row], ignore_index=True)
                game.in_progress = False

        pathology_results = pd.concat([pathology_results, game.game_results], ignore_index=True)
        envy_results_history[i] = game_history

    # ------------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------------
    pathology_results.to_csv(
        f'pathology_results_{date}_{games}_{game_title}_circle_{circle}.csv',
        index=False
    )

    # If you want to save envy_results_history as well:
    # pd.DataFrame.from_dict(envy_results_history, orient='index').to_csv(
    #     f'envy_results_{date}_{games}_{prompt_style}.csv', 
    #     index=False
    # )
