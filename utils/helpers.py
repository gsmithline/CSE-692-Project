import numpy as np
from itertools import product
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



def calculate_discounted_value(offer, values, gamma, round_num):
    if offer is None:
        return 0
    base_value = sum(v * q for v, q in zip(values, offer))
    return base_value * (gamma ** (round_num - 1))

# ------------------------------------------------------------------------
# Helper function: Discounted value
# ------------------------------------------------------------------------
def calculate_discounted_value(offer, values, gamma, realization_round):
    """
    Returns the discounted value of `offer` for an agent with utility `values`,
    discount factor `gamma`, and the realization round number.
    """
    if offer is None:
        return 0
    base_value = sum(v * q for v, q in zip(values, offer))
    return base_value * (gamma ** (realization_round - 1))

# ------------------------------------------------------------------------
# Helper function: Detect pathology #4 (accepting worse than outside)
# ------------------------------------------------------------------------
def check_accepting_worse_than_outside(current_player, p1_kept, p2_kept, game):
    """
    PATHOLOGY 4: Accepting an offer worse than your outside offer.
    If the current player accepted, check if the portion they get
    is less than their outside offer.
    """
    accepting_worse = False
    action = game.players[current_player - 1].action
    if action == "ACCEPT":
        if current_player == 1 and p1_kept is not None:
            if np.dot(game.player_values[0], p1_kept) < game.outside_offer_values[0]:
                accepting_worse = True
        elif current_player == 2 and p2_kept is not None:
            if np.dot(game.player_values[1], p2_kept) < game.outside_offer_values[1]:
                accepting_worse = True
    return accepting_worse

# ------------------------------------------------------------------------
# Helper function: Detect pathology #5 (walking away from a better offer)
# ------------------------------------------------------------------------
def check_walking_away_from_better(current_player, p1_kept, p2_kept, game):
    """
    PATHOLOGY 5: Walking away from an offer better than your outside offer.
    """
    walking_away_better = False
    action = game.players[current_player - 1].action
    if ("WALK" in action) or (
        current_player == 2
        and action == "COUNTEROFFER"
        and game.current_round == game.max_rounds
    ):
        if current_player == 1 and p1_kept is not None:
            if np.dot(game.player_values[0], p1_kept) > game.outside_offer_values[0]:
                walking_away_better = True
        elif current_player == 2 and p2_kept is not None:
            if np.dot(game.player_values[1], p2_kept) > game.outside_offer_values[1]:
                walking_away_better = True
    return walking_away_better

# ------------------------------------------------------------------------
# Helper function: Determine validity of a WALK
# ------------------------------------------------------------------------
def determine_walk_away_type(current_player, game):
    """
    Checks if the current player's action is 'INVALID WALK' or 'WALK'
    and returns an appropriate walk_away_type. Otherwise returns None.
    """
    action = game.players[current_player - 1].action
    if "INVALID WALK" in action:
        return "INVALID"
    elif "WALK" in action:
        return "VALID"
    return None

# ------------------------------------------------------------------------
# Helper function: Update who-keeps-what (p1_kept, p2_kept)
# ------------------------------------------------------------------------
def update_kept_portions(current_player, game, p1_kept, p2_kept):
    """
    If there's a new COUNTEROFFER from the current player, update
    p1_kept and p2_kept accordingly.
    """
    action = game.players[current_player - 1].action
    if action == "COUNTEROFFER":
        if current_player == 1:
            # P1 is proposing, so P1's kept portion is whatever is left
            # and P2 is offered game.current_offer.offer
            p1_kept = game.items - np.array(game.current_offer.offer)
            p2_kept = np.array(game.current_offer.offer)
        else:  # current_player == 2
            # P2 is proposing, so P2's kept portion is whatever is left
            # and P1 is offered game.current_offer.offer
            p1_kept = np.array(game.current_offer.offer)
            p2_kept = game.items - np.array(game.current_offer.offer)
    return p1_kept, p2_kept

# ------------------------------------------------------------------------
# Helper function: Final round resolution
# ------------------------------------------------------------------------
def handle_final_round(
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
):
    """
    If we are at the final round and it's Player 2's turn, handle
    accept/counter/walk logic, finalize payoffs, print results,
    and stop the game.
    """
    # If there's a new offer and it's not the same as the previous
    # -> P2 has effectively made a final COUNTEROFFER
    if game.current_offer and game.current_offer != prev_offer:
        print(f"\nPlayer 2's final action: COUNTEROFFER {game.current_offer.offer}")
        p1_value = game.outside_offer_values[0] * (game.gamma ** (current_round - 1))
        p2_value = game.outside_offer_values[1] * (game.gamma ** (current_round - 1))
        print("\nGame ended after max rounds - both players get outside offers")

    # If there's an offer and it matches the previous offer -> P2 ACCEPT
    elif game.current_offer == prev_offer:
        print("\nPlayer 2's final action: ACCEPT")
        # P2 accepted P1's final offer
        p1_kept = game.items - np.array(game.current_offer.offer)
        p1_value = calculate_discounted_value(
            p1_kept, game.player_values[0], game.gamma, current_round
        )
        p2_value = calculate_discounted_value(
            game.current_offer.offer, game.player_values[1], game.gamma, current_round
        )
        # Check pathology #4 after final acceptance
        if game.outside_offer_values[1] > np.dot(game.player_values[1], p1_offers[-1]):
            accepting_an_offer_worse_than_outside_offer = True

    else:
        # P2 walked away
        print("\nPlayer 2's final action: WALK")
        p1_value = game.outside_offer_values[0] * (game.gamma ** (current_round - 1))
        p2_value = game.outside_offer_values[1] * (game.gamma ** (current_round - 1))
        print("\nGame ended after max rounds - both players get outside offers")

    # Mark game as ended
    game.in_progress = False
    print(f"\nRound {current_round} Final Values:")
    print(f"Player 1: {p1_value:.2f}")
    print(f"Player 2: {p2_value:.2f}")

    p1_values.append(p1_value)
    p2_values.append(p2_value)

    return p1_kept, p2_kept, p1_values, p2_values, accepting_an_offer_worse_than_outside_offer


def find_allocation_less_than_outside_offer_dp(items, player_valuations, outside_offer, player_num):
    """
    Finds the allocation that yields the highest utility strictly less than the outside_offer.
    Using dynamic programming to find the best allocation.
    """
    num_items = len(items)
    best_utility = -1.0
    best_combo = None

    quantity_ranges = [range(items[i] + 1) for i in range(num_items)]
    
    for combo in product(*quantity_ranges):
        
        total_utility = 0.0
        for i in range(num_items):
            total_utility += player_valuations[i] * combo[i]

        if total_utility < outside_offer and total_utility > best_utility:
            best_utility = total_utility
            best_combo = combo

    if best_combo is None:
        return None
    allocation = {}
    for i in range(num_items):
        allocation[i] = best_combo[i]

    return allocation