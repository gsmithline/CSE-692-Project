#!/usr/bin/env python
# coding: utf-8

import json
import numpy as np
import os
import pandas as pd
from collections import defaultdict
import math
import random
from itertools import product

# Default discount factor if not specified
DEFAULT_DISCOUNT_FACTOR = 0.98

def parse_values(line):
    """Parse the line containing private values"""
    # Format: "Your private values are 16 for item 1, 46 for item 2, 100 for item 3, 23 for item 4, 47 for item 5."
    values = []
    parts = line.split("are ")[1].split(", ")
    for part in parts:
        value = int(part.split(" for item")[0])
        values.append(value)
    return values

def parse_outside_offer(line):
    """Parse the line containing outside offer"""
    # Format: "Your outside offer value is 145. Your objective..."
    return int(line.split("value is ")[1].split(".")[0])

def calculate_value(items_received, values):
    """Calculate value of items received given the player's values"""
    return sum(items_received[i] * values[i] for i in range(len(values)))

def compute_max_nash_welfare(item_counts, p1_valuations, p2_valuations):
    """Compute the maximum Nash welfare for a given set of item counts and valuations."""
    if len(item_counts) != len(p1_valuations) or len(item_counts) != len(p2_valuations):
        raise ValueError("item_counts, p1_valuations, p2_valuations must have the same length.")

    K = len(item_counts)
    max_nash = -1.0
    best_alloc = None
    outside_offer_player1 = np.random.randint(1, np.dot(item_counts, p1_valuations))
    outside_offer_player2 = np.random.randint(1, np.dot(item_counts, p2_valuations))
    ranges = [range(n_i + 1) for n_i in item_counts] 
    for allocation in product(*ranges):
        p1_util = 0.0
        p2_util = 0.0
        for i in range(K):
            x_i = allocation[i]
            n_i = item_counts[i]
            p1_util += x_i * p1_valuations[i]
            p2_util += (n_i - x_i) * p2_valuations[i]

        w = math.sqrt(max(p1_util, 0) * max(p2_util, 0))

        if w > max_nash:
            max_nash = w
            best_alloc = allocation

        #outside offer check
        if max_nash < math.sqrt(outside_offer_player1 * outside_offer_player2):
            max_nash = math.sqrt(outside_offer_player1 * outside_offer_player2)
            best_alloc = [0, 0, 0, 0, 0]

    return max_nash, list(best_alloc)

def analyze_single_game(file_path, discount_factor=DEFAULT_DISCOUNT_FACTOR):
    """
    Analyze a single game JSON file and extract relevant metrics.
    
    Args:
        file_path: Path to the JSON file containing the game data
        discount_factor: Discount factor to apply to utilities (default: DEFAULT_DISCOUNT_FACTOR)
        
    Returns:
        List of dictionaries containing analyzed game metrics
    """
    with open(file_path, 'r') as f:
        data = json.load(f)

    results = []
    for game in data['all_game_data']:
        agent1 = game['agent1']
        agent2 = game['agent2']
        
        # Remove Agent1_ and Agent2_ prefixes if they exist
        if agent1.startswith("Agent1_"):
            agent1 = agent1[7:]
        if agent2.startswith("Agent2_"):
            agent2 = agent2[7:]
        
        # Track the final state
        final_action = None
        final_round_index = len(game['round_data']) - 1
        # Convert to actual round number (2 turns = 1 round)
        final_round_number = (final_round_index // 2) + 1
        final_offer = None
        p1_outside_offer = None
        p2_outside_offer = None
        p1_values = None
        p2_values = None
        num_items = None
        full_items = None

        # First, gather all the necessary data from all rounds
        for round_idx, round_data in enumerate(game['round_data']):
            prompt = round_data['prompt']
            
            # Check if this is the final turn
            if round_idx == final_round_index:
                final_action = round_data['action']
            
            if "You are Player 1" in prompt:
                # Extract Player 1's values and outside offer
                for line in prompt.split('\n'):
                    if "Your private values are" in line:
                        p1_values = parse_values(line)
                    elif "Your outside offer value is" in line:
                        p1_outside_offer = parse_outside_offer(line)
                    elif "There are" in line and "units of item" in line:
                        if num_items is None:
                            # Extract numbers before "units of item"
                            nums = []
                            parts = line.split("There are ")[1].split(", ")
                            for part in parts:
                                num = int(part.split(" units")[0])
                                nums.append(num)
                            num_items = nums
            elif "You are Player 2" in prompt:
                # Extract Player 2's values and outside offer
                for line in prompt.split('\n'):
                    if "Your private values are" in line:
                        p2_values = parse_values(line)
                    elif "Your outside offer value is" in line:
                        p2_outside_offer = parse_outside_offer(line)
            
            # Track the current offer
            if "Current offer on the table" in prompt:
                offer_line = prompt.split("Current offer on the table")[1].split("\n")[0]
                if "None" not in offer_line and "[" in offer_line:
                    final_offer = eval(offer_line[offer_line.find("["):offer_line.find("]")+1])
        
        # Calculate final values
        p1_final_value = None
        p2_final_value = None
        
        # Calculate discount factor based on the round number
        round_discount = discount_factor ** (final_round_number - 1)
        p1_items = None
        p2_items = None
        if final_action == "WALK" or final_action == "INVALID WALK":
            # Check if this is player 1's first action
            if final_round_number == 1 and "You are Player 1" in game['round_data'][final_round_index]['prompt']:
                # Don't count the game if player 1 walks on their first action
                p1_final_value = None
                p2_final_value = None
            else:
                p1_final_value = p1_outside_offer * round_discount
                p2_final_value = p2_outside_offer * round_discount
        elif final_action == "ACCEPT":
            # Determine which player accepted the offer
            accepting_player = None
            for round_idx, round_data in enumerate(game['round_data']):
                if round_idx == final_round_index and round_data['action'] == "ACCEPT":
                    accepting_player = 1 if "You are Player 1" in round_data['prompt'] else 2
            
            if accepting_player == 2:
                # Player 2 accepted the offer
                # The offer represents what Player 2 receives
                p2_items = final_offer
                p1_items = [num_items[i] - final_offer[i] for i in range(len(final_offer))]
            else:
                # Player 1 accepted the offer
                # The offer represents what Player 1 receives
                p1_items = final_offer
                p2_items = [num_items[i] - final_offer[i] for i in range(len(final_offer))]
            
            p1_final_value = calculate_value(p1_items, p1_values) * round_discount
            p2_final_value = calculate_value(p2_items, p2_values) * round_discount

        elif final_action == "INVALID WALK":
            print(f"Invalid walk in game {game['game_id']}")
            
        try:
            circle_data = data['all_game_data'][0]['circle']
            if isinstance(circle_data, int):
                p1_circle = circle_data
                p2_circle = circle_data
            else:
                p1_circle = circle_data[0]
                p2_circle = circle_data[1]
        except (KeyError, TypeError, IndexError):
            # Handle missing or malformed data
            p1_circle = None
            p2_circle = None
        
        # Add circle values to agent names if available
        agent1_with_circle = f"{agent1}_circle_{p1_circle}" if p1_circle is not None else agent1
        agent2_with_circle = f"{agent2}_circle_{p2_circle}" if p2_circle is not None else agent2
      
        results.append({
            'agent1': agent1_with_circle,
            'agent2': agent2_with_circle,
            'final_action': final_action,
            'final_round': final_round_number,
            'discount_factor': round_discount,
            'agent1_value': p1_final_value,
            'agent2_value': p2_final_value,
            'p1_values': p1_values,
            'p2_values': p2_values,
            'p1_items': p1_items,
            'p2_items': p2_items,
            'full_items': num_items,
            'num_items': len(p1_values) if p1_values else None,
            'p1_outside_offer': p1_outside_offer,
            'p2_outside_offer': p2_outside_offer
        })
    
    return results

def get_canonical_name(agent_name):
    """Convert agent names to canonical format for data loading"""
    # Define bidirectional mappings with dated and clean versions
    model_mapping = {
        # Map from original form to dated form (for data loading)
        'openai_4o_circle_4': 'openai_4o_2024-08-06_circle_4',
        'openai_4o_circle_5': 'openai_4o_2024-08-06_circle_5',
        'openai_4o_circle_6': 'openai_4o_2024-08-06_circle_6',
        
        'anthropic_3.7_sonnet_circle_5': 'anthropic_3.7_sonnet_2025-02-19_circle_5',
        'anthropic_3.7_sonnet_circle_6': 'anthropic_3.7_sonnet_2025-02-19_circle_6',
        
        'openai_o3_mini_circle_0': 'openai_o3_mini_2025-01-31_circle_0',
    }
    
    return model_mapping.get(agent_name, agent_name)

def get_display_name(agent_name):
    """Convert dated agent names to clean display names without dates"""
    # Define bidirectional mappings with dated and clean versions
    model_mapping = {
        # Map from original form to dated form (for data loading)
        'openai_4o_circle_4': 'openai_4o_2024-08-06_circle_4',
        'openai_4o_circle_5': 'openai_4o_2024-08-06_circle_5',
        'openai_4o_circle_6': 'openai_4o_2024-08-06_circle_6',
        
        'anthropic_3.7_sonnet_circle_5': 'anthropic_3.7_sonnet_2025-02-19_circle_5',
        'anthropic_3.7_sonnet_circle_6': 'anthropic_3.7_sonnet_2025-02-19_circle_6',
        
        'openai_o3_mini_circle_0': 'openai_o3_mini_2025-01-31_circle_0',
    }
    
    reverse_model_mapping = {v: k for k, v in model_mapping.items()}
    
    if agent_name in reverse_model_mapping:
        return reverse_model_mapping[agent_name]
    
    import re
    date_pattern = r'_\d{4}-\d{2}-\d{2}'
    if re.search(date_pattern, agent_name):
        return re.sub(date_pattern, '', agent_name)
    
    return agent_name

def process_all_games(crossplay_dir="crossplay/game_matrix_2", discount_factor=DEFAULT_DISCOUNT_FACTOR):
    """
    Process all game data files in the specified directory and return compiled results.
    
    Args:
        crossplay_dir: Directory containing game data files
        discount_factor: Discount factor to apply to utilities (default: DEFAULT_DISCOUNT_FACTOR)
        
    Returns:
        tuple: (all_results, agent_performance, agent_final_rounds, agent_game_counts, agent_final_rounds_self_play)
    """
    all_results = []
    agent_performance = defaultdict(lambda: defaultdict(list))
    agent_final_rounds = defaultdict(list) 
    agent_game_counts = defaultdict(lambda: defaultdict(int))  
    agent_final_rounds_self_play = defaultdict(list)

    for root, dirs, files in os.walk(crossplay_dir):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                try:
                    game_results = analyze_single_game(file_path, discount_factor)
                    
                    # Skip some specific agents 
                    if any('openai_4o_circle_2' in result['agent1'] or 'openai_4o_circle_2' in result['agent2'] 
                          for result in game_results):
                        print(f"Skipping {file_path} - contains openai_4o_circle_2")
                        continue
                    if any('gemini_2.0_flash_circle_6' in result['agent1'] or 'gemini_2.0_flash_circle_6' in result['agent2'] 
                          for result in game_results):
                        print(f"Skipping {file_path} - contains gemini_2.0_flash_circle_6")
                        continue
                    if any('anthropic_3.7_sonnet_circle_4' in result['agent1'] or 'anthropic_3.7_sonnet_circle_4' in result['agent2'] 
                          for result in game_results):
                        print(f"Skipping {file_path} - contains anthropic_3.7_sonnet_circle_4")
                        continue
                    
                    for result in game_results:
                        result['agent1'] = get_canonical_name(result['agent1'])
                        result['agent2'] = get_canonical_name(result['agent2'])
                    
                    all_results.extend(game_results)
                    
                    for result in game_results:
                        agent1 = result['agent1'] 
                        agent2 = result['agent2']
                        
                        if result['agent1_value'] is not None:
                            agent_performance[agent1][agent2].append(result['agent1_value'])
                            agent_game_counts[agent1][agent2] += 1  
                        
                        if result['agent2_value'] is not None:
                            agent_performance[agent2][agent1].append(result['agent2_value'])
                            agent_game_counts[agent2][agent1] += 1  
                        
                        # Track final rounds for each agent
                        if result['final_round'] is not None:
                            agent_final_rounds[agent1].append(result['final_round'])
                            agent_final_rounds[agent2].append(result['final_round'])

                        if agent1 == agent2 or ("openai_o3" in agent1 and "openai_o3" in agent2):
                            agent_final_rounds_self_play[agent1].append(result['final_round'])
                    
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

    return all_results, agent_performance, agent_final_rounds, agent_game_counts, agent_final_rounds_self_play

def compute_global_max_values(num_samples=1000000):
    """Compute global maximum values for Nash welfare and standard welfare for comparison."""
    global_max_nash_welfare = []
    global_standard_max = []
    
    for i in range(num_samples):
        items = np.random.poisson(4, 5)

        player_values1 = np.random.randint(1, 101, 5) 
        player_values2 = np.random.randint(1, 101, 5)
      
        max_nash, _ = compute_max_nash_welfare(items, player_values1, player_values2)
        global_max_nash_welfare.append(max_nash)
        
        global_standard_max.append(np.dot(items, player_values1))

    return np.mean(global_max_nash_welfare), np.mean(global_standard_max) 