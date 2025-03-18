#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from collections import defaultdict
import math
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval.metrics import compute_pareto_frontier

def create_performance_matrices(all_results, agent_performance, agent_final_rounds):
    """
    Create performance matrices from the processed game results.
    
    Args:
        all_results: List of all game results
        agent_performance: Dictionary of performance by agent
        agent_final_rounds: Dictionary of final rounds by agent
        
    Returns:
        tuple: Dictionaries and DataFrames containing performance metrics
    """
    # First, calculate some basic statistics
    agent_vs_opponent_performance = {}
    for agent, opponents in agent_performance.items():
        for opponent, values in opponents.items():
            if values:
                key = f"{agent}_vs_{opponent}"
                agent_vs_opponent_performance[key] = np.mean(values)

    overall_agent_performance = {}
    for agent, opponents in agent_performance.items():
        all_values = []
        for values in opponents.values():
            all_values.extend(values)
        if all_values:
            overall_agent_performance[agent] = np.mean(all_values)

    average_final_rounds = {}
    average_final_rounds_self_play = {}
    for agent, rounds in agent_final_rounds.items():
        if rounds:
            average_final_rounds[agent] = np.mean(rounds)

    # Create performance matrices
    all_agents = sorted(list(overall_agent_performance.keys()))
    performance_matrix = pd.DataFrame(index=all_agents, columns=all_agents)
    std_dev_matrix = pd.DataFrame(index=all_agents, columns=all_agents)
    variance_matrix = pd.DataFrame(index=all_agents, columns=all_agents)
    count_matrix = pd.DataFrame(index=all_agents, columns=all_agents)

    for agent, opponents in agent_performance.items():
        for opponent, values in opponents.items():
            if values:
                performance_matrix.loc[agent, opponent] = np.mean(values)
                std_dev_matrix.loc[agent, opponent] = np.std(values)
                variance_matrix.loc[agent, opponent] = np.nanvar(values)
                count_matrix.loc[agent, opponent] = len(values)

    # Create scaled performance matrix
    scaled_performance_matrix = pd.DataFrame(index=all_agents, columns=all_agents)
    for agent in all_agents:
        for opponent in all_agents:
            scaled_values = []

            # When agent is Player 1
            for result in all_results:
                if (result['agent1'] == agent and result['agent2'] == opponent and 
                    result['agent1_value'] is not None and result['p1_values'] is not None):
                    max_possible = max(
                        sum(result['full_items'][i] * result['p1_values'][i] for i in range(len(result['p1_values']))),
                        result['p1_outside_offer']
                    )
                    if max_possible > 0:
                        scaled_values.append(result['agent1_value'] / max_possible)

                # When agent is Player 2
                if (result['agent2'] == agent and result['agent1'] == opponent and 
                    result['agent2_value'] is not None and result['p2_values'] is not None):
                    if 'p2_outside_offer' in result and result['p2_outside_offer'] is not None:
                        max_possible = max(
                            sum(result['full_items'][i] * result['p2_values'][i] for i in range(len(result['p2_values']))),
                            result['p2_outside_offer']
                        )
                    else:
                        max_possible = sum(result['full_items'][i] * result['p2_values'][i] for i in range(len(result['p2_values'])))
                    if max_possible > 0:
                        scaled_values.append(result['agent2_value'] / max_possible)


            if scaled_values:
                scaled_performance_matrix.loc[agent, opponent] = np.mean(scaled_values)
    
    return {
        'performance_matrix': performance_matrix,
        'std_dev_matrix': std_dev_matrix,
        'variance_matrix': variance_matrix,
        'scaled_performance_matrix': scaled_performance_matrix,
        'count_matrix': count_matrix,
        'overall_agent_performance': overall_agent_performance,
        'agent_vs_opponent_performance': agent_vs_opponent_performance,
        'average_final_rounds': average_final_rounds,
    }

def create_welfare_matrices(all_results, all_agents, global_max_nash_welfare):
    """
    Create matrices for various welfare and fairness metrics.
    
    Args:
        all_results: List of all game results
        all_agents: List of all agents
        global_max_nash_welfare: Global maximum Nash welfare for normalization
        
    Returns:
        dict: Dictionary of welfare matrices
    """
    # Initialize matrices
    nash_welfare_matrix = pd.DataFrame(index=all_agents, columns=all_agents)
    utilitarian_welfare_matrix = pd.DataFrame(index=all_agents, columns=all_agents)
    rawls_welfare_matrix = pd.DataFrame(index=all_agents, columns=all_agents)
    mad_matrix = pd.DataFrame(index=all_agents, columns=all_agents)
    gini_matrix = pd.DataFrame(index=all_agents, columns=all_agents)
    variance_welfare_matrix = pd.DataFrame(index=all_agents, columns=all_agents)
    cv_matrix = pd.DataFrame(index=all_agents, columns=all_agents)
    jain_matrix = pd.DataFrame(index=all_agents, columns=all_agents)
    envy_free_matrix = pd.DataFrame(index=all_agents, columns=all_agents)
    ef1_matrix = pd.DataFrame(index=all_agents, columns=all_agents)
    pareto_matrix = pd.DataFrame(index=all_agents, columns=all_agents)

    # Set all matrix values to NaN initially
    for matrix in [
        nash_welfare_matrix, utilitarian_welfare_matrix, rawls_welfare_matrix, 
        mad_matrix, gini_matrix, variance_welfare_matrix, cv_matrix, 
        jain_matrix, envy_free_matrix, ef1_matrix, pareto_matrix
    ]:
        matrix.values[:] = np.nan 

    processed_pairs = set()

    for agent1_idx, agent1 in enumerate(all_agents):
        for agent2_idx, agent2 in enumerate(all_agents):
            # For welfare metrics, only process each unique pair once
            # Skip if we've already processed this pair (in reverse order)
            pair_key = tuple(sorted([agent1, agent2]))
            
            if pair_key in processed_pairs and agent1 != agent2:
                continue
                
            processed_pairs.add(pair_key)
            
            agent1_vs_agent2_games = [result for result in all_results 
                                   if ((result['agent1'] == agent1 and result['agent2'] == agent2) or 
                                       (result['agent1'] == agent2 and result['agent2'] == agent1))]
            
            nash_values = []
            utilitarian_values = []
            rawls_values = []
            mad_values = []
            gini_values = []
            variance_values = []
            cv_values = []
            jain_values = []
            
            envy_free_count = 0
            ef1_count = 0
            pareto_count = 0
            valid_allocation_count = 0
            
            for game in agent1_vs_agent2_games:
                if game['agent1_value'] is not None and game['agent2_value'] is not None:
                    if game['agent1'] == agent1 and game['agent2'] == agent2:
                        v1 = game['agent1_value']
                        v2 = game['agent2_value']
                        p1_items = game['p1_items']
                        p2_items = game['p2_items']
                        p1_values = game['p1_values']
                        p2_values = game['p2_values']
                        is_on_pareto = game.get('PF', False)
                    else:
                        v2 = game['agent1_value']
                        v1 = game['agent2_value']
                        p2_items = game['p1_items']
                        p1_items = game['p2_items']
                        p2_values = game['p1_values']
                        p1_values = game['p2_values']
                        is_on_pareto = game.get('PF', False)
                    
                    mean_utility = (v1 + v2) / 2
                    
                    nash_welfare = np.sqrt(v1 * v2) / global_max_nash_welfare
                    
                    utilitarian_welfare = v1 + v2
                    
                    rawls_welfare = min(v1, v2)
                    
                    mad = abs(v1 - v2)
                    
                    gini = abs(v1 - v2) / (2 * 2 * mean_utility) if mean_utility > 0 else 0
                    
                    variance = ((v1 - mean_utility)**2 + (v2 - mean_utility)**2) / 2
                    
                    cv = np.sqrt(variance) / mean_utility if mean_utility > 0 else 0
                    
                    jain = 1 / (1 + cv**2) if cv > 0 else 1
                    
                    nash_values.append(nash_welfare)
                    utilitarian_values.append(utilitarian_welfare)
                    rawls_values.append(rawls_welfare)
                    mad_values.append(mad)
                    gini_values.append(gini)
                    variance_values.append(variance)
                    cv_values.append(cv)
                    jain_values.append(jain)
                    
                    valid_allocation_count += 1
                    
                    if is_on_pareto:
                        pareto_count += 1
                    elif game['final_action'] == "ACCEPT" and p1_items is not None and p2_items is not None and game['num_items'] is not None:
                        try:
                            full_items = game.get('full_items', game['num_items'])
                            
                            p1_values_np = np.array(p1_values)
                            p2_values_np = np.array(p2_values)
                            num_items = len(p1_values)
                            items_np = np.array(full_items)
                            outside_offers = [game.get('p1_outside_offer', np.dot(p1_values_np, items_np)), game.get('p2_outside_offer', np.dot(p2_values_np, full_items))]

                            
                            frontier = compute_pareto_frontier(p1_values_np, p2_values_np, num_items, items_np, outside_offers)
                            
                            is_pareto_optimal = False
                            for alloc in frontier:
                                if alloc['type'] == 'allocation':
                                    if alloc['agent1'] == p1_items and alloc['agent2'] == p2_items:
                                        is_pareto_optimal = True
                                        break
                                elif alloc['type'] == 'outside_offer':
                                    if game['final_action'] == "WALK" and abs(alloc['agent1_value'] - v1) < 0.01 and abs(alloc['agent2_value'] - v2) < 0.01:
                                        is_pareto_optimal = True
                                        break
                                        
                            if is_pareto_optimal:
                                pareto_count += 1
                                
                        except Exception as e:
                            print(f"Error calculating Pareto optimality: {e}")
                    
                    if game['final_action'] == "ACCEPT" and p1_items is not None and p2_items is not None and p1_values is not None and p2_values is not None:
                        
                        p1_own_bundle_value = sum(p1_values[i] * p1_items[i] for i in range(len(p1_values)))
                        
                        p1_other_bundle_value = sum(p1_values[i] * p2_items[i] for i in range(len(p1_values)))
                        
                        p1_is_envy_free = p1_own_bundle_value >= p1_other_bundle_value
                        
                        p2_own_bundle_value = sum(p2_values[i] * p2_items[i] for i in range(len(p2_values)))
                        
                        p2_other_bundle_value = sum(p2_values[i] * p1_items[i] for i in range(len(p2_values)))
                        
                        p2_is_envy_free = p2_own_bundle_value >= p2_other_bundle_value
                        
                        if p1_is_envy_free and p2_is_envy_free:
                            envy_free_count += 1
                        
                        p1_is_ef1 = p1_is_envy_free 
                        
                        if not p1_is_envy_free:
                            for j in range(len(p1_values)):
                                if p2_items[j] > 0:
                                    adjusted_value = p1_other_bundle_value - (p1_values[j] * 1)  # Remove 1 unit
                                    if p1_own_bundle_value >= adjusted_value:
                                        p1_is_ef1 = True
                                        break
                        
                        p2_is_ef1 = p2_is_envy_free  
                        
                        if not p2_is_envy_free:
                            for j in range(len(p2_values)):
                                if p1_items[j] > 0:
                                    adjusted_value = p2_other_bundle_value - (p2_values[j] * 1)  # Remove 1 unit
                                    if p2_own_bundle_value >= adjusted_value:
                                        p2_is_ef1 = True
                                        break
                        
                        if p1_is_ef1 and p2_is_ef1:
                            ef1_count += 1
            
            if nash_values:
                welfare_value = np.mean(nash_values)
                nash_welfare_matrix.loc[agent1, agent2] = welfare_value
                nash_welfare_matrix.loc[agent2, agent1] = welfare_value
                
                welfare_value = np.mean(utilitarian_values)
                utilitarian_welfare_matrix.loc[agent1, agent2] = welfare_value
                utilitarian_welfare_matrix.loc[agent2, agent1] = welfare_value
                
                welfare_value = np.mean(rawls_values)
                rawls_welfare_matrix.loc[agent1, agent2] = welfare_value
                rawls_welfare_matrix.loc[agent2, agent1] = welfare_value
                
                welfare_value = np.mean(mad_values)
                mad_matrix.loc[agent1, agent2] = welfare_value
                mad_matrix.loc[agent2, agent1] = welfare_value
                
                welfare_value = np.mean(gini_values)
                gini_matrix.loc[agent1, agent2] = welfare_value
                gini_matrix.loc[agent2, agent1] = welfare_value
                
                welfare_value = np.mean(variance_values)
                variance_welfare_matrix.loc[agent1, agent2] = welfare_value
                variance_welfare_matrix.loc[agent2, agent1] = welfare_value
                
                welfare_value = np.mean(cv_values)
                cv_matrix.loc[agent1, agent2] = welfare_value
                cv_matrix.loc[agent2, agent1] = welfare_value
                
                welfare_value = np.mean(jain_values)
                jain_matrix.loc[agent1, agent2] = welfare_value
                jain_matrix.loc[agent2, agent1] = welfare_value
            
            if valid_allocation_count > 0:
                ef_value = envy_free_count / valid_allocation_count
                envy_free_matrix.loc[agent1, agent2] = ef_value
                envy_free_matrix.loc[agent2, agent1] = ef_value
                
                ef1_value = ef1_count / valid_allocation_count
                ef1_matrix.loc[agent1, agent2] = ef1_value
                ef1_matrix.loc[agent2, agent1] = ef1_value
                
                pareto_value = pareto_count / valid_allocation_count
                pareto_matrix.loc[agent1, agent2] = pareto_value
                pareto_matrix.loc[agent2, agent1] = pareto_value
    
    return {
        'nash_welfare_matrix': nash_welfare_matrix,
        'utilitarian_welfare_matrix': utilitarian_welfare_matrix,
        'rawls_welfare_matrix': rawls_welfare_matrix,
        'mad_matrix': mad_matrix,
        'gini_matrix': gini_matrix,
        'variance_welfare_matrix': variance_welfare_matrix,
        'cv_matrix': cv_matrix,
        'jain_matrix': jain_matrix,
        'envy_free_matrix': envy_free_matrix,
        'ef1_matrix': ef1_matrix,
        'pareto_matrix': pareto_matrix
    }

def clean_matrix_names(matrix, get_display_name_func):
    """
    Clean matrix index and column names using the get_display_name function.
    
    Args:
        matrix: DataFrame to clean
        get_display_name_func: Function to convert agent names to display names
        
    Returns:
        DataFrame: Matrix with cleaned names
    """
    clean_matrix = matrix.copy()
    
    clean_matrix.index = [get_display_name_func(agent) for agent in matrix.index]
    clean_matrix.columns = [get_display_name_func(agent) for agent in matrix.columns]
    
    return clean_matrix

def filter_matrices(matrices, exclude_agents=None):
    """
    Filter a dictionary of matrices to exclude specified agents.
    
    Args:
        matrices: Dictionary of DataFrames
        exclude_agents: List of agents to exclude
        
    Returns:
        dict: Filtered dictionary of DataFrames
    """
    if exclude_agents is None:
        exclude_agents = []
        
    filtered_matrices = {}
    for name, matrix in matrices.items():
        filtered_matrix = matrix
        for agent in exclude_agents:
            filtered_matrix = filtered_matrix[filtered_matrix.index != agent].drop(columns=[agent], errors='ignore')
        filtered_matrices[name] = filtered_matrix
        
    return filtered_matrices 