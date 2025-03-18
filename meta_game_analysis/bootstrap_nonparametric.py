"""
Non-parametric bootstrapping implementation for game theoretic analysis.
"""

import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nash_equilibrium.nash_solver import milp_max_sym_ent_2p

def nonparametric_bootstrap_from_raw_data(all_results, num_bootstrap=1000, confidence=0.95):
    """
    Proper non-parametric bootstrapping directly from raw game results
    
    Args:
        all_results: List of dictionaries, each representing a processed game outcome from data_processing.py
        num_bootstrap: Number of bootstrap replicas
        confidence: Confidence level for intervals
        
    Returns:
        Dictionary of bootstrap results with statistics and confidence intervals
    """
    print(f"Performing non-parametric bootstrapping with {num_bootstrap} samples...")
    
    # Extract all unique agents from the raw data
    all_agents = set()
    for game in all_results:
        if game['agent1'] is not None:
            all_agents.add(game['agent1'])
        if game['agent2'] is not None:
            all_agents.add(game['agent2'])
    
    all_agents = sorted(list(all_agents))
    num_agents = len(all_agents)
    print(f"Identified {num_agents} unique agents")
    
    # Initialize results
    bootstrap_results = {
        'ne_regret': [],               # Relative performance (expected_utils - nash_value)
        'traditional_regret': [],      # Traditional regret (max_utils - expected_utils)
        'ne_strategy': [],
        'agent_expected_utility': [],
        'agent_max_utility': [],
        'nash_value': [],              # Nash equilibrium value for each bootstrap
        'bootstrapped_matrices': []    # Store each bootstrap's performance matrix
    }
    
    # For each bootstrap iteration
    for b in range(num_bootstrap):
        if b % 100 == 0 and b > 0:
            print(f"Processed {b} bootstrap samples...")
        
        # 1. Generate a bootstrap sample by resampling ENTIRE GAMES with replacement
        bootstrap_indices = np.random.choice(
            range(len(all_results)), 
            size=len(all_results), 
            replace=True
        )
        
        # Get the resampled games
        resampled_games = [all_results[i] for i in bootstrap_indices]
        
        # 2. Compute performance matrix from this bootstrap sample
        agent_performance = defaultdict(lambda: defaultdict(list))
        
        # Process all resampled games
        for game in resampled_games:
            agent1 = game['agent1']
            agent2 = game['agent2']
            
            # Record agent1's performance against agent2
            if agent1 and agent2 and 'agent1_value' in game and game['agent1_value'] is not None:
                agent_performance[agent1][agent2].append(game['agent1_value'])
                
            # Record agent2's performance against agent1
            if agent1 and agent2 and 'agent2_value' in game and game['agent2_value'] is not None:
                agent_performance[agent2][agent1].append(game['agent2_value'])
        
        # Create performance matrix
        performance_matrix = pd.DataFrame(index=all_agents, columns=all_agents)
        
        # Fill the matrix with average performance values
        for agent1 in all_agents:
            for agent2 in all_agents:
                values = agent_performance[agent1][agent2]
                if values:
                    performance_matrix.loc[agent1, agent2] = np.mean(values)
                else:
                    performance_matrix.loc[agent1, agent2] = np.nan
        
        bootstrap_results['bootstrapped_matrices'].append(performance_matrix)
        
        # 3. Compute Nash equilibrium and metrics from this matrix
        try:
            # Convert matrix to numpy array for computation
            game_matrix_np = performance_matrix.to_numpy()
            
            # Handle missing values
            for i in range(game_matrix_np.shape[0]):
                for j in range(game_matrix_np.shape[1]):
                    if np.isnan(game_matrix_np[i, j]):
                        # Try column mean first (more relevant for opponent-specific performance)
                        col_mean = np.nanmean(game_matrix_np[:, j])
                        if not np.isnan(col_mean):
                            game_matrix_np[i, j] = col_mean
                        else:
                            # Fall back to row mean if column mean is not available
                            row_mean = np.nanmean(game_matrix_np[i, :])
                            game_matrix_np[i, j] = row_mean if not np.isnan(row_mean) else 0
            
            # Calculate Nash equilibrium
            nash_strategy = milp_max_sym_ent_2p(game_matrix_np, 100)
            
            # Calculate expected utilities against the Nash mixture
            expected_utils = np.dot(game_matrix_np, nash_strategy)
            
            # Calculate Nash equilibrium value (expected utility of Nash mixture against itself)
            nash_value = nash_strategy.reshape((1, -1)) @ game_matrix_np @ nash_strategy.reshape((-1, 1))
            nash_value = nash_value.item()  # Convert to scalar
            
            # Calculate max possible utility for each agent
            max_utils = np.max(game_matrix_np, axis=1)
            
            # Calculate both regret metrics
            relative_regrets = expected_utils - nash_value  # Can be positive or negative
            traditional_regrets = max_utils - expected_utils  # Always non-negative
            
            # Store results
            bootstrap_results['ne_regret'].append(relative_regrets)
            bootstrap_results['traditional_regret'].append(traditional_regrets)
            bootstrap_results['ne_strategy'].append(nash_strategy)
            bootstrap_results['agent_expected_utility'].append(expected_utils)
            bootstrap_results['agent_max_utility'].append(max_utils)
            bootstrap_results['nash_value'].append(nash_value)
        except Exception as e:
            print(f"Error in bootstrap sample {b}: {e}")
            continue
    
    # Calculate statistics and confidence intervals
    bootstrap_stats = analyze_bootstrap_results(bootstrap_results, all_agents, confidence)
    bootstrap_results['statistics'] = bootstrap_stats
    
    return bootstrap_results, all_agents

def analyze_bootstrap_results(bootstrap_results, agent_names, confidence=0.95):
    """
    Analyze bootstrap results and compute confidence intervals
    
    Args:
        bootstrap_results: Dictionary with bootstrap samples
        agent_names: List of agent names
        confidence: Confidence level (default: 0.95)
        
    Returns:
        DataFrame with statistics and confidence intervals
    """
    if not bootstrap_results['ne_regret']:
        print("No bootstrap results to analyze. Try running with different parameters.")
        return pd.DataFrame({
            'Agent': agent_names,
            'Mean NE Regret': [np.nan] * len(agent_names),
            'Std NE Regret': [np.nan] * len(agent_names),
            f'Lower {confidence*100:.0f}% CI (Regret)': [np.nan] * len(agent_names),
            f'Upper {confidence*100:.0f}% CI (Regret)': [np.nan] * len(agent_names),
            'Mean Traditional Regret': [np.nan] * len(agent_names),
            'Std Traditional Regret': [np.nan] * len(agent_names),
            f'Lower {confidence*100:.0f}% CI (Trad Regret)': [np.nan] * len(agent_names),
            f'Upper {confidence*100:.0f}% CI (Trad Regret)': [np.nan] * len(agent_names),
            'Mean Expected Utility': [np.nan] * len(agent_names),
            'Std Expected Utility': [np.nan] * len(agent_names),
            f'Lower {confidence*100:.0f}% CI (Utility)': [np.nan] * len(agent_names),
            f'Upper {confidence*100:.0f}% CI (Utility)': [np.nan] * len(agent_names)
        })
    
    # Initialize arrays to store statistics for each agent
    num_agents = len(agent_names)
    mean_regrets = np.zeros(num_agents)
    std_regrets = np.zeros(num_agents)
    lower_regrets = np.zeros(num_agents)
    upper_regrets = np.zeros(num_agents)
    
    mean_trad_regrets = np.zeros(num_agents)
    std_trad_regrets = np.zeros(num_agents)
    lower_trad_regrets = np.zeros(num_agents)
    upper_trad_regrets = np.zeros(num_agents)
    
    mean_expected_utils = np.zeros(num_agents)
    std_expected_utils = np.zeros(num_agents)
    lower_utils = np.zeros(num_agents)
    upper_utils = np.zeros(num_agents)
    
    # Calculate percentile thresholds
    alpha = 1 - confidence
    lower_percentile = alpha / 2 * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    # Process each agent's statistics individually to avoid dimensionality issues
    for agent_idx in range(num_agents):
        # Extract values for this agent from all bootstrap samples
        agent_ne_regrets = []
        agent_trad_regrets = []
        agent_expected_utils = []
        
        for b in range(len(bootstrap_results['ne_regret'])):
            try:
                regret_sample = bootstrap_results['ne_regret'][b]
                trad_regret_sample = bootstrap_results['traditional_regret'][b]
                util_sample = bootstrap_results['agent_expected_utility'][b]
                
                # Check if samples are the right shape
                if agent_idx < len(regret_sample):
                    agent_ne_regrets.append(float(regret_sample[agent_idx]))
                
                if agent_idx < len(trad_regret_sample):
                    agent_trad_regrets.append(float(trad_regret_sample[agent_idx]))
                
                if agent_idx < len(util_sample):
                    agent_expected_utils.append(float(util_sample[agent_idx]))
            except (IndexError, TypeError) as e:
                print(f"Warning: Error processing bootstrap sample {b} for agent {agent_idx}: {e}")
                continue
        
        # Calculate statistics if we have samples
        if agent_ne_regrets:
            mean_regrets[agent_idx] = np.mean(agent_ne_regrets)
            std_regrets[agent_idx] = np.std(agent_ne_regrets)
            lower_regrets[agent_idx] = np.percentile(agent_ne_regrets, lower_percentile)
            upper_regrets[agent_idx] = np.percentile(agent_ne_regrets, upper_percentile)
        
        if agent_trad_regrets:
            mean_trad_regrets[agent_idx] = np.mean(agent_trad_regrets)
            std_trad_regrets[agent_idx] = np.std(agent_trad_regrets)
            lower_trad_regrets[agent_idx] = np.percentile(agent_trad_regrets, lower_percentile)
            upper_trad_regrets[agent_idx] = np.percentile(agent_trad_regrets, upper_percentile)
        
        if agent_expected_utils:
            mean_expected_utils[agent_idx] = np.mean(agent_expected_utils)
            std_expected_utils[agent_idx] = np.std(agent_expected_utils)
            lower_utils[agent_idx] = np.percentile(agent_expected_utils, lower_percentile)
            upper_utils[agent_idx] = np.percentile(agent_expected_utils, upper_percentile)
    
    # Create a DataFrame with results
    results = pd.DataFrame({
        'Agent': agent_names,
        'Mean NE Regret': mean_regrets,
        'Std NE Regret': std_regrets,
        f'Lower {confidence*100:.0f}% CI (Regret)': lower_regrets,
        f'Upper {confidence*100:.0f}% CI (Regret)': upper_regrets,
        'Mean Traditional Regret': mean_trad_regrets,
        'Std Traditional Regret': std_trad_regrets,
        f'Lower {confidence*100:.0f}% CI (Trad Regret)': lower_trad_regrets,
        f'Upper {confidence*100:.0f}% CI (Trad Regret)': upper_trad_regrets,
        'Mean Expected Utility': mean_expected_utils,
        'Std Expected Utility': std_expected_utils,
        f'Lower {confidence*100:.0f}% CI (Utility)': lower_utils,
        f'Upper {confidence*100:.0f}% CI (Utility)': upper_utils
    })
    
    # Sort by mean NE regret
    results = results.sort_values(by='Mean NE Regret', ascending=False)
    
    return results 