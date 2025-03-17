"""
Bootstrapping and statistical analysis utilities for game theoretic analysis.
"""

import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
from scipy import stats
from nash_equilibrium.nash_solver import milp_max_sym_ent_2p

def bootstrap_performance_metrics(performance_matrix, num_bootstrap=100, with_replacement=True):
    """
    Use bootstrapping to estimate distributions of performance metrics
    
    Args:
        performance_matrix: DataFrame with agent performance data
        num_bootstrap: Number of bootstrap samples
        with_replacement: Whether to sample with replacement
        
    Returns:
        Dictionary of bootstrap results
    """
    all_agents = performance_matrix.index.tolist()
    num_agents = len(all_agents)
    
    # Metrics to track
    bootstrap_results = {
        'ne_regret': [],
        'ne_strategy': [],
        'agent_expected_utility': [],
        'agent_max_utility': []
    }
    
    for i in range(num_bootstrap):
        # Create bootstrap sample by adding noise to performance matrix
        # This simulates resampling in a simple way
        noise_level = 0.01 # 5% noise level
        bootstrap_matrix = performance_matrix.copy()
        
        for agent1 in all_agents:
            for agent2 in all_agents:
                if not np.isnan(bootstrap_matrix.loc[agent1, agent2]):
                    if with_replacement:
                        # Add random noise
                        noise = np.random.normal(0, noise_level * bootstrap_matrix.loc[agent1, agent2])
                        bootstrap_matrix.loc[agent1, agent2] += noise
        
        # Convert to numpy for computations
        game_matrix_np = bootstrap_matrix.to_numpy()
        
        # Handle NaN values by filling with column or row means
        for i in range(game_matrix_np.shape[0]):
            for j in range(game_matrix_np.shape[1]):
                if np.isnan(game_matrix_np[i, j]):
                    # Try column mean first
                    col_mean = np.nanmean(game_matrix_np[:, j])
                    if not np.isnan(col_mean):
                        game_matrix_np[i, j] = col_mean
                    else:
                        # If column is all NaN, use row mean
                        row_mean = np.nanmean(game_matrix_np[i, :])
                        game_matrix_np[i, j] = row_mean if not np.isnan(row_mean) else 0
        
        try:
            nash_strategy = milp_max_sym_ent_2p(game_matrix_np)
            
            expected_utils = np.dot(game_matrix_np, nash_strategy)
            
            max_utils = np.max(game_matrix_np, axis=1)
            
            regrets = max_utils - expected_utils
            
            nash_strategy = np.array(nash_strategy, dtype=np.float64)
            expected_utils = np.array(expected_utils, dtype=np.float64)
            max_utils = np.array(max_utils, dtype=np.float64)
            regrets = np.array(regrets, dtype=np.float64)
            
            bootstrap_results['ne_regret'].append(regrets)
            bootstrap_results['ne_strategy'].append(nash_strategy)
            bootstrap_results['agent_expected_utility'].append(expected_utils)
            bootstrap_results['agent_max_utility'].append(max_utils)
        except Exception as e:
            print(f"Error in bootstrap sample {i}: {e}")
            continue
    
    # Convert lists to numpy arrays for easier manipulation later
    bootstrap_results['ne_regret'] = [np.array(r, dtype=np.float64) for r in bootstrap_results['ne_regret']]
    bootstrap_results['ne_strategy'] = [np.array(s, dtype=np.float64) for s in bootstrap_results['ne_strategy']]
    bootstrap_results['agent_expected_utility'] = [np.array(u, dtype=np.float64) for u in bootstrap_results['agent_expected_utility']]
    bootstrap_results['agent_max_utility'] = [np.array(m, dtype=np.float64) for m in bootstrap_results['agent_max_utility']]
    
    return bootstrap_results

def analyze_bootstrap_results(bootstrap_results, agent_names, confidence=0.95):
    """
    Analyze bootstrap results and compute confidence intervals
    
    Args:
        bootstrap_results: Dictionary with bootstrap samples
        agent_names: List of agent names
        confidence: Confidence level (default: 0.95)
        
    Returns:
        DataFrame with statistics
    """
    # Check if we have any bootstrap results
    if not bootstrap_results['ne_regret']:
        print("No bootstrap results to analyze. Try running with different parameters.")
        # Return an empty DataFrame with the expected columns
        return pd.DataFrame({
            'Agent': agent_names,
            'Mean NE Regret': [np.nan] * len(agent_names),
            'Std NE Regret': [np.nan] * len(agent_names),
            f'Lower {confidence*100:.0f}% CI (Regret)': [np.nan] * len(agent_names),
            f'Upper {confidence*100:.0f}% CI (Regret)': [np.nan] * len(agent_names),
            'Mean Expected Utility': [np.nan] * len(agent_names),
            'Std Expected Utility': [np.nan] * len(agent_names),
            f'Lower {confidence*100:.0f}% CI (Utility)': [np.nan] * len(agent_names),
            f'Upper {confidence*100:.0f}% CI (Utility)': [np.nan] * len(agent_names)
        })
    
    # Stack the arrays along a new first axis
    try:
        ne_regrets = np.stack(bootstrap_results['ne_regret'])
        expected_utils = np.stack(bootstrap_results['agent_expected_utility'])
    except ValueError:
        # If stacking fails due to different shapes, try a more flexible approach
        print("Warning: Bootstrap samples have inconsistent shapes. Using a more flexible approach.")
        # Get the first result to determine the shape
        first_regret = bootstrap_results['ne_regret'][0]
        first_util = bootstrap_results['agent_expected_utility'][0]
        
        # Initialize arrays with the right shape
        ne_regrets = np.zeros((len(bootstrap_results['ne_regret']), len(first_regret)), dtype=np.float64)
        expected_utils = np.zeros((len(bootstrap_results['agent_expected_utility']), len(first_util)), dtype=np.float64)
        
        # Fill arrays manually
        for i, (regret, util) in enumerate(zip(bootstrap_results['ne_regret'], bootstrap_results['agent_expected_utility'])):
            if len(regret) == len(first_regret):
                ne_regrets[i] = regret
            else:
                print(f"Warning: Skipping regret sample {i} due to shape mismatch")
                
            if len(util) == len(first_util):
                expected_utils[i] = util
            else:
                print(f"Warning: Skipping utility sample {i} due to shape mismatch")
    
    # Calculate means
    mean_regrets = np.mean(ne_regrets, axis=0)
    mean_expected_utils = np.mean(expected_utils, axis=0)
    
    # Calculate standard deviations safely
    std_regrets = np.zeros_like(mean_regrets)
    std_expected_utils = np.zeros_like(mean_expected_utils)
    
    for i in range(len(mean_regrets)):
        std_regrets[i] = np.std(ne_regrets[:, i])
        std_expected_utils[i] = np.std(expected_utils[:, i])
    
    # Calculate confidence intervals
    n_samples = len(bootstrap_results['ne_regret'])
    t_value = stats.t.ppf((1 + confidence) / 2, n_samples - 1)
    
    regret_ci = t_value * std_regrets / np.sqrt(n_samples)
    utility_ci = t_value * std_expected_utils / np.sqrt(n_samples)
    
    # Create a DataFrame with results
    results = pd.DataFrame({
        'Agent': agent_names,
        'Mean NE Regret': mean_regrets,
        'Std NE Regret': std_regrets,
        f'Lower {confidence*100:.0f}% CI (Regret)': mean_regrets - regret_ci,
        f'Upper {confidence*100:.0f}% CI (Regret)': mean_regrets + regret_ci,
        'Mean Expected Utility': mean_expected_utils,
        'Std Expected Utility': std_expected_utils,
        f'Lower {confidence*100:.0f}% CI (Utility)': mean_expected_utils - utility_ci,
        f'Upper {confidence*100:.0f}% CI (Utility)': mean_expected_utils + utility_ci
    })
    
    # Sort by mean regret (ascending)
    results = results.sort_values(by='Mean NE Regret')
    
    return results

def plot_regret_distributions(bootstrap_results, agent_names, figsize=(12, 8)):
    """
    Plot distributions of NE regret for each agent
    
    Args:
        bootstrap_results: Dictionary with bootstrap samples
        agent_names: List of agent names
        figsize: Figure size
    """
    if not bootstrap_results['ne_regret']:
        print("No bootstrap results to plot.")
        return None
    
    try:
        ne_regrets = np.stack(bootstrap_results['ne_regret'])
    except ValueError:
        print("Warning: Bootstrap samples have inconsistent shapes. Using a more flexible approach.")
        # Get the first result to determine the shape
        first_regret = bootstrap_results['ne_regret'][0]
        
        # Initialize array with the right shape
        ne_regrets = np.zeros((len(bootstrap_results['ne_regret']), len(first_regret)), dtype=np.float64)
        
        # Fill array manually
        for i, regret in enumerate(bootstrap_results['ne_regret']):
            if len(regret) == len(first_regret):
                ne_regrets[i] = regret
            else:
                print(f"Warning: Skipping regret sample {i} due to shape mismatch")
    
    n_agents = len(agent_names)
    
    fig, axs = plt.subplots(int(np.ceil(n_agents/3)), 3, figsize=figsize)
    axs = axs.flatten()
    
    for i, agent in enumerate(agent_names):
        if i < len(axs) and i < ne_regrets.shape[1]:
            axs[i].hist(ne_regrets[:, i], bins=20, alpha=0.7)
            axs[i].set_title(agent)
            axs[i].set_xlabel('NE Regret')
            axs[i].set_ylabel('Frequency')
            
            # Add mean line
            mean_regret = np.mean(ne_regrets[:, i])
            axs[i].axvline(mean_regret, color='r', linestyle='--', 
                           label=f'Mean: {mean_regret:.4f}')
            
            # Add 95% CI
            lower_ci = np.percentile(ne_regrets[:, i], 2.5)
            upper_ci = np.percentile(ne_regrets[:, i], 97.5)
            axs[i].axvline(lower_ci, color='g', linestyle=':')
            axs[i].axvline(upper_ci, color='g', linestyle=':', 
                           label=f'95% CI: [{lower_ci:.4f}, {upper_ci:.4f}]')
            
            axs[i].legend(fontsize='small')
    
    # Hide any unused subplots
    for j in range(min(i+1, len(axs))):
        if j >= ne_regrets.shape[1]:
            axs[j].axis('off')
    
    plt.tight_layout()
    
    return fig

def compute_acceptance_ratio_matrix(all_results, agent_names):
    """
    Compute a matrix of acceptance ratios between agents
    
    Args:
        all_results: List of game results
        agent_names: List of agent names in canonical form
        
    Returns:
        DataFrame with acceptance ratios
    """
    acceptance_matrix = pd.DataFrame(index=agent_names, columns=agent_names)
    acceptance_count = defaultdict(lambda: defaultdict(int))
    total_games = defaultdict(lambda: defaultdict(int))
    
    # Debugging information
    print(f"Processing {len(all_results)} game results")
    print(f"Looking for {len(agent_names)} agent types: {agent_names[:3]}...")
    
    # Count agents in the results to check for name mismatches
    agent_in_results = set()
    for result in all_results:
        agent_in_results.add(result['agent1'])
        agent_in_results.add(result['agent2'])
    
    print(f"Found {len(agent_in_results)} unique agents in results")
    print(f"Example agents in results: {list(agent_in_results)[:3]}...")
    
    # Create a mapping between names in results and canonical names we're looking for
    name_mapping = {}
    
    # Function to normalize agent name (remove dates and standardize format)
    def normalize_name(name):
        # Handle date patterns (yyyy-mm-dd format)
        import re
        name = re.sub(r'_\d{4}-\d{2}-\d{2}', '', name)
        return name
    
    # First: Create mapping based on exact matches after normalization
    for result_agent in agent_in_results:
        normalized_name = normalize_name(result_agent)
        for agent_name in agent_names:
            if normalized_name == agent_name:
                name_mapping[result_agent] = agent_name
                print(f"Normalized match: '{result_agent}' → '{agent_name}'")
                break
    
    # Second: Try substring matches for any remaining unmatched agents
    if len(name_mapping) < len(agent_in_results):
        print("Some agents not matched with normalization, trying substring matching...")
        match_count = 0
        for result_agent in agent_in_results:
            if result_agent not in name_mapping:
                normalized_name = normalize_name(result_agent)
                best_match = None
                best_match_score = 0
                
                for agent_name in agent_names:
                    # Simple matching score (length of common substring)
                    if agent_name in normalized_name:
                        score = len(agent_name)
                        if score > best_match_score:
                            best_match_score = score
                            best_match = agent_name
                    elif normalized_name in agent_name:
                        score = len(normalized_name)
                        if score > best_match_score:
                            best_match_score = score
                            best_match = agent_name
                
                if best_match and best_match_score > 5:  # Only match if we have a reasonable substring match
                    name_mapping[result_agent] = best_match
                    match_count += 1
                    if match_count <= 10:  # Limit debug output
                        print(f"Substring match: '{result_agent}' → '{best_match}'")
    
    # Report matching statistics
    print(f"Mapped {len(name_mapping)}/{len(agent_in_results)} agent names")
    
    # Process games using the name mapping
    matches_found = 0
    for result in all_results:
        agent1 = result['agent1']
        agent2 = result['agent2']
        
        # Map the result agents to canonical names
        agent1_mapped = name_mapping.get(agent1, None)
        agent2_mapped = name_mapping.get(agent2, None)
        
        # Only count if both agents are in our mapping
        if agent1_mapped and agent2_mapped:
            matches_found += 1
            # Count the game
            total_games[agent1_mapped][agent2_mapped] += 1
            total_games[agent2_mapped][agent1_mapped] += 1
            
            # If the game ended with ACCEPT, count it
            if result.get('final_action') == "ACCEPT":
                acceptance_count[agent1_mapped][agent2_mapped] += 1
                acceptance_count[agent2_mapped][agent1_mapped] += 1
    
    print(f"Found {matches_found} games with mapped agent names")
    
    # Calculate ratios
    for agent1 in agent_names:
        for agent2 in agent_names:
            if total_games[agent1][agent2] > 0:
                ratio = acceptance_count[agent1][agent2] / total_games[agent1][agent2]
                acceptance_matrix.loc[agent1, agent2] = ratio
    
    # Report statistics
    filled_cells = acceptance_matrix.count().sum()
    total_cells = len(agent_names) * len(agent_names)
    print(f"Filled {filled_cells} out of {total_cells} cells in acceptance matrix ({filled_cells/total_cells:.1%})")
    
    return acceptance_matrix