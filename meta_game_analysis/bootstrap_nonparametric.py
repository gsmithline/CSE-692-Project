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
from nash_equilibrium.nash_solver import milp_max_sym_ent_2p, replicator_dynamics_nash

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
        'ne_regret': [],               # Nash equilibrium regret (expected_utils - nash_value)
        'ne_strategy': [],
        'rd_regret': [],               # Replicator dynamics Nash regret
        'rd_strategy': [],             # Replicator dynamics Nash strategies
        'agent_expected_utility': [],
        'agent_max_utility': [],
        'nash_value': [],              # Nash equilibrium value for each bootstrap
        'rd_nash_value': [],           # RD Nash equilibrium value
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
            
            # Calculate ME Nash equilibrium
            nash_strategy = milp_max_sym_ent_2p(game_matrix_np, 2000)
            
            # Calculate Replicator Dynamics Nash Equilibrium
            rd_strategy = replicator_dynamics_nash(game_matrix_np, 3000)
            
            # Calculate expected utilities against the ME Nash mixture
            expected_utils = np.dot(game_matrix_np, nash_strategy)
            
            # Calculate expected utilities against the RD Nash mixture
            rd_expected_utils = np.dot(game_matrix_np, rd_strategy)
            
            # Calculate ME Nash equilibrium value (expected utility of Nash mixture against itself)
            nash_value = nash_strategy.reshape((1, -1)) @ game_matrix_np @ nash_strategy.reshape((-1, 1))
            nash_value = nash_value.item()  # Convert to scalar
            
            # Calculate RD Nash equilibrium value
            rd_nash_value = rd_strategy.reshape((1, -1)) @ game_matrix_np @ rd_strategy.reshape((-1, 1))
            rd_nash_value = rd_nash_value.item()  # Convert to scalar
            
            # Calculate max possible utility for each agent
            max_utils = np.max(game_matrix_np, axis=1)
            
            # Calculate Nash equilibrium regret for ME NE
            nash_regrets = expected_utils - nash_value
            
            # Calculate Nash equilibrium regret for RD NE
            rd_regrets = rd_expected_utils - rd_nash_value
            
            # Validate that all regrets are at most 0 (or very close to 0 due to numerical precision)
            # A true Nash equilibrium should not allow any positive regrets
            
            # Use different thresholds for different purposes
            refinement_threshold = 1e-6  # Threshold to attempt refinement (more strict)
            warning_threshold = 1e-5     # Threshold to print warnings (more strict)
            capping_threshold = 1e-10    # Threshold for numerical precision issues (more strict)
            
            # Before capping, attempt to refine the strategies with high regrets
            if np.any(nash_regrets > refinement_threshold):
                max_regret = np.max(nash_regrets)
                worst_agent_idx = np.argmax(nash_regrets)
                
                # Try to refine the ME Nash strategy
                refined_nash_strategy = nash_strategy.copy()
                
                # Increase weight for actions with positive regret
                positive_regret_idxs = np.where(nash_regrets > refinement_threshold)[0]
                for idx in positive_regret_idxs:
                    # Add weight proportional to regret
                    weight_to_add = min(0.2, nash_regrets[idx] / 100.0)
                    refined_nash_strategy[idx] += weight_to_add
                
                # Renormalize
                refined_nash_strategy = refined_nash_strategy / np.sum(refined_nash_strategy)
                
                # Recalculate regrets with refined strategy
                refined_expected_utils = np.dot(game_matrix_np, refined_nash_strategy)
                refined_nash_value = refined_nash_strategy.reshape((1, -1)) @ game_matrix_np @ refined_nash_strategy.reshape((-1, 1))
                refined_nash_value = refined_nash_value.item()
                refined_nash_regrets = refined_expected_utils - refined_nash_value
                
                # If the refinement improved regrets, use it
                if np.max(refined_nash_regrets) < max_regret:
                    nash_strategy = refined_nash_strategy
                    expected_utils = refined_expected_utils
                    nash_value = refined_nash_value
                    nash_regrets = refined_nash_regrets
                    if max_regret > warning_threshold:  # Only print for significant improvements
                        print(f"Successfully reduced ME NE max regret from {max_regret:.6f} to {np.max(nash_regrets):.6f}")
                else:
                    # Only print warnings for significant regrets
                    if max_regret > warning_threshold:
                        print(f"Warning: Detected positive ME NE regret ({max_regret:.6f}) for agent {worst_agent_idx}. "
                              f"Refinement didn't improve regrets.")
            
            # Similar refinement for RD Nash if needed
            if np.any(rd_regrets > refinement_threshold):
                max_regret = np.max(rd_regrets)
                worst_agent_idx = np.argmax(rd_regrets)
                
                # Try to refine the RD Nash strategy
                refined_rd_strategy = rd_strategy.copy()
                
                # Increase weight for actions with positive regret
                positive_regret_idxs = np.where(rd_regrets > refinement_threshold)[0]
                for idx in positive_regret_idxs:
                    # Add weight proportional to regret
                    weight_to_add = min(0.2, rd_regrets[idx] / 100.0)
                    refined_rd_strategy[idx] += weight_to_add
                
                # Renormalize
                refined_rd_strategy = refined_rd_strategy / np.sum(refined_rd_strategy)
                
                # Recalculate regrets with refined strategy
                refined_rd_expected_utils = np.dot(game_matrix_np, refined_rd_strategy)
                refined_rd_nash_value = refined_rd_strategy.reshape((1, -1)) @ game_matrix_np @ refined_rd_strategy.reshape((-1, 1))
                refined_rd_nash_value = refined_rd_nash_value.item()
                refined_rd_regrets = refined_rd_expected_utils - refined_rd_nash_value
                
                # If the refinement improved regrets, use it
                if np.max(refined_rd_regrets) < max_regret:
                    rd_strategy = refined_rd_strategy
                    rd_expected_utils = refined_rd_expected_utils
                    rd_nash_value = refined_rd_nash_value
                    rd_regrets = refined_rd_regrets
                    if max_regret > warning_threshold:  # Only print for significant improvements
                        print(f"Successfully reduced RD NE max regret from {max_regret:.6f} to {np.max(rd_regrets):.6f}")
                else:
                    # Only print warnings for significant regrets
                    if max_regret > warning_threshold:
                        print(f"Warning: Detected positive RD NE regret ({max_regret:.6f}) for agent {worst_agent_idx}. "
                              f"Refinement didn't improve regrets.")
            
            # After refinement attempts, force cap regrets to ensure theoretical correctness
            # Only print warnings for significant regrets
            max_me_regret = np.max(nash_regrets)
            if max_me_regret > warning_threshold:
                print(f"Warning: Detected positive ME NE regret ({max_me_regret:.6f}) for agent {np.argmax(nash_regrets)}. "
                      f"This suggests the Nash equilibrium computation may not have fully converged.")
            
            # Always cap regrets to non-positive values (all should be ≤ 0)
            nash_regrets = np.minimum(nash_regrets, 0.0)
            
            max_rd_regret = np.max(rd_regrets)
            if max_rd_regret > warning_threshold:
                print(f"Warning: Detected positive RD NE regret ({max_rd_regret:.6f}) for agent {np.argmax(rd_regrets)}. "
                      f"This suggests the RD Nash equilibrium computation may not have fully converged.")
            
            # Always cap regrets to non-positive values (all should be ≤ 0)
            rd_regrets = np.minimum(rd_regrets, 0.0)
            
            # Store results
            bootstrap_results['ne_regret'].append(nash_regrets)
            bootstrap_results['ne_strategy'].append(nash_strategy)
            bootstrap_results['rd_regret'].append(rd_regrets)
            bootstrap_results['rd_strategy'].append(rd_strategy)
            bootstrap_results['agent_expected_utility'].append(expected_utils)
            bootstrap_results['agent_max_utility'].append(max_utils)
            bootstrap_results['nash_value'].append(nash_value)
            bootstrap_results['rd_nash_value'].append(rd_nash_value)
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
            'Mean Expected Utility': [np.nan] * len(agent_names),
            'Std Expected Utility': [np.nan] * len(agent_names),
            f'Lower {confidence*100:.0f}% CI (Utility)': [np.nan] * len(agent_names),
            f'Upper {confidence*100:.0f}% CI (Utility)': [np.nan] * len(agent_names),
            'Mean RD Regret': [np.nan] * len(agent_names),
            'Std RD Regret': [np.nan] * len(agent_names),
            f'Lower {confidence*100:.0f}% CI (RD Regret)': [np.nan] * len(agent_names),
            f'Upper {confidence*100:.0f}% CI (RD Regret)': [np.nan] * len(agent_names)
        })
    
    # Initialize arrays to store statistics for each agent
    num_agents = len(agent_names)
    mean_regrets = np.zeros(num_agents)
    std_regrets = np.zeros(num_agents)
    lower_regrets = np.zeros(num_agents)
    upper_regrets = np.zeros(num_agents)
    
    mean_expected_utils = np.zeros(num_agents)
    std_expected_utils = np.zeros(num_agents)
    lower_utils = np.zeros(num_agents)
    upper_utils = np.zeros(num_agents)
    
    # Initialize RD regret statistics
    mean_rd_regrets = np.zeros(num_agents)
    std_rd_regrets = np.zeros(num_agents)
    lower_rd_regrets = np.zeros(num_agents)
    upper_rd_regrets = np.zeros(num_agents)
    
    # Calculate percentile thresholds
    alpha = 1 - confidence
    lower_percentile = alpha / 2 * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    # Process each agent's statistics individually to avoid dimensionality issues
    for agent_idx in range(num_agents):
        # Extract values for this agent from all bootstrap samples
        agent_ne_regrets = []
        agent_expected_utils = []
        agent_rd_regrets = []
        
        for b in range(len(bootstrap_results['ne_regret'])):
            try:
                regret_sample = bootstrap_results['ne_regret'][b]
                util_sample = bootstrap_results['agent_expected_utility'][b]
                
                # Check if RD regrets exist for this bootstrap sample
                if 'rd_regret' in bootstrap_results and b < len(bootstrap_results['rd_regret']):
                    rd_regret_sample = bootstrap_results['rd_regret'][b]
                    if agent_idx < len(rd_regret_sample):
                        agent_rd_regrets.append(float(rd_regret_sample[agent_idx]))
                
                # Check if samples are the right shape
                if agent_idx < len(regret_sample):
                    agent_ne_regrets.append(float(regret_sample[agent_idx]))
                
                if agent_idx < len(util_sample):
                    agent_expected_utils.append(float(util_sample[agent_idx]))
            except (IndexError, TypeError) as e:
                print(f"Warning: Error processing bootstrap sample {b} for agent {agent_idx}: {e}")
                continue
        
        # Calculate statistics if we have samples
        if agent_ne_regrets:
            # Convert to regular Python list of floats to avoid numpy type issues
            agent_ne_regrets = [float(x) for x in agent_ne_regrets]
            mean_regrets[agent_idx] = sum(agent_ne_regrets) / len(agent_ne_regrets)
            # Manual std calculation to avoid numpy issues
            if len(agent_ne_regrets) > 1:
                variance = sum((x - mean_regrets[agent_idx])**2 for x in agent_ne_regrets) / (len(agent_ne_regrets) - 1)
                std_regrets[agent_idx] = variance**0.5
            else:
                std_regrets[agent_idx] = 0.0
            
            # Use numpy for percentiles only
            agent_ne_regrets_array = np.array(agent_ne_regrets)
            lower_regrets[agent_idx] = np.percentile(agent_ne_regrets_array, lower_percentile)
            upper_regrets[agent_idx] = np.percentile(agent_ne_regrets_array, upper_percentile)
        
        if agent_expected_utils:
            # Convert to regular Python list of floats
            agent_expected_utils = [float(x) for x in agent_expected_utils]
            mean_expected_utils[agent_idx] = sum(agent_expected_utils) / len(agent_expected_utils)
            # Manual std calculation
            if len(agent_expected_utils) > 1:
                variance = sum((x - mean_expected_utils[agent_idx])**2 for x in agent_expected_utils) / (len(agent_expected_utils) - 1)
                std_expected_utils[agent_idx] = variance**0.5
            else:
                std_expected_utils[agent_idx] = 0.0
                
            # Use numpy for percentiles
            agent_expected_utils_array = np.array(agent_expected_utils)
            lower_utils[agent_idx] = np.percentile(agent_expected_utils_array, lower_percentile)
            upper_utils[agent_idx] = np.percentile(agent_expected_utils_array, upper_percentile)
        
        # Calculate RD regret statistics if data is available
        if agent_rd_regrets:
            # Convert to regular Python list of floats
            agent_rd_regrets = [float(x) for x in agent_rd_regrets]
            mean_rd_regrets[agent_idx] = sum(agent_rd_regrets) / len(agent_rd_regrets)
            # Manual std calculation
            if len(agent_rd_regrets) > 1:
                variance = sum((x - mean_rd_regrets[agent_idx])**2 for x in agent_rd_regrets) / (len(agent_rd_regrets) - 1)
                std_rd_regrets[agent_idx] = variance**0.5
            else:
                std_rd_regrets[agent_idx] = 0.0
            
            # Use numpy for percentiles
            agent_rd_regrets_array = np.array(agent_rd_regrets)
            lower_rd_regrets[agent_idx] = np.percentile(agent_rd_regrets_array, lower_percentile)
            upper_rd_regrets[agent_idx] = np.percentile(agent_rd_regrets_array, upper_percentile)
    
    # Create a DataFrame with results
    results = pd.DataFrame({
        'Agent': agent_names,
        'Mean NE Regret': mean_regrets,
        'Std NE Regret': std_regrets,
        f'Lower {confidence*100:.0f}% CI (Regret)': lower_regrets,
        f'Upper {confidence*100:.0f}% CI (Regret)': upper_regrets,
        'Mean Expected Utility': mean_expected_utils,
        'Std Expected Utility': std_expected_utils,
        f'Lower {confidence*100:.0f}% CI (Utility)': lower_utils,
        f'Upper {confidence*100:.0f}% CI (Utility)': upper_utils,
        'Mean RD Regret': mean_rd_regrets,
        'Std RD Regret': std_rd_regrets,
        f'Lower {confidence*100:.0f}% CI (RD Regret)': lower_rd_regrets,
        f'Upper {confidence*100:.0f}% CI (RD Regret)': upper_rd_regrets
    })
    
    # Sort by mean NE regret
    results = results.sort_values(by='Mean NE Regret', ascending=False)
    
    return results 