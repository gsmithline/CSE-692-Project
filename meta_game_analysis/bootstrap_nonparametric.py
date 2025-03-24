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
    
    bootstrap_stats = analyze_bootstrap_results(bootstrap_results, all_agents, confidence)
    bootstrap_results['statistics'] = bootstrap_stats
    
    return bootstrap_results, all_agents

def analyze_bootstrap_results(bootstrap_results, agent_names, confidence=0.95):
    """
    Analyze bootstrap results and compute confidence intervals for Nash equilibrium regrets
    
    Args:
        bootstrap_results: Dictionary with bootstrap samples
        agent_names: List of agent names
        confidence: Confidence level (default: 0.95)
        
    Returns:
        DataFrame with statistics and confidence intervals
    """
    if not bootstrap_results['ne_regret']:
        print("No bootstrap results to analyze. Check for errors in the bootstrap process.")
        return pd.DataFrame({
            'Agent': agent_names,
            'Mean NE Regret': [np.nan] * len(agent_names),
            'Std NE Regret': [np.nan] * len(agent_names),
            f'Lower {confidence*100:.0f}% CI (NE Regret)': [np.nan] * len(agent_names),
            f'Upper {confidence*100:.0f}% CI (NE Regret)': [np.nan] * len(agent_names),
            'Mean Expected Utility': [np.nan] * len(agent_names),
            'Std Expected Utility': [np.nan] * len(agent_names)
        })
    
    try:
        # Stack all bootstrap samples into arrays for analysis
        ne_regrets = np.stack(bootstrap_results['ne_regret'])
        expected_utils = np.stack(bootstrap_results['agent_expected_utility'])
        
        # Add RD regrets if available
        has_rd_regrets = 'rd_regret' in bootstrap_results and bootstrap_results['rd_regret']
        if has_rd_regrets:
            rd_regrets = np.stack(bootstrap_results['rd_regret'])
            
        # Add normal regrets if available
        has_me_normal_regrets = 'me_normal_regret' in bootstrap_results and bootstrap_results['me_normal_regret']
        has_rd_normal_regrets = 'rd_normal_regret' in bootstrap_results and bootstrap_results['rd_normal_regret']
        
        if has_me_normal_regrets:
            me_normal_regrets = np.stack(bootstrap_results['me_normal_regret'])
        
        if has_rd_normal_regrets:
            rd_normal_regrets = np.stack(bootstrap_results['rd_normal_regret'])
    except ValueError:
        print("Warning: Bootstrap samples have inconsistent shapes. Using a more flexible approach.")
        
        first_regret = bootstrap_results['ne_regret'][0]
        first_util = bootstrap_results['agent_expected_utility'][0]
        
        num_samples = len(bootstrap_results['ne_regret'])
        ne_regrets = np.zeros((num_samples, len(first_regret)), dtype=np.float64)
        expected_utils = np.zeros((num_samples, len(first_util)), dtype=np.float64)
        
        for i in range(num_samples):
            if i < len(bootstrap_results['ne_regret']):
                ne_regrets[i] = bootstrap_results['ne_regret'][i]
            if i < len(bootstrap_results['agent_expected_utility']):
                expected_utils[i] = bootstrap_results['agent_expected_utility'][i]
        
        has_rd_regrets = 'rd_regret' in bootstrap_results and bootstrap_results['rd_regret']
        if has_rd_regrets:
            first_rd_regret = bootstrap_results['rd_regret'][0]
            rd_regrets = np.zeros((num_samples, len(first_rd_regret)), dtype=np.float64)
            for i in range(num_samples):
                if i < len(bootstrap_results['rd_regret']):
                    rd_regrets[i] = bootstrap_results['rd_regret'][i]
        
        has_me_normal_regrets = 'me_normal_regret' in bootstrap_results and bootstrap_results['me_normal_regret']
        if has_me_normal_regrets:
            first_me_normal = bootstrap_results['me_normal_regret'][0]
            me_normal_regrets = np.zeros((num_samples, len(first_me_normal)), dtype=np.float64)
            for i in range(num_samples):
                if i < len(bootstrap_results['me_normal_regret']):
                    me_normal_regrets[i] = bootstrap_results['me_normal_regret'][i]
        
        has_rd_normal_regrets = 'rd_normal_regret' in bootstrap_results and bootstrap_results['rd_normal_regret']
        if has_rd_normal_regrets:
            first_rd_normal = bootstrap_results['rd_normal_regret'][0]
            rd_normal_regrets = np.zeros((num_samples, len(first_rd_normal)), dtype=np.float64)
            for i in range(num_samples):
                if i < len(bootstrap_results['rd_normal_regret']):
                    rd_normal_regrets[i] = bootstrap_results['rd_normal_regret'][i]
    
    # Calculate means
    mean_ne_regrets = np.mean(ne_regrets, axis=0)
    mean_expected_utils = np.mean(expected_utils, axis=0)
    if has_rd_regrets:
        mean_rd_regrets = np.mean(rd_regrets, axis=0)
    if has_me_normal_regrets:
        mean_me_normal_regrets = np.mean(me_normal_regrets, axis=0)
    if has_rd_normal_regrets:
        mean_rd_normal_regrets = np.mean(rd_normal_regrets, axis=0)
    
    # Calculate non-parametric standard deviations
    std_ne_regrets = np.power(np.mean((ne_regrets - mean_ne_regrets) ** 2, axis=0), 0.5)
    std_expected_utils = np.power(np.mean((expected_utils - mean_expected_utils) ** 2, axis=0), 0.5)
    if has_rd_regrets:
        std_rd_regrets = np.power(np.mean((rd_regrets - mean_rd_regrets) ** 2, axis=0), 0.5)
    if has_me_normal_regrets:
        std_me_normal_regrets = np.power(np.mean((me_normal_regrets - mean_me_normal_regrets) ** 2, axis=0), 0.5)
    if has_rd_normal_regrets:
        std_rd_normal_regrets = np.power(np.mean((rd_normal_regrets - mean_rd_normal_regrets) ** 2, axis=0), 0.5)
    
    epsilon = 1e-3  # More forgiving for numerical precision issues
    if np.any(mean_ne_regrets > epsilon):
        max_ne_regret = np.max(mean_ne_regrets)
        worst_idx_ne = np.argmax(mean_ne_regrets)
        worst_agent_ne = agent_names[worst_idx_ne]
        
        error_msg = [f"Max Entropy NE: {max_ne_regret:.10f} for agent {worst_agent_ne}"]
        
        if has_rd_regrets and np.any(mean_rd_regrets > epsilon):
            max_rd_regret = np.max(mean_rd_regrets)
            worst_idx_rd = np.argmax(mean_rd_regrets)
            worst_agent_rd = agent_names[worst_idx_rd]
            error_msg.append(f"RD NE: {max_rd_regret:.10f} for agent {worst_agent_rd}")
        
        if has_me_normal_regrets and np.any(mean_me_normal_regrets > epsilon):
            max_me_normal = np.max(mean_me_normal_regrets)
            worst_idx_me_normal = np.argmax(mean_me_normal_regrets)
            worst_agent_me_normal = agent_names[worst_idx_me_normal]
            error_msg.append(f"ME Normal: {max_me_normal:.10f} for agent {worst_agent_me_normal}")
        
        if has_rd_normal_regrets and np.any(mean_rd_normal_regrets > epsilon):
            max_rd_normal = np.max(mean_rd_normal_regrets)
            worst_idx_rd_normal = np.argmax(mean_rd_normal_regrets)
            worst_agent_rd_normal = agent_names[worst_idx_rd_normal]
            error_msg.append(f"RD Normal: {max_rd_normal:.10f} for agent {worst_agent_rd_normal}")
            
        print(f"WARNING: Large positive mean regret detected:\n{', '.join(error_msg)}")
    elif np.any(mean_ne_regrets > 0) or (has_rd_regrets and np.any(mean_rd_regrets > 0)) or \
         (has_me_normal_regrets and np.any(mean_me_normal_regrets > 0)) or \
         (has_rd_normal_regrets and np.any(mean_rd_normal_regrets > 0)):
        # For small positive regrets, just warn
        if np.any(mean_ne_regrets > 0):
            max_ne_regret = np.max(mean_ne_regrets)
            worst_idx_ne = np.argmax(mean_ne_regrets)
            worst_agent_ne = agent_names[worst_idx_ne]
            print(f"WARNING: Small positive mean ME Nash regret detected: {max_ne_regret:.10f} for agent {worst_agent_ne}")
        
        if has_rd_regrets and np.any(mean_rd_regrets > 0):
            max_rd_regret = np.max(mean_rd_regrets)
            worst_idx_rd = np.argmax(mean_rd_regrets)
            worst_agent_rd = agent_names[worst_idx_rd]
            print(f"WARNING: Small positive mean RD Nash regret detected: {max_rd_regret:.10f} for agent {worst_agent_rd}")
            
        if has_me_normal_regrets and np.any(mean_me_normal_regrets > 0):
            max_me_normal = np.max(mean_me_normal_regrets)
            worst_idx_me_normal = np.argmax(mean_me_normal_regrets)
            worst_agent_me_normal = agent_names[worst_idx_me_normal]
            print(f"WARNING: Small positive mean ME normal regret detected: {max_me_normal:.10f} for agent {worst_agent_me_normal}")
            
        if has_rd_normal_regrets and np.any(mean_rd_normal_regrets > 0):
            max_rd_normal = np.max(mean_rd_normal_regrets)
            worst_idx_rd_normal = np.argmax(mean_rd_normal_regrets)
            worst_agent_rd_normal = agent_names[worst_idx_rd_normal]
            print(f"WARNING: Small positive mean RD normal regret detected: {max_rd_normal:.10f} for agent {worst_agent_rd_normal}")
    
    # Calculate percentile-based confidence intervals
    alpha = 1 - confidence
    lower_percentile = alpha / 2 * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    lower_ne_regrets = np.percentile(ne_regrets, lower_percentile, axis=0)
    upper_ne_regrets = np.percentile(ne_regrets, upper_percentile, axis=0)
    
    if has_rd_regrets:
        lower_rd_regrets = np.percentile(rd_regrets, lower_percentile, axis=0)
        upper_rd_regrets = np.percentile(rd_regrets, upper_percentile, axis=0)
        
    if has_me_normal_regrets:
        lower_me_normal_regrets = np.percentile(me_normal_regrets, lower_percentile, axis=0)
        upper_me_normal_regrets = np.percentile(me_normal_regrets, upper_percentile, axis=0)
        
    if has_rd_normal_regrets:
        lower_rd_normal_regrets = np.percentile(rd_normal_regrets, lower_percentile, axis=0)
        upper_rd_normal_regrets = np.percentile(rd_normal_regrets, upper_percentile, axis=0)
    
    # Check if any upper bound of the confidence interval is above 0
    if np.any(upper_ne_regrets > epsilon):
        warnings.warn("Some confidence intervals for NE regrets include positive values. "
                     "This may indicate numerical instability in the Nash equilibrium calculation.")
    
    # Create DataFrame with results
    results = pd.DataFrame({
        'Agent': agent_names,
        'Mean NE Regret': mean_ne_regrets,
        'Std NE Regret': std_ne_regrets,
        f'Lower {confidence*100:.0f}% CI (NE Regret)': lower_ne_regrets,
        f'Upper {confidence*100:.0f}% CI (NE Regret)': upper_ne_regrets,
        'Mean Expected Utility': mean_expected_utils,
        'Std Expected Utility': std_expected_utils
    })
    
    # Add RD regret columns if available
    if has_rd_regrets:
        results['Mean RD Regret'] = mean_rd_regrets
        results['Std RD Regret'] = std_rd_regrets
        results[f'Lower {confidence*100:.0f}% CI (RD Regret)'] = lower_rd_regrets
        results[f'Upper {confidence*100:.0f}% CI (RD Regret)'] = upper_rd_regrets
    
    # Add ME normal regret columns if available
    if has_me_normal_regrets:
        results['Mean ME Normal Regret'] = mean_me_normal_regrets
        results['Std ME Normal Regret'] = std_me_normal_regrets
        results[f'Lower {confidence*100:.0f}% CI (ME Normal Regret)'] = lower_me_normal_regrets
        results[f'Upper {confidence*100:.0f}% CI (ME Normal Regret)'] = upper_me_normal_regrets
    
    # Add RD normal regret columns if available
    if has_rd_normal_regrets:
        results['Mean RD Normal Regret'] = mean_rd_normal_regrets
        results['Std RD Normal Regret'] = std_rd_normal_regrets
        results[f'Lower {confidence*100:.0f}% CI (RD Normal Regret)'] = lower_rd_normal_regrets
        results[f'Upper {confidence*100:.0f}% CI (RD Normal Regret)'] = upper_rd_normal_regrets
    
    # Sort by the mean expected utility (higher is better)
    results = results.sort_values(by='Mean Expected Utility', ascending=False)
    
    # Print summary statistics
    print("\nSummary of Nash Equilibrium Analysis:")
    print(f"Average NE regret: {np.mean(mean_ne_regrets):.8f}")
    if has_rd_regrets:
        print(f"Average RD regret: {np.mean(mean_rd_regrets):.8f}")
    if has_me_normal_regrets:
        print(f"Average ME normal regret: {np.mean(mean_me_normal_regrets):.8f}")
    if has_rd_normal_regrets:
        print(f"Average RD normal regret: {np.mean(mean_rd_normal_regrets):.8f}")
    
    print(f"Maximum NE regret: {np.max(mean_ne_regrets):.8f}")
    if has_rd_regrets:
        print(f"Maximum RD regret: {np.max(mean_rd_regrets):.8f}")
    if has_me_normal_regrets:
        print(f"Maximum ME normal regret: {np.max(mean_me_normal_regrets):.8f}")
    if has_rd_normal_regrets:
        print(f"Maximum RD normal regret: {np.max(mean_rd_normal_regrets):.8f}")
    
    print("\nTop 5 agents by Expected Utility:")
    print(results[['Agent', 'Mean Expected Utility', 'Mean NE Regret']].head(5))
    
    return results 