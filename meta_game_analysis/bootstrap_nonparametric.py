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
        'rd_regret': [],              
        'rd_strategy': [],             
        'agent_expected_utility': [],
        'agent_max_utility': [],
        'nash_value': [],              
        'rd_nash_value': [],          
        'bootstrapped_matrices': []    
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

def run_bootstrap_analysis(performance_matrix, num_bootstrap=1000, confidence=0.95):
    """
    Run bootstrap analysis on a performance matrix using non-parametric bootstrapping.
    
    Args:
        performance_matrix: DataFrame with agent performance data (mean performance for each pair)
        num_bootstrap: Number of bootstrap replicas to generate
        confidence: Confidence level for intervals
        
    Returns:
        List of bootstrap sample results
    """
    print(f"Running non-parametric bootstrap analysis with {num_bootstrap} samples...")
    
    agents = performance_matrix.index.tolist()
    num_agents = len(agents)
    
    # Convert the performance matrix to a numpy array
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
    
    # List to store bootstrap results
    bootstrap_results = []
    
    # For each bootstrap iteration
    for b in range(num_bootstrap):
        if b % 100 == 0 and b > 0:
            print(f"Processed {b} bootstrap samples...")
        
        # Create a bootstrap sample by resampling with replacement
        bootstrap_indices = np.random.choice(
            range(game_matrix_np.shape[0]), 
            size=game_matrix_np.shape[0], 
            replace=True
        )
        
        # Create a bootstrap game matrix
        bootstrap_game_matrix = game_matrix_np[bootstrap_indices][:, bootstrap_indices]
        
        # Calculate ME Nash equilibrium
        try:
            nash_strategy = milp_max_sym_ent_2p(bootstrap_game_matrix, 2000)
            
            # Calculate Replicator Dynamics Nash Equilibrium
            rd_strategy = replicator_dynamics_nash(bootstrap_game_matrix, 1000)
            
            # Calculate expected utilities against the ME Nash mixture
            expected_utils = np.dot(bootstrap_game_matrix, nash_strategy)
            
            # Calculate expected utilities against the RD Nash mixture
            rd_expected_utils = np.dot(bootstrap_game_matrix, rd_strategy)
            
            # Calculate ME Nash equilibrium value
            nash_value = nash_strategy.reshape((1, -1)) @ bootstrap_game_matrix @ nash_strategy.reshape((-1, 1))
            nash_value = nash_value.item()  # Convert to scalar
            
            # Calculate RD Nash equilibrium value
            rd_nash_value = rd_strategy.reshape((1, -1)) @ bootstrap_game_matrix @ rd_strategy.reshape((-1, 1))
            rd_nash_value = rd_nash_value.item()  # Convert to scalar
            
            # Calculate Nash equilibrium regret
            nash_regrets = expected_utils - nash_value
            
            # Calculate Nash equilibrium regret for RD NE
            rd_regrets = rd_expected_utils - rd_nash_value
            
            # Store results for this bootstrap sample
            bootstrap_results.append({
                'ne_regrets': nash_regrets,
                'ne_strategy': nash_strategy,
                'rd_regrets': rd_regrets,
                'rd_strategy': rd_strategy,
                'expected_utils': expected_utils,
                'nash_value': nash_value,
                'rd_nash_value': rd_nash_value
            })
        except Exception as e:
            print(f"Error in bootstrap sample {b}: {e}")
    
    print(f"Successfully completed {len(bootstrap_results)} bootstrap samples.")
    return bootstrap_results

def plot_bootstrap_iteration(bootstrap_results, statistic_key, agent_names, output_dir='bootstrap_analysis'):
    """
    Plot running bootstrap statistics to examine convergence as bootstrap sample size increases.
    
    Args:
        bootstrap_results: Dictionary containing bootstrap results
        statistic_key: Key in bootstrap_results for the statistic to analyze ('ne_regret', 'rd_regret', etc.)
        agent_names: Names of agents for labeling
        output_dir: Directory to save output plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data based on dictionary format
    if isinstance(bootstrap_results, dict) and statistic_key in bootstrap_results:
        data = bootstrap_results[statistic_key]
    else:
        print(f"Error: couldn't find {statistic_key} in bootstrap results")
        return
        
    data_array = np.array(data)
    n_samples = len(data)
    n_agents = data_array.shape[1]
    
    print(f"Creating bootstrap iteration plot for {statistic_key} with {n_samples} samples and {n_agents} agents")
    
    # Plotting setup
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()
    
    # Plot 1: Running mean
    ax = axes[0]
    running_means = np.zeros((n_samples, n_agents))
    for i in range(1, n_samples + 1):
        running_means[i-1] = np.mean(data_array[:i], axis=0)
    
    for agent_idx in range(n_agents):
        ax.plot(range(1, n_samples + 1), running_means[:, agent_idx], 
                label=agent_names[agent_idx] if agent_idx < len(agent_names) else f"Agent {agent_idx}")
    
    ax.set_title(f"Running Mean - {statistic_key}")
    ax.set_xlabel("Number of Bootstrap Samples")
    ax.set_ylabel("Mean Value")
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Running standard deviation
    ax = axes[1]
    running_stds = np.zeros((n_samples, n_agents))
    for i in range(2, n_samples + 1):  # Start at 2 to avoid std with single sample
        running_stds[i-1] = np.std(data_array[:i], axis=0, ddof=1)
    
    for agent_idx in range(n_agents):
        ax.plot(range(2, n_samples + 1), running_stds[1:, agent_idx], 
                label=agent_names[agent_idx] if agent_idx < len(agent_names) else f"Agent {agent_idx}")
    
    ax.set_title(f"Running Standard Deviation - {statistic_key}")
    ax.set_xlabel("Number of Bootstrap Samples")
    ax.set_ylabel("Standard Deviation")
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Running Monte Carlo error (SE = std / sqrt(n))
    ax = axes[2]
    running_errors = np.zeros((n_samples, n_agents))
    for i in range(2, n_samples + 1):
        try:
            std_values = np.std(data_array[:i], axis=0, ddof=1)
            # Use float conversion to ensure we get a float sqrt result
            sqrt_i = float(np.sqrt(i))
            running_errors[i-1] = std_values / sqrt_i
        except Exception as e:
            print(f"Error processing window data: {e}")
            print(f"                            Window data shape: {data_array[:i].shape}")
            running_errors[i-1] = np.zeros(n_agents)
    
    for agent_idx in range(n_agents):
        # Use actual agent name instead of generic "Agent {agent_idx}"
        agent_label = agent_names[agent_idx] if agent_idx < len(agent_names) else f"Agent {agent_idx}"
        ax.plot(range(2, n_samples + 1), running_errors[1:, agent_idx], label=agent_label)
    
    ax.set_title(f"Running Monte Carlo Error - {statistic_key}")
    ax.set_xlabel("Number of Bootstrap Samples")
    ax.set_ylabel("Monte Carlo Error")
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Running 95% confidence intervals (using percentiles)
    ax = axes[3]
    lower_percentile = 2.5
    upper_percentile = 97.5
    
    # Only calculate after we have at least 10 samples
    start_idx = min(10, n_samples - 1)
    
    running_lower_ci = np.zeros((n_samples - start_idx, n_agents))
    running_upper_ci = np.zeros((n_samples - start_idx, n_agents))
    running_ci_width = np.zeros((n_samples - start_idx, n_agents))
    
    for i in range(start_idx, n_samples):
        running_lower_ci[i-start_idx] = np.percentile(data_array[:i+1], lower_percentile, axis=0)
        running_upper_ci[i-start_idx] = np.percentile(data_array[:i+1], upper_percentile, axis=0)
        running_ci_width[i-start_idx] = running_upper_ci[i-start_idx] - running_lower_ci[i-start_idx]
    
    for agent_idx in range(n_agents):
        ax.plot(range(start_idx, n_samples), running_ci_width[:, agent_idx], 
                label=agent_names[agent_idx] if agent_idx < len(agent_names) else f"Agent {agent_idx}")
    
    ax.set_title(f"Running CI Width - {statistic_key}")
    ax.set_xlabel("Number of Bootstrap Samples")
    ax.set_ylabel("95% CI Width")
    ax.grid(True, alpha=0.3)
    
    # Add a legend - shared for all subplots to save space
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=min(5, n_agents), bbox_to_anchor=(0.5, 0))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)  # Make room for the legend
    
    # Save the figure
    filename = f"{statistic_key}_bootstrap_iteration.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Saved bootstrap iteration plot to {os.path.join(output_dir, filename)}")
    
    return running_means, running_stds, running_errors, running_ci_width

def plot_confidence_interval_stability(bootstrap_results, statistic_key, agent_names, output_dir='bootstrap_analysis'):
    """
    Plot confidence interval stability to determine if more bootstrap samples are needed.
    
    Args:
        bootstrap_results: Dictionary containing bootstrap results
        statistic_key: Key in bootstrap_results for the statistic to analyze ('ne_regret', 'rd_regret', etc.)
        agent_names: Names of agents for labeling
        output_dir: Directory to save output plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data based on dictionary format
    if isinstance(bootstrap_results, dict) and statistic_key in bootstrap_results:
        data = bootstrap_results[statistic_key]
    else:
        print(f"Error: couldn't find {statistic_key} in bootstrap results")
        return
        
    data_array = np.array(data)
    n_samples = len(data)
    n_agents = data_array.shape[1]
    
    print(f"Creating confidence interval stability plot for {statistic_key}")
    
    # Plot confidence interval bounds as sample size increases
    fig, axes = plt.subplots(n_agents, 1, figsize=(12, 4 * n_agents), sharex=True)
    
    # Handle case with single agent
    if n_agents == 1:
        axes = [axes]
    
    # Define confidence levels to check stability across
    confidence_levels = [0.90, 0.95, 0.99]
    colors = ['blue', 'green', 'red']
    
    # Start after accumulating at least 10 samples
    start_idx = min(10, n_samples - 1)
    
    for agent_idx in range(n_agents):
        ax = axes[agent_idx]
        agent_name = agent_names[agent_idx] if agent_idx < len(agent_names) else f"Agent {agent_idx}"
        
        for ci_idx, confidence in enumerate(confidence_levels):
            lower_percentile = (1 - confidence) / 2 * 100
            upper_percentile = (1 - (1 - confidence) / 2) * 100
            
            running_lower_ci = np.zeros(n_samples - start_idx)
            running_upper_ci = np.zeros(n_samples - start_idx)
            
            for i in range(start_idx, n_samples):
                running_lower_ci[i-start_idx] = np.percentile(data_array[:i+1, agent_idx], lower_percentile)
                running_upper_ci[i-start_idx] = np.percentile(data_array[:i+1, agent_idx], upper_percentile)
            
            # Plot the CIs
            x_vals = range(start_idx, n_samples)
            ax.plot(x_vals, running_lower_ci, '--', color=colors[ci_idx], alpha=0.7, 
                    label=f"{confidence*100:.0f}% CI Lower")
            ax.plot(x_vals, running_upper_ci, '-', color=colors[ci_idx], alpha=0.7, 
                    label=f"{confidence*100:.0f}% CI Upper")
            
            # Fill between the CIs
            ax.fill_between(x_vals, running_lower_ci, running_upper_ci, color=colors[ci_idx], alpha=0.1)
        
        # Add a line for the final mean
        final_mean = np.mean(data_array[:, agent_idx])
        ax.axhline(final_mean, color='black', linestyle='-', label="Mean")
        
        ax.set_title(f"{agent_name} - {statistic_key} Confidence Interval Stability")
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.3)
        
        # Only add legend to the first subplot to save space
        if agent_idx == 0:
            ax.legend()
    
    axes[-1].set_xlabel("Number of Bootstrap Samples")
    plt.tight_layout()
    
    # Save the figure
    filename = f"{statistic_key}_ci_stability.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Saved confidence interval stability plot to {os.path.join(output_dir, filename)}")

def analyze_bootstrap_results_for_convergence(bootstrap_results, agent_names, output_dir='bootstrap_analysis'):
    """
    Analyze bootstrap results for convergence based on the bootstrap paper methods.
    This helps determine if you need more bootstrap samples or more simulator data.
    
    Args:
        bootstrap_results: Dictionary containing bootstrap results
        agent_names: Names of agents for labeling
        output_dir: Directory to save output plots
    
    Returns:
        Dictionary with convergence metrics and assessment
    """
    os.makedirs(output_dir, exist_ok=True)
    print("\nAnalyzing bootstrap convergence using methods from the bootstrap paper...")
    
    # Define the statistics to analyze
    statistics = []
    
    # Check which statistics are available in the bootstrap_results
    if 'ne_regret' in bootstrap_results and bootstrap_results['ne_regret']:
        statistics.append(('ne_regret', 'Nash Equilibrium Regrets'))
    
    if 'rd_regret' in bootstrap_results and bootstrap_results['rd_regret']:
        statistics.append(('rd_regret', 'Replicator Dynamics Regrets'))
    
    if 'agent_expected_utility' in bootstrap_results and bootstrap_results['agent_expected_utility']:
        statistics.append(('agent_expected_utility', 'Expected Utilities'))
    
    if not statistics:
        print("No valid statistics found in bootstrap results for convergence analysis.")
        return {}
    
    # Generate iteration plots and confidence interval stability plots for each statistic
    convergence_results = {}
    
    for stat_key, stat_name in statistics:
        print(f"\nAnalyzing {stat_name}...")
        
        # Generate bootstrap iteration plots
        running_means, running_stds, running_errors, running_ci_width = plot_bootstrap_iteration(
            bootstrap_results, stat_key, agent_names, output_dir
        )
        
        # Generate confidence interval stability plots
        plot_confidence_interval_stability(
            bootstrap_results, stat_key, agent_names, output_dir
        )
        
        # Calculate final Monte Carlo errors and convergence assessment
        data = bootstrap_results[stat_key]
        data_array = np.array(data)
        n_samples = len(data)
        n_agents = data_array.shape[1]
        
        # Calculate final Monte Carlo errors (SE = std / sqrt(n))
        final_means = np.mean(data_array, axis=0)
        final_stds = np.std(data_array, axis=0, ddof=1)
        final_errors = final_stds / np.sqrt(n_samples)
        
        # Calculate relative Monte Carlo errors (error / |mean|)
        # For very small means close to zero, use the std instead to avoid division by zero
        relative_errors = np.zeros(n_agents)
        for i in range(n_agents):
            if abs(final_means[i]) > 1e-8:
                relative_errors[i] = abs(final_errors[i] / final_means[i])
            else:
                relative_errors[i] = final_errors[i] / max(final_stds[i], 1e-8)
        
        # Assess convergence based on relative errors and CI stability
        # Rule of thumb: relative error < 0.05 (5%) is good, < 0.01 (1%) is excellent
        convergence_assessment = [""] * n_agents
        for i in range(n_agents):
            if relative_errors[i] < 0.01:
                convergence_assessment[i] = "Excellent"
            elif relative_errors[i] < 0.05:
                convergence_assessment[i] = "Good"
            elif relative_errors[i] < 0.10:
                convergence_assessment[i] = "Fair"
            else:
                convergence_assessment[i] = "Poor - More samples needed"
        
        # Store results
        convergence_results[stat_key] = {
            'final_means': final_means,
            'final_stds': final_stds,
            'monte_carlo_errors': final_errors,
            'relative_errors': relative_errors,
            'convergence_assessment': convergence_assessment
        }
        
        # Print summary
        print(f"\n{stat_name} Monte Carlo Errors:")
        for i in range(n_agents):
            agent_name = agent_names[i] if i < len(agent_names) else f"Agent {i}"
            print(f"  {agent_name}: {final_errors[i]:.6f} (Relative: {relative_errors[i]:.2%}) - {convergence_assessment[i]}")
        
        # Save detailed convergence results to CSV
        results_df = pd.DataFrame({
            'Agent': agent_names[:n_agents],
            'Mean': final_means,
            'Std Dev': final_stds,
            'Monte Carlo Error': final_errors,
            'Relative Error': relative_errors,
            'Assessment': convergence_assessment
        })
        
        results_df.to_csv(os.path.join(output_dir, f"{stat_key}_convergence.csv"), index=False)
        print(f"Saved detailed convergence results to {os.path.join(output_dir, f'{stat_key}_convergence.csv')}")
    
    return convergence_results

def check_bootstrap_convergence(bootstrap_results, agent_names, window_size=None, verbose=True):
    """
    Check if bootstrap results have converged by examining the stability of statistics.
    
    Args:
        bootstrap_results: Dictionary of bootstrap results
        agent_names: List of agent names
        window_size: Size of the window for checking stability (default: 20% of samples)
        verbose: Whether to print detailed output
        
    Returns:
        Dictionary with convergence status for different metrics
    """
    # Extract key statistics
    try:
        ne_regrets = bootstrap_results.get('ne_regret', [])
        
        # Try both keys for expected utilities (for backward compatibility)
        expected_utils = bootstrap_results.get('agent_expected_utility', bootstrap_results.get('expected_utility', []))
        
        rd_regrets = bootstrap_results.get('rd_regret', [])
        
        n_samples = len(ne_regrets) if ne_regrets else 0
        if n_samples == 0:
            print("No bootstrap samples available for convergence check.")
            return {'ne_converged': False, 'eu_converged': False, 'rd_converged': False}
        
        # Set default window size if not specified
        if window_size is None:
            window_size = max(1, n_samples // 5)  # Use 20% of samples
        
        window_size = min(window_size, n_samples - 1)  # Ensure window size is valid
        
        if verbose:
            print(f"\nUsing window size of {window_size} for convergence analysis")
            print(f"Number of NE regret samples: {len(ne_regrets)}")
            print(f"Number of expected utility samples: {len(expected_utils)}")
            print(f"Number of RD regret samples: {len(rd_regrets)}")
        
        # Calculate Monte Carlo errors with error handling
        def calculate_monte_carlo_error(data, agent_labels=None):
            try:
                # Check if we have any data
                if not data or len(data) == 0:
                    print("Warning: Empty data array in Monte Carlo error calculation")
                    return np.array([0.0])
                    
                # Convert to numpy array with proper error handling
                try:
                    # Try to handle both array and scalar data
                    if isinstance(data[0], (np.float64, float, int, np.int64)) or (hasattr(data[0], 'shape') and len(data[0].shape) == 0):
                        # Convert scalar data to proper array first
                        data_array = np.array(data, dtype=np.float64).reshape(-1, 1)
                    else:
                        # Use standard array calculation
                        data_array = np.array(data, dtype=np.float64)
                    
                    # Ensure proper dimensions
                    if len(data_array.shape) == 1:
                        data_array = data_array.reshape(-1, 1)
                    
                    # Calculate statistics
                    means = np.mean(data_array, axis=0)
                    stds = np.std(data_array, axis=0, ddof=1)
                    
                    # Handle division by sqrt manually to avoid type errors
                    n = len(data_array)
                    if n > 1:
                        sqrt_n = float(np.sqrt(n))  # Explicitly convert to float
                        mc_errors = stds / sqrt_n
                        
                        # Add agent names to the output if provided
                        if agent_labels and verbose:
                            for i, (error, label) in enumerate(zip(mc_errors, agent_labels)):
                                print(f"{label}: {error:.6f}")
                        return mc_errors
                    else:
                        return np.zeros_like(means)
                        
                except (ValueError, TypeError) as e:
                    print(f"Warning: Error converting data for Monte Carlo calculation: {e}")
                    # Create placeholder results with agent names
                    errors = np.zeros(len(agent_labels) if agent_labels else 1)
                    if agent_labels and verbose:
                        for i, label in enumerate(agent_labels):
                            print(f"{label}: 0.000000")
                    return errors
                    
            except Exception as e:
                print(f"Error calculating Monte Carlo error: {e}")
                # Return zero errors with agent names
                errors = np.zeros(len(agent_labels) if agent_labels else 1)
                if agent_labels and verbose:
                    for i, label in enumerate(agent_labels):
                        print(f"{label}: 0.000000")
                return errors
        
        # Check convergence of a metric
        def check_metric_convergence(data_array, threshold=0.05):
            try:
                if len(data_array) < window_size + 1:
                    return False, "Not enough samples"
                
                # Convert to numpy array if needed
                if not isinstance(data_array, np.ndarray):
                    data_array = np.array(data_array, dtype=np.float64)
                
                # Check if array shape is correct
                if len(data_array.shape) == 1:
                    data_array = data_array.reshape(-1, 1)
                
                # Calculate statistics for final window vs full dataset
                window_mean = np.mean(data_array[-window_size:], axis=0)
                full_mean = np.mean(data_array, axis=0)
                
                # Calculate relative difference
                abs_diff = np.abs(window_mean - full_mean)
                rel_diff = np.where(
                    np.abs(full_mean) > 1e-10,
                    abs_diff / np.abs(full_mean),
                    np.zeros_like(abs_diff)
                )
                
                # Check if all agents have converged
                max_rel_diff = np.max(rel_diff)
                
                return max_rel_diff < threshold, f"Max relative difference: {max_rel_diff:.4f}"
            except Exception as e:
                print(f"Error checking convergence: {e}")
                return False, f"Error in convergence check: {e}"
        
        # Check convergence of the three key metrics
        ne_converged, ne_status = check_metric_convergence(ne_regrets)
        eu_converged, eu_status = check_metric_convergence(expected_utils)
        rd_converged, rd_status = check_metric_convergence(rd_regrets)
        
        # Print convergence summary
        if verbose:
            print("\nBootstrap Convergence Analysis:")
            print("--------------------------------------------------")
            
            print("\nNash Equilibrium Regrets:")
            print(f"Status: {ne_status}")
            print("Monte Carlo Errors:")
            ne_errors = calculate_monte_carlo_error(ne_regrets, agent_names)
            
            print("\nExpected Utilities:")
            print(f"Status: {eu_status}")
            print("Monte Carlo Errors:")
            eu_errors = calculate_monte_carlo_error(expected_utils, agent_names)
            
            print("\nReplicator Dynamics Regrets:")
            print(f"Status: {rd_status}")
            print("Monte Carlo Errors:")
            rd_errors = calculate_monte_carlo_error(rd_regrets, agent_names)
            
            if not (ne_converged and eu_converged and rd_converged):
                print("\nWARNING: Some statistics have not converged. Consider increasing the number of bootstrap samples.")
        
        return {
            'ne_converged': ne_converged, 
            'eu_converged': eu_converged, 
            'rd_converged': rd_converged,
            'ne_errors': ne_errors if 'ne_errors' in locals() else np.zeros(len(agent_names)),
            'eu_errors': eu_errors if 'eu_errors' in locals() else np.zeros(len(agent_names)),
            'rd_errors': rd_errors if 'rd_errors' in locals() else np.zeros(len(agent_names))
        }
    except Exception as e:
        print(f"Error during convergence analysis: {e}")
        return {
            'ne_converged': False, 
            'eu_converged': False, 
            'rd_converged': False,
            'ne_errors': np.zeros(len(agent_names)),
            'eu_errors': np.zeros(len(agent_names)),
            'rd_errors': np.zeros(len(agent_names))
        }

def analyze_bootstrap_convergence(bootstrap_results, agent_names, window_size=None, threshold=0.05, output_dir=None):
    """
    Analyze convergence of bootstrap results with detailed statistics and visualization options.
    
    Args:
        bootstrap_results: Dictionary containing bootstrap results
        agent_names: List of agent names
        window_size: Size of the window for checking stability (default: 20% of samples)
        threshold: Threshold for determining convergence (default: 0.05)
        output_dir: Directory to save output plots and statistics
        
    Returns:
        Dictionary with convergence status and detailed metrics
    """
    print("\nAnalyzing bootstrap convergence using methods from the bootstrap paper...")
    
    # First, use the basic convergence check
    basic_convergence = check_bootstrap_convergence(bootstrap_results, agent_names, window_size)
    
    # If output_dir is provided, run the more comprehensive analysis
    if output_dir is not None:
        try:
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Run enhanced analysis that creates plots and saves results
            enhanced_results = analyze_bootstrap_results_for_convergence(
                bootstrap_results, agent_names, output_dir
            )
            
            # Combine the results from both analyses
            combined_results = {**basic_convergence, 'enhanced': enhanced_results}
            return combined_results
        except Exception as e:
            print(f"Error running enhanced convergence analysis: {e}")
            print("Continuing with basic convergence analysis.")
    
    return basic_convergence

def plot_regret_distributions(regrets_list, agent_names, title="Regret Distribution", plot_type="histogram"):
    """
    Create distribution plots for regrets across agents.
    
    Args:
        regrets_list: List of regret arrays from bootstrap samples
        agent_names: List of agent names
        title: Title for the plot
        plot_type: Type of plot to create ('histogram', 'box', or 'running_mean')
        
    Returns:
        matplotlib.figure.Figure: The generated plot
    """
    try:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Memory optimization: limit number of samples if too many
        max_samples = 1000  # Set a reasonable limit
        if len(regrets_list) > max_samples:
            print(f"Warning: Too many bootstrap samples ({len(regrets_list)}). Using a random subset of {max_samples} samples.")
            import random
            random.seed(42)  # For reproducibility
            regrets_list = random.sample(regrets_list, max_samples)
        
        # Safely convert regrets to numpy array
        try:
            regrets_array = np.array(regrets_list, dtype=np.float64)
        except ValueError:
            print("Warning: Bootstrap samples have inconsistent shapes. Using a more flexible approach.")
            first_shape = np.shape(regrets_list[0])
            if len(first_shape) == 0:  # Handle scalar values
                regrets_array = np.array(regrets_list, dtype=np.float64).reshape(-1, 1)
            else:
                # Create array with the right shape and fill it as much as possible
                regrets_array = np.zeros((len(regrets_list), len(agent_names)), dtype=np.float64)
                for i, regret in enumerate(regrets_list):
                    try:
                        if np.array(regret).size == len(agent_names):
                            regrets_array[i] = np.array(regret).flatten()[:len(agent_names)]
                        else:
                            print(f"Warning: Skipping regret sample {i} due to shape mismatch")
                    except:
                        print(f"Warning: Error processing regret sample {i}")
        
        # Ensure axes reflect the full data range, including any positive values
        min_regret = np.nanmin(regrets_array)
        max_regret = np.nanmax(regrets_array)
        
        # Print range information for debugging (without showing on plot)
        print(f"\nRegret range for {title}: [{min_regret:.8f}, {max_regret:.8f}]")
        
        # Check for positive regrets (just log it, don't annotate)
        if max_regret > 0:
            print(f"WARNING: Detected positive regret values (max: {max_regret:.8f}) in {title}")
        
        # Create different types of plots based on plot_type
        if plot_type == "histogram":
            # Create a histogram style distribution plot
            for i, agent_name in enumerate(agent_names):
                if i < regrets_array.shape[1]:  # Ensure we're not out of bounds
                    agent_data = regrets_array[:, i]
                    # Remove any NaN values
                    agent_data = agent_data[~np.isnan(agent_data)]
                    sns.histplot(agent_data, label=agent_name, kde=True, alpha=0.6, ax=ax)
            
            plt.xlabel('Regret Value')
            plt.ylabel('Frequency')
            
            # Set x-axis range based on actual data, with a small buffer
            buffer = (max_regret - min_regret) * 0.05 if max_regret > min_regret else abs(min_regret) * 0.05
            ax.set_xlim(min_regret - buffer, max_regret + buffer)
            
        elif plot_type == "box":
            # Create a box plot with data validation
            valid_data = {}
            for i, agent_name in enumerate(agent_names):
                if i < regrets_array.shape[1]:  # Ensure we're not out of bounds
                    agent_data = regrets_array[:, i]
                    # Remove any NaN values
                    agent_data = agent_data[~np.isnan(agent_data)]
                    valid_data[agent_name] = agent_data
            
            df = pd.DataFrame(valid_data)
            sns.boxplot(data=df, ax=ax)
            
            plt.xlabel('Agent')
            plt.ylabel('Regret Value')
            plt.xticks(rotation=45, ha='right')
            
            # Set y-axis range based on actual data
            buffer = (max_regret - min_regret) * 0.05 if max_regret > min_regret else abs(min_regret) * 0.05
            ax.set_ylim(min_regret - buffer, max_regret + buffer)
        
        elif plot_type == "running_mean":
            # Create a running mean plot to show convergence
            window_size = max(1, len(regrets_list) // 10)  # Use 1/10 of samples for window
            
            running_mean_min = float('inf')
            running_mean_max = float('-inf')
            
            for i, agent_name in enumerate(agent_names):
                # Calculate running mean
                running_means = []
                for j in range(window_size, len(regrets_list) + 1):
                    mean_value = np.mean(regrets_array[j-window_size:j, i])
                    running_means.append(mean_value)
                    running_mean_min = min(running_mean_min, mean_value)
                    running_mean_max = max(running_mean_max, mean_value)
                
                plt.plot(range(window_size, len(regrets_list) + 1), running_means, label=agent_name)
            
            plt.xlabel('Number of Bootstrap Samples')
            plt.ylabel('Running Mean Regret')
            
            # Set y-axis range for running mean plots
            buffer = (running_mean_max - running_mean_min) * 0.05 if running_mean_max > running_mean_min else abs(running_mean_min) * 0.05
            ax.set_ylim(running_mean_min - buffer, running_mean_max + buffer)
        
        else:
            raise ValueError(f"Unsupported plot_type: {plot_type}. Use 'histogram', 'box', or 'running_mean'.")
        
        # Clean title - remove any constraint text
        if "Regret" in title and "Nash" in title:
            # Make sure the title doesn't have equilibrium constraints
            title = title.split("(")[0].strip()
        
        plt.title(title)
        plt.grid(True, alpha=0.3)
        
        # Add legend if not boxplot (boxplot has built-in labels)
        if plot_type != "box":
            plt.legend(loc='best')
        
        # Add a zero line for reference
        if min_regret < 0 and max_regret > 0:
            if plot_type == "box" or plot_type == "running_mean":
                ax.axhline(y=0, color='r', linestyle='-', alpha=0.3)
            else:
                ax.axvline(x=0, color='r', linestyle='-', alpha=0.3)
        
        plt.tight_layout()
        return fig
    except Exception as e:
        print(f"Error creating regret distribution plot: {e}")
        return None 