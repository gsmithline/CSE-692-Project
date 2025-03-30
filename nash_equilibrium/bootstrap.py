"""
Bootstrapping and statistical analysis utilities for game theoretic analysis.
"""

import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
from nash_equilibrium.nash_solver import (
    milp_max_sym_ent_2p,
    replicator_dynamics_nash,
    calculate_max_regret,
    minimize_max_regret,
    EPSILON
)

# Define a consistent numerical threshold for the entire module
EPSILON = EPSILON  # Use the imported EPSILON value

def bootstrap_performance_metrics(performance_matrix, num_bootstrap=1000, data_matrix=None):
    """
    Bootstrap analysis to estimate distributions of Nash equilibrium regrets.
    Supports both parametric and non-parametric bootstrapping.
    
    Args:
        performance_matrix: DataFrame with agent performance data (average performance for each pair)
        num_bootstrap: Number of bootstrap replicas (default: 1000)
        data_matrix: Dictionary of raw performance data {(agent1, agent2): [list of raw payoff values]}
                     If provided, performs non-parametric bootstrapping.
                     If None, performs parametric bootstrapping using the performance_matrix.
        
    """
    
    all_agents = performance_matrix.index.tolist()
    num_agents = len(all_agents)
    
    bootstrap_results = {
        'ne_regret': [],               # Nash equilibrium regret (should be ≤ 0)
        'ne_strategy': [],             # Nash equilibrium strategies
        'rd_regret': [],               # Replicator dynamics Nash regret (should be ≤ 0)
        'rd_strategy': [],             # Replicator dynamics Nash strategies
        'agent_expected_utility': [],  # Expected utility for each agent
        'nash_value': [],              # Nash equilibrium value for each bootstrap
        'bootstrapped_matrices': [],   # Store each bootstrap's performance matrix for reference
        'me_normal_regret': [],        # Normal regret for ME NE (unilateral deviation from NE)
        'rd_normal_regret': []         # Normal regret for RD NE (unilateral deviation from NE)
    }
    
    is_nonparametric = data_matrix is not None and len(data_matrix) > 0
    
    if is_nonparametric:
        print(f"Performing non-parametric bootstrapping with {num_bootstrap} samples...")
    else:
        print(f"Performing parametric bootstrapping with {num_bootstrap} samples...")
    
    for b in range(num_bootstrap):
        if b % 100 == 0 and b > 0:
            print(f"Processed {b} bootstrap samples...")
            
        bootstrap_matrix = performance_matrix.copy()
        
        if is_nonparametric:
            # Non-parametric bootstrapping: resample with replacement from raw data
            for agent1 in all_agents:
                for agent2 in all_agents:
                    key = (agent1, agent2)
                    
                    if key in data_matrix and len(data_matrix[key]) > 0:
                        # Resample with replacement from raw data
                        raw_values = data_matrix[key]
                        bootstrap_samples = np.random.choice(raw_values, size=len(raw_values), replace=True)
                        bootstrap_matrix.loc[agent1, agent2] = np.mean(bootstrap_samples)
        else:
            # Parametric bootstrapping: generate from normal distribution with same mean/std
            for agent1 in all_agents:
                for agent2 in all_agents:
                    # Get the original value
                    orig_value = bootstrap_matrix.loc[agent1, agent2]
                    if not pd.isna(orig_value):
                        std_dev = abs(orig_value) * 0.1  
                        bootstrap_matrix.loc[agent1, agent2] = np.random.normal(orig_value, std_dev)
        
        bootstrap_results['bootstrapped_matrices'].append(bootstrap_matrix)
        
        game_matrix_np = bootstrap_matrix.to_numpy()
        
        for i in range(game_matrix_np.shape[0]):
            for j in range(game_matrix_np.shape[1]):
                if np.isnan(game_matrix_np[i, j]):
                    col_mean = np.nanmean(game_matrix_np[:, j])
                    if not np.isnan(col_mean):
                        game_matrix_np[i, j] = col_mean
                    else:
                        row_mean = np.nanmean(game_matrix_np[i, :])
                        game_matrix_np[i, j] = row_mean if not np.isnan(row_mean) else 0
        
        try:
            # 1. Calculate Max Entropy Nash Equilibrium
            me_nash_strategy = milp_max_sym_ent_2p(game_matrix_np)
            
            # 2. Calculate Replicator Dynamics Nash Equilibrium
            rd_nash_strategy, _ = replicator_dynamics_nash(game_matrix_np)
            
            # 3. ME NE: Calculate expected utilities against the Nash mixture
            me_expected_utils = np.dot(game_matrix_np, me_nash_strategy)
            
            # 4. ME NE: Calculate Nash equilibrium value (expected utility of Nash mixture against itself)
            try:
                # Try matrix multiplication the numpy way first
                me_nash_value = me_nash_strategy.T @ game_matrix_np @ me_nash_strategy
                if hasattr(me_nash_value, 'item'):
                    me_nash_value = me_nash_value.item()  # Convert to scalar if it's a numpy array
            except Exception as e:
                # Fallback to manual calculation if matrix multiplication fails
                me_nash_value = sum(me_nash_strategy[i] * sum(game_matrix_np[i, j] * me_nash_strategy[j] 
                                                         for j in range(len(me_nash_strategy))) 
                                for i in range(len(me_nash_strategy)))
            
            # 5. ME NE: Compute Nash Equilibrium regret (expected_utils - nash_value)
            me_ne_regrets = me_expected_utils - me_nash_value
            
            me_normal_regrets = np.zeros(len(all_agents))
            for i in range(len(all_agents)):
                e_i = np.zeros(len(all_agents))
                e_i[i] = 1.0
                
                agent_payoff = me_expected_utils[i]
                
                me_normal_regrets[i] = agent_payoff - me_nash_value
                
            if np.any(me_ne_regrets > EPSILON):
                max_regret = np.max(me_ne_regrets)
                worst_agent_idx = np.argmax(me_ne_regrets)
                worst_agent = all_agents[worst_agent_idx]
                print(f"NOTE: Detected positive ME Nash regret ({max_regret:.10f}) for agent {worst_agent}. "
                     f"This may indicate a non-equilibrium strategy.")
            elif np.any(me_ne_regrets > 0):
                # If we have very small positive values, just note them
                max_regret = np.max(me_ne_regrets)
                worst_agent_idx = np.argmax(me_ne_regrets)
                worst_agent = all_agents[worst_agent_idx]
                print(f"NOTE: Detected small positive ME Nash regret ({max_regret:.10f}) for agent {worst_agent}.")
            
            # 8. ME NE: Validate that all normal regrets are at most 0
            if np.any(me_normal_regrets > EPSILON):
                max_normal_regret = np.max(me_normal_regrets)
                worst_normal_idx = np.argmax(me_normal_regrets)
                worst_normal_agent = all_agents[worst_normal_idx]
                print(f"NOTE: Detected positive ME normal regret ({max_normal_regret:.10f}) for agent {worst_normal_agent}. "
                     f"This may indicate a non-equilibrium strategy.")
            elif np.any(me_normal_regrets > 0):
                # If we have very small positive values, just note them
                max_normal_regret = np.max(me_normal_regrets)
                worst_normal_idx = np.argmax(me_normal_regrets)
                worst_normal_agent = all_agents[worst_normal_idx]
                print(f"NOTE: Detected small positive ME normal regret ({max_normal_regret:.10f}) for agent {worst_normal_agent}.")
            
            # 9. RD NE: Calculate expected utilities against the RD Nash mixture
            rd_expected_utils = np.dot(game_matrix_np, rd_nash_strategy)
            
            # 10. RD NE: Calculate Nash equilibrium value (expected utility of Nash mixture against itself)
            try:
                rd_nash_value = rd_nash_strategy.T @ game_matrix_np @ rd_nash_strategy
                if hasattr(rd_nash_value, 'item'):
                    rd_nash_value = rd_nash_value.item()  # Convert to scalar if it's a numpy array
            except Exception as e:
                # Fallback to manual calculation if matrix multiplication fails
                rd_nash_value = sum(rd_nash_strategy[i] * sum(game_matrix_np[i, j] * rd_nash_strategy[j] 
                                                         for j in range(len(rd_nash_strategy))) 
                                for i in range(len(rd_nash_strategy)))
            
            # 11. RD NE: Compute Nash Equilibrium regret (expected_utils - nash_value)
            rd_ne_regrets = rd_expected_utils - rd_nash_value
            
            # 12. CALCULATE NORMAL REGRET FOR RD NE
            rd_normal_regrets = np.zeros(len(all_agents))
            for i in range(len(all_agents)):
                # Agent i's payoff when deviating to pure strategy is the expected utility
                agent_payoff = rd_expected_utils[i]
                
                # Calculate normal regret: agent_payoff - nash_value
                rd_normal_regrets[i] = agent_payoff - rd_nash_value
            
            # 13. RD NE: Validate that all regrets are at most 0
            if np.any(rd_ne_regrets > EPSILON):
                max_regret = np.max(rd_ne_regrets)
                worst_agent_idx = np.argmax(rd_ne_regrets)
                worst_agent = all_agents[worst_agent_idx]
                print(f"NOTE: Detected positive RD Nash regret ({max_regret:.10f}) for agent {worst_agent}. "
                     f"This may indicate a non-equilibrium strategy.")
            elif np.any(rd_ne_regrets > 0):
                # If we have very small positive values, just note them
                max_regret = np.max(rd_ne_regrets)
                worst_agent_idx = np.argmax(rd_ne_regrets)
                worst_agent = all_agents[worst_agent_idx]
                print(f"NOTE: Detected small positive RD Nash regret ({max_regret:.10f}) for agent {worst_agent}.")
            
            # 14. RD NE: Validate that all normal regrets are at most 0
            if np.any(rd_normal_regrets > EPSILON):
                max_normal_regret = np.max(rd_normal_regrets)
                worst_normal_idx = np.argmax(rd_normal_regrets)
                worst_normal_agent = all_agents[worst_normal_idx]
                print(f"NOTE: Detected positive RD normal regret ({max_normal_regret:.10f}) for agent {worst_normal_agent}. "
                     f"This may indicate a non-equilibrium strategy.")
            elif np.any(rd_normal_regrets > 0):
                # If we have very small positive values, just note them
                max_normal_regret = np.max(rd_normal_regrets)
                worst_normal_idx = np.argmax(rd_normal_regrets)
                worst_normal_agent = all_agents[worst_normal_idx]
                print(f"NOTE: Detected small positive RD normal regret ({max_normal_regret:.10f}) for agent {worst_normal_agent}.")
            
            # Store the results for this bootstrap sample
            bootstrap_results['ne_regret'].append(me_ne_regrets)
            bootstrap_results['ne_strategy'].append(me_nash_strategy)
            bootstrap_results['rd_regret'].append(rd_ne_regrets)
            bootstrap_results['rd_strategy'].append(rd_nash_strategy)
            bootstrap_results['agent_expected_utility'].append(me_expected_utils)
            bootstrap_results['nash_value'].append(me_nash_value)
            bootstrap_results['me_normal_regret'].append(me_normal_regrets)
            bootstrap_results['rd_normal_regret'].append(rd_normal_regrets)
            
        except Exception as e:
            # Log the error but continue with other samples
            print(f"Error in bootstrap sample {b}: {e}")
    
    # Check if we have enough valid samples
    if len(bootstrap_results['ne_regret']) < 10:
        print(f"WARNING: Only {len(bootstrap_results['ne_regret'])} valid bootstrap samples. Results may not be statistically reliable.")
    
    # At least return what we have, even if very few samples
    if len(bootstrap_results['ne_regret']) == 0:
        print("No valid bootstrap samples were generated. Check for errors in the Nash computations.")
        return {key: [] for key in bootstrap_results.keys()}
    
    return bootstrap_results

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
            
        # Print warning but don't raise error or cap values
        print(f"NOTE: Large positive mean regret detected:\n{', '.join(error_msg)}")
        print("Continuing analysis with positive regrets. Results may indicate non-equilibrium strategies.")
        
    elif np.any(mean_ne_regrets > 0) or (has_rd_regrets and np.any(mean_rd_regrets > 0)) or \
         (has_me_normal_regrets and np.any(mean_me_normal_regrets > 0)) or \
         (has_rd_normal_regrets and np.any(mean_rd_normal_regrets > 0)):
        # For small positive regrets, just warn
        if np.any(mean_ne_regrets > 0):
            max_ne_regret = np.max(mean_ne_regrets)
            worst_idx_ne = np.argmax(mean_ne_regrets)
            worst_agent_ne = agent_names[worst_idx_ne]
            print(f"NOTE: Small positive mean ME Nash regret detected: {max_ne_regret:.10f} for agent {worst_agent_ne}")
        
        if has_rd_regrets and np.any(mean_rd_regrets > 0):
            max_rd_regret = np.max(mean_rd_regrets)
            worst_idx_rd = np.argmax(mean_rd_regrets)
            worst_agent_rd = agent_names[worst_idx_rd]
            print(f"NOTE: Small positive mean RD Nash regret detected: {max_rd_regret:.10f} for agent {worst_agent_rd}")
            
        if has_me_normal_regrets and np.any(mean_me_normal_regrets > 0):
            max_me_normal = np.max(mean_me_normal_regrets)
            worst_idx_me_normal = np.argmax(mean_me_normal_regrets)
            worst_agent_me_normal = agent_names[worst_idx_me_normal]
            print(f"NOTE: Small positive mean ME normal regret detected: {max_me_normal:.10f} for agent {worst_agent_me_normal}")
            
        if has_rd_normal_regrets and np.any(mean_rd_normal_regrets > 0):
            max_rd_normal = np.max(mean_rd_normal_regrets)
            worst_idx_rd_normal = np.argmax(mean_rd_normal_regrets)
            worst_agent_rd_normal = agent_names[worst_idx_rd_normal]
            print(f"NOTE: Small positive mean RD normal regret detected: {max_rd_normal:.10f} for agent {worst_agent_rd_normal}")
    
    # Calculate standard deviations
    std_ne_regrets = np.std(ne_regrets, axis=0)
    std_expected_utils = np.std(expected_utils, axis=0)
    if has_rd_regrets:
        std_rd_regrets = np.std(rd_regrets, axis=0)
    if has_me_normal_regrets:
        std_me_normal_regrets = np.std(me_normal_regrets, axis=0)
    if has_rd_normal_regrets:
        std_rd_normal_regrets = np.std(rd_normal_regrets, axis=0)
    
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
    print(f"Average NE regret: {np.mean(mean_ne_regrets):.8f} (should be ≤ 0)")
    if has_rd_regrets:
        print(f"Average RD regret: {np.mean(mean_rd_regrets):.8f} (should be ≤ 0)")
    if has_me_normal_regrets:
        print(f"Average ME normal regret: {np.mean(mean_me_normal_regrets):.8f} (should be ≤ 0)")
    if has_rd_normal_regrets:
        print(f"Average RD normal regret: {np.mean(mean_rd_normal_regrets):.8f} (should be ≤ 0)")
    
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

def visualize_dual_regret(bootstrap_results, agent_names, figsize=(16, 10)):
    """
    Visualize Nash equilibrium regret types side by side
    
    Args:
        bootstrap_results: Dictionary with bootstrap samples
        agent_names: List of agent names
        figsize: Figure size tuple
        
    Returns:
        matplotlib figure
    """
    # Check if we have any regret data
    if not bootstrap_results.get('ne_regret'):
        print("Warning: No ME Nash regret data found in bootstrap results.")
        fig = plt.figure(figsize=figsize)
        plt.figtext(0.5, 0.5, "Nash regret data not available", 
                   ha='center', va='center', fontsize=14)
        plt.close()
        return fig
    
    # Check shapes and ensure we have consistent data
    try:
        # Get Max Entropy Nash equilibrium regrets
        me_ne_regrets = np.stack(bootstrap_results['ne_regret'])
        mean_me_ne_regrets = np.mean(me_ne_regrets, axis=0)
    except (ValueError, TypeError):
        print("Warning: ME Nash regrets have inconsistent shapes. Using a more robust approach.")
        # Get the number of agents
        num_agents = len(agent_names)
        # Initialize arrays for mean regrets
        mean_me_ne_regrets = np.zeros(num_agents)
        
        # Calculate means manually
        for agent_idx in range(num_agents):
            agent_regrets = []
            for regret_sample in bootstrap_results['ne_regret']:
                if agent_idx < len(regret_sample):
                    agent_regrets.append(float(regret_sample[agent_idx]))
            if agent_regrets:
                mean_me_ne_regrets[agent_idx] = sum(agent_regrets) / len(agent_regrets)
    
    # Check if RD regrets are available
    has_rd_regrets = 'rd_regret' in bootstrap_results and bootstrap_results['rd_regret']
    
    if has_rd_regrets:
        try:
            # Get Replicator Dynamics Nash equilibrium regrets
            rd_ne_regrets = np.stack(bootstrap_results['rd_regret'])
            mean_rd_ne_regrets = np.mean(rd_ne_regrets, axis=0)
        except (ValueError, TypeError):
            print("Warning: RD Nash regrets have inconsistent shapes. Using a more robust approach.")
            # Get the number of agents
            num_agents = len(agent_names)
            # Initialize arrays for mean regrets
            mean_rd_ne_regrets = np.zeros(num_agents)
            
            # Calculate means manually
            for agent_idx in range(num_agents):
                agent_regrets = []
                for regret_sample in bootstrap_results['rd_regret']:
                    if agent_idx < len(regret_sample):
                        agent_regrets.append(float(regret_sample[agent_idx]))
                if agent_regrets:
                    mean_rd_ne_regrets[agent_idx] = sum(agent_regrets) / len(agent_regrets)
    else:
        # If RD regrets are not available, use ME regrets as a fallback
        print("Warning: RD Nash regrets not found in bootstrap results. Using ME Nash regrets instead.")
        mean_rd_ne_regrets = mean_me_ne_regrets.copy()
    
    # Create a DataFrame with both regret types
    dual_regret_df = pd.DataFrame({
        'Agent': agent_names,
        'ME Nash Regret': mean_me_ne_regrets,  # Should be ≤ 0 at equilibrium
        'RD Nash Regret': mean_rd_ne_regrets   # Should be ≤ 0 at equilibrium
    })
    
    # Ensure all regrets are at most 0 (with small numerical tolerance)
    epsilon = 1e-10
    me_positive = dual_regret_df['ME Nash Regret'] > epsilon
    rd_positive = dual_regret_df['RD Nash Regret'] > epsilon
    
    if me_positive.any() or rd_positive.any():
        print(f"Warning: Some Nash regrets are positive. Displaying actual regret values without modification.")
        # No longer capping positive regrets at 0 as requested
        # dual_regret_df.loc[me_positive, 'ME Nash Regret'] = 0.0
        # dual_regret_df.loc[rd_positive, 'RD Nash Regret'] = 0.0
    
    # Sort agents by ME Nash Regret (closer to 0 is better, so descending)
    me_ne_regret_df = dual_regret_df.sort_values('ME Nash Regret', ascending=False)
    
    # Sort agents by RD Nash Regret (closer to 0 is better, so descending)
    rd_ne_regret_df = dual_regret_df.sort_values('RD Nash Regret', ascending=False)
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=figsize)
    fig.suptitle('Comparison of Nash Equilibrium Regrets\n(All values should be ≤ 0)', 
                fontsize=16, fontweight='bold')
    
    # Side-by-side bar charts
    ax1 = plt.subplot2grid((2, 2), (0, 0))
    ax2 = plt.subplot2grid((2, 2), (0, 1))
    
    # ME Nash Equilibrium Regret
    bars1 = ax1.barh(me_ne_regret_df['Agent'], me_ne_regret_df['ME Nash Regret'])
    ax1.set_title('Max Entropy Nash Equilibrium Regret', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Regret: expected_utils - nash_value\n(Closer to 0 is Better, Should be ≤ 0)', fontsize=12)
    ax1.axvline(x=0, color='black', linestyle='--', alpha=0.7)
    ax1.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Color bars based on value (closer to 0 is better)
    for i, bar in enumerate(bars1):
        value = me_ne_regret_df['ME Nash Regret'].iloc[i]
        bar.set_color('darkgreen')
        # Position text based on value
        text_offset = abs(me_ne_regret_df['ME Nash Regret'].min()) * 0.02
        ax1.text(value - text_offset, i, f'{value:.6f}', va='center', fontsize=9)
    
    # RD Nash Equilibrium Regret
    bars2 = ax2.barh(rd_ne_regret_df['Agent'], rd_ne_regret_df['RD Nash Regret'])
    ax2.set_title('Replicator Dynamics Nash Equilibrium Regret', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Regret: expected_utils - nash_value\n(Closer to 0 is Better, Should be ≤ 0)', fontsize=12)
    ax2.axvline(x=0, color='black', linestyle='--', alpha=0.7)
    ax2.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Color bars for RD regret
    for i, bar in enumerate(bars2):
        value = rd_ne_regret_df['RD Nash Regret'].iloc[i]
        bar.set_color('darkblue')
        text_offset = abs(rd_ne_regret_df['RD Nash Regret'].min()) * 0.02
        ax2.text(value - text_offset, i, f'{value:.6f}', va='center', fontsize=9)
    
    # Scatterplot comparing both regret types
    ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=2)
    
    scatter = ax3.scatter(dual_regret_df['ME Nash Regret'], 
                         dual_regret_df['RD Nash Regret'],
                         alpha=0.7, s=100, color='purple')
    
    # Add agent labels to scatter points
    for i, agent in enumerate(dual_regret_df['Agent']):
        x = dual_regret_df['ME Nash Regret'].iloc[i]
        y = dual_regret_df['RD Nash Regret'].iloc[i]
        ax3.annotate(agent, (x, y), fontsize=10, 
                    xytext=(5, 5), textcoords='offset points')
    
    ax3.set_xlabel('Max Entropy Nash Regret (Closer to 0 is Better)', fontsize=12)
    ax3.set_ylabel('Replicator Dynamics Nash Regret (Closer to 0 is Better)', fontsize=12)
    ax3.set_title('Comparison Between Nash Equilibrium Regret Types', fontsize=14, fontweight='bold')
    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.7)
    ax3.axvline(x=0, color='black', linestyle='--', alpha=0.7)
    ax3.grid(True, alpha=0.3)
    
    # Add reference line indicating where both regrets are equal
    min_val = min(dual_regret_df['ME Nash Regret'].min(), dual_regret_df['RD Nash Regret'].min())
    ax3.plot([min_val, 0], [min_val, 0], 'r--', alpha=0.5, label='Equal regrets')
    ax3.legend()
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    return fig

def plot_regret_distributions(bootstrap_results, agent_names, figsize=(12, 8), regret_type='ne_regret'):
    """
    Plot distributions of Nash equilibrium regrets for each agent
    
    Args:
        bootstrap_results: Dictionary with bootstrap samples
        agent_names: List of agent names
        figsize: Figure size
        regret_type: Type of regret to plot ('ne_regret' or 'rd_regret')
    """
    if regret_type not in bootstrap_results:
        print(f"No {regret_type} results to plot.")
        return None
    
    try:
        regrets = np.stack(bootstrap_results[regret_type])
    except ValueError:
        print(f"Warning: Bootstrap samples for {regret_type} have inconsistent shapes. Using a more flexible approach.")
        first_regret = bootstrap_results[regret_type][0]
        
        regrets = np.zeros((len(bootstrap_results[regret_type]), len(first_regret)), dtype=np.float64)
        
        for i, regret in enumerate(bootstrap_results[regret_type]):
            if len(regret) == len(first_regret):
                regrets[i] = regret
            else:
                print(f"Warning: Skipping regret sample {i} due to shape mismatch")
    
    n_agents = len(agent_names)
    
    fig, axs = plt.subplots(int(np.ceil(n_agents/3)), 3, figsize=figsize)
    axs = axs.flatten()
    
    title_prefix = "Max Entropy" if regret_type == 'ne_regret' else "Replicator Dynamics"
    regret_name = f"{title_prefix} Nash Equilibrium Regret"
    
    epsilon = EPSILON  # Use consistent epsilon threshold
    positive_regrets = (regrets > epsilon)
    
    if np.any(positive_regrets):
        count_positive = np.sum(positive_regrets)
        total_regrets = regrets.size
        percent_positive = (count_positive / total_regrets) * 100
        
        # Count regrets above epsilon by agent
        agent_violation_counts = np.sum(positive_regrets, axis=0)
        worst_agent_idx = np.argmax(agent_violation_counts)
        worst_agent = agent_names[worst_agent_idx]
        samples_per_agent = regrets.shape[0]
        
        print(f"Warning: {count_positive}/{total_regrets} regret values ({percent_positive:.2f}%) are above EPSILON={epsilon:.2e}.")
        print(f"         Worst agent: '{worst_agent}' with {agent_violation_counts[worst_agent_idx]}/{samples_per_agent} samples ({(agent_violation_counts[worst_agent_idx]/samples_per_agent)*100:.2f}%) showing positive regrets.")
    
    # Add a note to the title about positive regrets if they exist
    positive_note = f"\n({count_positive}/{total_regrets} samples have positive regrets)" if np.any(positive_regrets) else ""
    fig.suptitle(f"{regret_name} Distributions\n(Values should be ≤ 0 at equilibrium){positive_note}", 
                fontsize=16, fontweight='bold')
    
    for i, agent in enumerate(agent_names):
        if i < len(axs) and i < regrets.shape[1]:
            agent_regrets = regrets[:, i]
            
            # Calculate bin edges to cover the entire range of data
            min_regret = min(agent_regrets)
            max_regret = max(agent_regrets)
            # Ensure we have enough bins to represent the full distribution
            # Add a small padding to the min/max to guarantee no values are outside the range
            bin_edges = np.linspace(min_regret - abs(min_regret)*0.01, 
                                  max_regret + abs(max_regret)*0.01, 
                                  40)  # Use 40 bins for higher resolution
            
            # Plot histogram with original (uncapped) regrets
            axs[i].hist(agent_regrets, bins=bin_edges, alpha=0.7, 
                       color='darkgreen' if regret_type == 'ne_regret' else 'darkblue')
            axs[i].set_title(agent)
            axs[i].set_xlabel(regret_name)
            axs[i].set_ylabel('Frequency')
            
            # Add mean line
            mean_regret = np.mean(agent_regrets)
            axs[i].axvline(mean_regret, color='r', linestyle='--', 
                          label=f'Mean: {mean_regret:.6f}')
            
            # Add 95% CI
            lower_ci = np.percentile(agent_regrets, 2.5)
            upper_ci = np.percentile(agent_regrets, 97.5)
            axs[i].axvline(lower_ci, color='orange', linestyle=':')
            axs[i].axvline(upper_ci, color='orange', linestyle=':', 
                          label=f'95% CI: [{lower_ci:.6f}, {upper_ci:.6f}]')
            
            # Add a reference line at 0 with explicit label
            axs[i].axvline(0, color='black', linestyle='-', alpha=0.7, 
                          label='Zero regret (equilibrium)')
            
            # Set x-axis limit to ensure all data points are visible
            left_buffer = abs(min_regret) * 0.1  # Add 10% buffer on left side
            right_buffer = abs(max_regret) * 0.1  # Add 10% buffer on right side
            axs[i].set_xlim(min_regret - left_buffer, max_regret + right_buffer)
            
            # Add a text annotation explaining what the plot shows
            axs[i].text(0.5, 0.97, "Regret must be ≤ 0 at equilibrium", 
                      transform=axs[i].transAxes, ha='center', va='top',
                      bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'),
                      fontsize=8)
            
            # Add text showing the full range of regret values
            axs[i].text(0.5, 0.89, f"Full range: [{min_regret:.2f}, {max_regret:.2f}]", 
                      transform=axs[i].transAxes, ha='center', va='top',
                      bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'),
                      fontsize=8)
            
            # If there are positive regrets for this agent, add info
            agent_pos_count = np.sum(agent_regrets > epsilon)
            if agent_pos_count > 0:
                axs[i].text(0.5, 0.82, 
                          f"{agent_pos_count}/{len(agent_regrets)} samples ({(agent_pos_count/len(agent_regrets))*100:.1f}%) above EPSILON", 
                          transform=axs[i].transAxes, ha='center', va='top',
                          color='red', bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'),
                          fontsize=8)
            
            # Move legend to bottom left to avoid blocking data
            axs[i].legend(fontsize='small', loc='lower left')
    
    # Hide any unused subplots
    for j in range(i+1, len(axs)):
        axs[j].axis('off')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # Make room for the suptitle
    
    return fig

def plot_normal_regret_distributions(bootstrap_results, agent_names, figsize=(12, 8), regret_type='me_normal_regret'):
    """
    Plot distributions of normal regrets for each agent
    
    Args:
        bootstrap_results: Dictionary with bootstrap samples
        agent_names: List of agent names
        figsize: Figure size
        regret_type: Type of regret to plot ('me_normal_regret' or 'rd_normal_regret')
    """
    if regret_type not in bootstrap_results:
        print(f"No {regret_type} results to plot.")
        return None
    
    try:
        regrets = np.stack(bootstrap_results[regret_type])
    except ValueError:
        print(f"Warning: Bootstrap samples for {regret_type} have inconsistent shapes. Using a more flexible approach.")
        first_regret = bootstrap_results[regret_type][0]
        
        regrets = np.zeros((len(bootstrap_results[regret_type]), len(first_regret)), dtype=np.float64)
        
        for i, regret in enumerate(bootstrap_results[regret_type]):
            if len(regret) == len(first_regret):
                regrets[i] = regret
            else:
                print(f"Warning: Skipping regret sample {i} due to shape mismatch")
    
    n_agents = len(agent_names)
    
    fig, axs = plt.subplots(int(np.ceil(n_agents/3)), 3, figsize=figsize)
    axs = axs.flatten()
    
    title_prefix = "Max Entropy" if regret_type == 'me_normal_regret' else "Replicator Dynamics"
    regret_name = f"{title_prefix} Normal Regret"
    
    epsilon = EPSILON  # Use consistent epsilon threshold
    positive_regrets = (regrets > epsilon)
    
    if np.any(positive_regrets):
        count_positive = np.sum(positive_regrets)
        total_regrets = regrets.size
        percent_positive = (count_positive / total_regrets) * 100
        
        # Count regrets above epsilon by agent
        agent_violation_counts = np.sum(positive_regrets, axis=0)
        worst_agent_idx = np.argmax(agent_violation_counts)
        worst_agent = agent_names[worst_agent_idx]
        samples_per_agent = regrets.shape[0]
        
        print(f"Warning: {count_positive}/{total_regrets} normal regret values ({percent_positive:.2f}%) are above EPSILON={epsilon:.2e}.")
        print(f"         Worst agent: '{worst_agent}' with {agent_violation_counts[worst_agent_idx]}/{samples_per_agent} samples ({(agent_violation_counts[worst_agent_idx]/samples_per_agent)*100:.2f}%) showing positive normal regrets.")
    
    # Add a note to the title about positive normal regrets if they exist
    positive_note = f"\n({count_positive}/{total_regrets} samples have positive normal regrets)" if np.any(positive_regrets) else ""
    fig.suptitle(f"{regret_name} Distributions\n(Values should be ≤ 0 at equilibrium){positive_note}", 
                fontsize=16, fontweight='bold')
    
    for i, agent in enumerate(agent_names):
        if i < len(axs) and i < regrets.shape[1]:
            agent_regrets = regrets[:, i]
            
            # Calculate bin edges to cover the entire range of data
            min_regret = min(agent_regrets)
            max_regret = max(agent_regrets)
            # Ensure we have enough bins to represent the full distribution
            # Add a small padding to the min/max to guarantee no values are outside the range
            bin_edges = np.linspace(min_regret - abs(min_regret)*0.01, 
                                  max_regret + abs(max_regret)*0.01, 
                                  40)  # Use 40 bins for higher resolution
            
            # Plot histogram with original (uncapped) regrets
            axs[i].hist(agent_regrets, bins=bin_edges, alpha=0.7, 
                       color='darkblue' if regret_type == 'rd_normal_regret' else 'darkgreen')
            axs[i].set_title(agent)
            axs[i].set_xlabel(regret_name)
            axs[i].set_ylabel('Frequency')
            
            # Add mean line
            mean_regret = np.mean(agent_regrets)
            axs[i].axvline(mean_regret, color='r', linestyle='--', 
                          label=f'Mean: {mean_regret:.6f}')
            
            # Add 95% CI
            lower_ci = np.percentile(agent_regrets, 2.5)
            upper_ci = np.percentile(agent_regrets, 97.5)
            axs[i].axvline(lower_ci, color='orange', linestyle=':')
            axs[i].axvline(upper_ci, color='orange', linestyle=':', 
                          label=f'95% CI: [{lower_ci:.6f}, {upper_ci:.6f}]')
            
            # Add a reference line at 0 with explicit label
            axs[i].axvline(0, color='black', linestyle='-', alpha=0.7, 
                          label='Zero regret (equilibrium)')
            
            # Set x-axis limit to ensure all data points are visible
            left_buffer = abs(min_regret) * 0.1  # Add 10% buffer on left side
            right_buffer = abs(max_regret) * 0.1  # Add 10% buffer on right side
            axs[i].set_xlim(min_regret - left_buffer, max_regret + right_buffer)
            
            # Add a text annotation explaining what the plot shows
            axs[i].text(0.5, 0.97, "Normal Regret must be ≤ 0 at equilibrium", 
                      transform=axs[i].transAxes, ha='center', va='top',
                      bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'),
                      fontsize=8)
            
            # Add text showing the full range of regret values
            axs[i].text(0.5, 0.89, f"Full range: [{min_regret:.2f}, {max_regret:.2f}]", 
                      transform=axs[i].transAxes, ha='center', va='top',
                      bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'),
                      fontsize=8)
            
            # If there are positive normal regrets for this agent, add info
            agent_pos_count = np.sum(agent_regrets > epsilon)
            if agent_pos_count > 0:
                axs[i].text(0.5, 0.82, 
                          f"{agent_pos_count}/{len(agent_regrets)} samples ({(agent_pos_count/len(agent_regrets))*100:.1f}%) above EPSILON", 
                          transform=axs[i].transAxes, ha='center', va='top',
                          color='red', bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'),
                          fontsize=8)
            
            # Move legend to bottom left to avoid blocking data
            axs[i].legend(fontsize='small', loc='lower left')
    
    # Hide any unused subplots
    for j in range(i+1, len(axs)):
        axs[j].axis('off')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # Make room for the suptitle
    
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
        
        if agent1_mapped and agent2_mapped:
            matches_found += 1
            total_games[agent1_mapped][agent2_mapped] += 1
            total_games[agent2_mapped][agent1_mapped] += 1
            
            if result.get('final_action') == "ACCEPT":
                acceptance_count[agent1_mapped][agent2_mapped] += 1
                acceptance_count[agent2_mapped][agent1_mapped] += 1
    
    print(f"Found {matches_found} games with mapped agent names")
    
    for agent1 in agent_names:
        for agent2 in agent_names:
            if total_games[agent1][agent2] > 0:
                ratio = acceptance_count[agent1][agent2] / total_games[agent1][agent2]
                acceptance_matrix.loc[agent1, agent2] = ratio
    
    filled_cells = acceptance_matrix.count().sum()
    total_cells = len(agent_names) * len(agent_names)
    print(f"Filled {filled_cells} out of {total_cells} cells in acceptance matrix ({filled_cells/total_cells:.1%})")
    
    return acceptance_matrix

def prepare_raw_data_from_df(game_df, agent_names=None, value_col='agent1_value'):
    """
    Prepare raw data from a game results dataframe for bootstrapping
    
    Args:
        game_df: DataFrame with raw game results
        agent_names: List of agent names to include. If None, extracts unique agents from data
        value_col: Column name containing the performance/payoff value
        
    Returns:
        Tuple of (performance_matrix, raw_data_dict)
        - performance_matrix: pandas DataFrame with average values
        - raw_data_dict: Dictionary mapping agent pairs to lists of raw values
    """
    import numpy as np
    import pandas as pd
    import re
    
    if agent_names is None:
        # Extract unique agents from the game_df
        all_agents = set(game_df['agent1'].unique()) | set(game_df['agent2'].unique())
        agent_names = sorted(list(all_agents))
    
    # Normalize agent names by removing date patterns if present
    def normalize_agent_name(name):
        # Remove date patterns (e.g., 2025-02-19)
        return re.sub(r'_\d{4}-\d{2}-\d{2}', '', name)
    
    # Create a mapping to canonical names if needed
    agent_map = {}
    for name in game_df['agent1'].unique():
        normalized = normalize_agent_name(name)
        if normalized not in agent_map:
            agent_map[name] = normalized
    
    for name in game_df['agent2'].unique():
        normalized = normalize_agent_name(name)
        if normalized not in agent_map:
            agent_map[name] = normalized
    
    raw_data = {}
    for agent1 in agent_names:
        for agent2 in agent_names:
            raw_data[(agent1, agent2)] = []
    
    performance_matrix = pd.DataFrame(index=agent_names, columns=agent_names)
    
    for _, row in game_df.iterrows():
        agent1 = agent_map.get(row['agent1'], row['agent1'])
        agent2 = agent_map.get(row['agent2'], row['agent2'])
        
        if agent1 not in agent_names or agent2 not in agent_names:
            continue
        
        value1 = row.get(value_col)
        if value1 is not None and not pd.isna(value1):
            raw_data[(agent1, agent2)].append(float(value1))
        
        agent2_value_col = value_col.replace('agent1', 'agent2') if 'agent1' in value_col else None
        if agent2_value_col and agent2_value_col in row:
            value2 = row.get(agent2_value_col)
            if value2 is not None and not pd.isna(value2):
                raw_data[(agent2, agent1)].append(float(value2))
    
    for agent1 in agent_names:
        for agent2 in agent_names:
            values = raw_data[(agent1, agent2)]
            if values:
                performance_matrix.loc[agent1, agent2] = np.mean(values)
            else:
                performance_matrix.loc[agent1, agent2] = np.nan
    
    print(f"Prepared raw data for {len(agent_names)} agents")
    print(f"Number of agent pairs with data: {sum(1 for v in raw_data.values() if v)}")
    
    return performance_matrix, raw_data


def bootstrap_from_dataframe(game_df, agent_names=None, value_col='agent1_value', 
                           num_bootstrap=10000, confidence=0.95):
    """
    Perform bootstrap analysis directly from a game results dataframe
    
    Args:
        game_df: DataFrame with raw game results
        agent_names: List of agent names to include. If None, extracts unique agents from data
        value_col: Column name containing the performance/payoff value
        num_bootstrap: Number of bootstrap replicas
        confidence: Confidence level for intervals
        
    Returns:
        Dictionary of bootstrap results
    """
    # Prepare raw data from the dataframe
    performance_matrix, raw_data = prepare_raw_data_from_df(game_df, agent_names, value_col)
    
    # Run the bootstrap analysis using the raw data
    bootstrap_results = bootstrap_performance_metrics(
        performance_matrix, 
        num_bootstrap=num_bootstrap,
        data_matrix=raw_data
    )
    
    return bootstrap_results

def visualize_normal_regret_comparison(bootstrap_results, agent_names, figsize=(16, 10)):
    """
    Visualize and compare normal regrets for both ME NE and RD NE side by side
    
    Args:
        bootstrap_results: Dictionary with bootstrap samples
        agent_names: List of agent names
        figsize: Figure size tuple
        
    Returns:
        matplotlib figure
    """
    # Check if we have any normal regret data
    has_me_normal = 'me_normal_regret' in bootstrap_results and bootstrap_results['me_normal_regret']
    has_rd_normal = 'rd_normal_regret' in bootstrap_results and bootstrap_results['rd_normal_regret']
    
    if not has_me_normal and not has_rd_normal:
        print("Warning: No normal regret data found in bootstrap results.")
        fig = plt.figure(figsize=figsize)
        plt.figtext(0.5, 0.5, "Normal regret data not available", 
                   ha='center', va='center', fontsize=14)
        plt.close()
        return fig
    
    # Get the normal regret data
    if has_me_normal:
        try:
            me_normal_regrets = np.stack(bootstrap_results['me_normal_regret'])
            mean_me_normal = np.mean(me_normal_regrets, axis=0)
        except (ValueError, TypeError):
            print("Warning: ME normal regrets have inconsistent shapes. Using a more robust approach.")
            num_agents = len(agent_names)
            mean_me_normal = np.zeros(num_agents)
            
            for agent_idx in range(num_agents):
                agent_regrets = []
                for regret_sample in bootstrap_results['me_normal_regret']:
                    if agent_idx < len(regret_sample):
                        agent_regrets.append(float(regret_sample[agent_idx]))
                if agent_regrets:
                    mean_me_normal[agent_idx] = sum(agent_regrets) / len(agent_regrets)
    else:
        mean_me_normal = np.zeros(len(agent_names))
    
    if has_rd_normal:
        try:
            rd_normal_regrets = np.stack(bootstrap_results['rd_normal_regret'])
            mean_rd_normal = np.mean(rd_normal_regrets, axis=0)
        except (ValueError, TypeError):
            print("Warning: RD normal regrets have inconsistent shapes. Using a more robust approach.")
            num_agents = len(agent_names)
            mean_rd_normal = np.zeros(num_agents)
            
            for agent_idx in range(num_agents):
                agent_regrets = []
                for regret_sample in bootstrap_results['rd_normal_regret']:
                    if agent_idx < len(regret_sample):
                        agent_regrets.append(float(regret_sample[agent_idx]))
                if agent_regrets:
                    mean_rd_normal[agent_idx] = sum(agent_regrets) / len(agent_regrets)
    else:
        mean_rd_normal = np.zeros(len(agent_names))
    
    # Create a DataFrame with both normal regret types
    normal_regret_df = pd.DataFrame({
        'Agent': agent_names,
        'ME Normal Regret': mean_me_normal,  # Should be ≤ 0 at equilibrium
        'RD Normal Regret': mean_rd_normal   # Should be ≤ 0 at equilibrium
    })
    
    # Check for positive regrets
    epsilon = EPSILON
    me_positive = normal_regret_df['ME Normal Regret'] > epsilon
    rd_positive = normal_regret_df['RD Normal Regret'] > epsilon
    
    # Store original values before capping
    original_df = normal_regret_df.copy()
    
    if me_positive.any() or rd_positive.any():
        me_count = me_positive.sum()
        rd_count = rd_positive.sum()
        total_agents = len(agent_names)
        
        if me_positive.any():
            me_worst_agent = normal_regret_df.loc[me_positive, 'Agent'].iloc[np.argmax(normal_regret_df.loc[me_positive, 'ME Normal Regret'])]
            me_max_regret = normal_regret_df.loc[me_positive, 'ME Normal Regret'].max()
            print(f"Warning: {me_count}/{total_agents} agents ({me_count/total_agents*100:.1f}%) have positive ME normal regrets.")
            print(f"         Worst agent: '{me_worst_agent}' with regret {me_max_regret:.2e}")
            
        if rd_positive.any():
            rd_worst_agent = normal_regret_df.loc[rd_positive, 'Agent'].iloc[np.argmax(normal_regret_df.loc[rd_positive, 'RD Normal Regret'])]
            rd_max_regret = normal_regret_df.loc[rd_positive, 'RD Normal Regret'].max()
            print(f"Warning: {rd_count}/{total_agents} agents ({rd_count/total_agents*100:.1f}%) have positive RD normal regrets.")
            print(f"         Worst agent: '{rd_worst_agent}' with regret {rd_max_regret:.2e}")
            
       # print(f"Capping positive regrets at 0 for visualization.")
        #normal_regret_df.loc[me_positive, 'ME Normal Regret'] = 0.0
       # normal_regret_df.loc[rd_positive, 'RD Normal Regret'] = 0.0
    
    # Sort agents by ME Normal Regret (closer to 0 is better, so descending)
    me_normal_regret_df = normal_regret_df.sort_values('ME Normal Regret', ascending=False)
    
    # Sort agents by RD Normal Regret (closer to 0 is better, so descending)
    rd_normal_regret_df = normal_regret_df.sort_values('RD Normal Regret', ascending=False)
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=figsize)
    title_suffix = ""
    if me_positive.any() or rd_positive.any():
        me_str = f"{me_count}/{total_agents}" if me_positive.any() else "0"
        rd_str = f"{rd_count}/{total_agents}" if rd_positive.any() else "0"
        title_suffix = f"\n(ME: {me_str} agents, RD: {rd_str} agents with positive regrets were capped to 0)"
    
    fig.suptitle(f'Comparison of Normal Regrets\n(All values should be ≤ 0){title_suffix}', 
                fontsize=16, fontweight='bold')
    
    # Side-by-side bar charts
    ax1 = plt.subplot2grid((2, 2), (0, 0))
    ax2 = plt.subplot2grid((2, 2), (0, 1))
    
    # ME Normal Regret
    bars1 = ax1.barh(me_normal_regret_df['Agent'], me_normal_regret_df['ME Normal Regret'])
    ax1.set_title('Max Entropy Normal Regret', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Regret: agent_payoff - Nash equilibrium payoff\n(Closer to 0 is Better, Should be ≤ 0)', fontsize=12)
    ax1.axvline(x=0, color='black', linestyle='--', alpha=0.7)
    ax1.grid(axis='x', linestyle='--', alpha=0.7)
    
    for i, bar in enumerate(bars1):
        value = me_normal_regret_df['ME Normal Regret'].iloc[i]
        agent = me_normal_regret_df['Agent'].iloc[i]
        
        original_value = original_df.loc[original_df['Agent'] == agent, 'ME Normal Regret'].iloc[0]
        was_capped = original_value > epsilon
        
        bar.set_color('darkgreen' if not was_capped else 'lightgreen')
        
        text_offset = abs(me_normal_regret_df['ME Normal Regret'].min()) * 0.02
        text_value = f'{value:.6f}'
        if was_capped:
            text_value += f' (was {original_value:.6f})'
        ax1.text(value - text_offset, i, text_value, va='center', fontsize=9)
    
    bars2 = ax2.barh(rd_normal_regret_df['Agent'], rd_normal_regret_df['RD Normal Regret'])
    ax2.set_title('Replicator Dynamics Normal Regret', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Regret: agent_payoff - Nash equilibrium payoff\n(Closer to 0 is Better, Should be ≤ 0)', fontsize=12)
    ax2.axvline(x=0, color='black', linestyle='--', alpha=0.7)
    ax2.grid(axis='x', linestyle='--', alpha=0.7)
    
    for i, bar in enumerate(bars2):
        value = rd_normal_regret_df['RD Normal Regret'].iloc[i]
        agent = rd_normal_regret_df['Agent'].iloc[i]
        
        original_value = original_df.loc[original_df['Agent'] == agent, 'RD Normal Regret'].iloc[0]
        was_capped = original_value > epsilon
        
        bar.set_color('darkblue' if not was_capped else 'lightblue')
        
        text_offset = abs(rd_normal_regret_df['RD Normal Regret'].min()) * 0.02
        text_value = f'{value:.6f}'
        if was_capped:
            text_value += f' (was {original_value:.6f})'
        ax2.text(value - text_offset, i, text_value, va='center', fontsize=9)
    
    ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=2)
    
    me_capped = original_df['ME Normal Regret'] > epsilon
    rd_capped = original_df['RD Normal Regret'] > epsilon
    
    uncapped_mask = ~(me_capped | rd_capped)
    if np.any(uncapped_mask):
        ax3.scatter(normal_regret_df.loc[uncapped_mask, 'ME Normal Regret'], 
                   normal_regret_df.loc[uncapped_mask, 'RD Normal Regret'],
                   alpha=0.7, s=100, color='purple', label='Valid regrets')
    
    me_only_mask = me_capped & ~rd_capped
    if np.any(me_only_mask):
        ax3.scatter(normal_regret_df.loc[me_only_mask, 'ME Normal Regret'], 
                   normal_regret_df.loc[me_only_mask, 'RD Normal Regret'],
                   alpha=0.7, s=100, color='green', marker='s', label='ME regret capped')
    
    rd_only_mask = rd_capped & ~me_capped
    if np.any(rd_only_mask):
        ax3.scatter(normal_regret_df.loc[rd_only_mask, 'ME Normal Regret'], 
                   normal_regret_df.loc[rd_only_mask, 'RD Normal Regret'],
                   alpha=0.7, s=100, color='blue', marker='s', label='RD regret capped')
    
    both_capped_mask = me_capped & rd_capped
    if np.any(both_capped_mask):
        ax3.scatter(normal_regret_df.loc[both_capped_mask, 'ME Normal Regret'], 
                   normal_regret_df.loc[both_capped_mask, 'RD Normal Regret'],
                   alpha=0.7, s=100, color='red', marker='s', label='Both regrets capped')
    
    for i, agent in enumerate(normal_regret_df['Agent']):
        x = normal_regret_df['ME Normal Regret'].iloc[i]
        y = normal_regret_df['RD Normal Regret'].iloc[i]
        ax3.annotate(agent, (x, y), fontsize=10, 
                    xytext=(5, 5), textcoords='offset points')
    
    ax3.set_xlabel('Max Entropy Normal Regret (Closer to 0 is Better)', fontsize=12)
    ax3.set_ylabel('Replicator Dynamics Normal Regret (Closer to 0 is Better)', fontsize=12)
    ax3.set_title('Comparison Between Normal Regret Types', fontsize=14, fontweight='bold')
    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.7)
    ax3.axvline(x=0, color='black', linestyle='--', alpha=0.7)
    ax3.grid(True, alpha=0.3)
    
    min_val = min(normal_regret_df['ME Normal Regret'].min(), normal_regret_df['RD Normal Regret'].min())
    ax3.plot([min_val, 0], [min_val, 0], 'r--', alpha=0.5, label='Equal regrets')
    ax3.legend()
    
    ax3.text(0.5, -0.15, 
             "Normal regret measures how much an agent could gain by unilaterally deviating from the Nash equilibrium.\n"
             "In a true Nash equilibrium, no agent can improve their payoff by deviating, so all normal regrets must be ≤ 0.",
             transform=ax3.transAxes, ha='center', va='center', fontsize=10,
             bbox=dict(facecolor='lightyellow', alpha=0.5, boxstyle='round,pad=0.5'))
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9, bottom=0.15)  # Make room for suptitle and explanation
    
    return fig

def plot_nash_distributions(bootstrap_results, agents):
    """
    Plot the distributions of Nash equilibrium metrics.
    
    Args:
        bootstrap_results: Bootstrap results dictionary
        agents: List of agent names
        
    Returns:
        tuple: (me_ne_regret_fig, rd_ne_regret_fig, me_normal_regret_fig, rd_normal_regret_fig, normal_comparison_fig, dual_regret_fig)
    """
    me_ne_regret_fig = plot_regret_distributions(bootstrap_results, agents, regret_type='ne_regret')
    
    rd_ne_regret_fig = plot_regret_distributions(bootstrap_results, agents, regret_type='rd_regret')
    
    me_normal_regret_fig = plot_normal_regret_distributions(bootstrap_results, agents, regret_type='me_normal_regret')
    
    rd_normal_regret_fig = plot_normal_regret_distributions(bootstrap_results, agents, regret_type='rd_normal_regret')
    
    normal_comparison_fig = visualize_normal_regret_comparison(bootstrap_results, agents)
    
    dual_regret_fig = visualize_dual_regret(bootstrap_results, agents)
    
    return me_ne_regret_fig, rd_ne_regret_fig, me_normal_regret_fig, rd_normal_regret_fig, normal_comparison_fig, dual_regret_fig

def visualize_nash_mixture_with_ci(bootstrap_results, agent_names, figsize=(14, 8), confidence=0.95):
    """
    Visualize the average Nash equilibrium strategies from bootstrapping with confidence intervals.
    Shows both ME NE and RD NE distributions side by side if both are available.
    
    Args:
        bootstrap_results: Dictionary with bootstrap samples containing 'ne_strategy' and optionally 'rd_strategy'
        agent_names: List of agent names
        figsize: Figure size tuple
        confidence: Confidence level for intervals (default: 0.95)
        
    Returns:
        matplotlib figure
    """
    # Check if bootstrap results contain strategies
    if 'ne_strategy' not in bootstrap_results or not bootstrap_results['ne_strategy']:
        raise ValueError("Bootstrap results do not contain Nash equilibrium strategies")
    
    has_rd = 'rd_strategy' in bootstrap_results and bootstrap_results['rd_strategy']
    
    # Calculate percentile thresholds for confidence intervals
    alpha = 1 - confidence
    lower_percentile = alpha / 2 * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    # Process ME Nash strategies
    me_strategies = np.stack(bootstrap_results['ne_strategy'])
    mean_me_strategy = np.mean(me_strategies, axis=0)
    lower_me_ci = np.percentile(me_strategies, lower_percentile, axis=0)
    upper_me_ci = np.percentile(me_strategies, upper_percentile, axis=0)
    
    # Create DataFrame with ME Nash results
    me_results = pd.DataFrame({
        'Agent': agent_names,
        'ME Nash Probability': mean_me_strategy,
        f'Lower {confidence*100:.0f}% CI': lower_me_ci,
        f'Upper {confidence*100:.0f}% CI': upper_me_ci
    })
    
    # Sort by probability (descending)
    me_results = me_results.sort_values('ME Nash Probability', ascending=False)
    
    # Process RD Nash strategies if available
    if has_rd:
        rd_strategies = np.stack(bootstrap_results['rd_strategy'])
        mean_rd_strategy = np.mean(rd_strategies, axis=0)
        lower_rd_ci = np.percentile(rd_strategies, lower_percentile, axis=0)
        upper_rd_ci = np.percentile(rd_strategies, upper_percentile, axis=0)
        
        # Create DataFrame with RD Nash results
        rd_results = pd.DataFrame({
            'Agent': agent_names,
            'RD Nash Probability': mean_rd_strategy,
            f'Lower {confidence*100:.0f}% CI': lower_rd_ci,
            f'Upper {confidence*100:.0f}% CI': upper_rd_ci
        })
        
        # Sort by probability (descending)
        rd_results = rd_results.sort_values('RD Nash Probability', ascending=False)
    
    # Create visualization
    fig, axes = plt.subplots(1, 2 if has_rd else 1, figsize=figsize)
    
    # If only ME Nash is available, axes will be a single axis not a list
    if not has_rd:
        axes = [axes]
    
    # Plot ME Nash probabilities
    me_ax = axes[0]
    me_bars = me_ax.barh(me_results['Agent'], me_results['ME Nash Probability'], color='lightblue')
    
    # Add error bars for ME Nash
    me_ax.errorbar(
        me_results['ME Nash Probability'],
        me_results['Agent'],
        xerr=[
            me_results['ME Nash Probability'] - me_results[f'Lower {confidence*100:.0f}% CI'],
            me_results[f'Upper {confidence*100:.0f}% CI'] - me_results['ME Nash Probability']
        ],
        fmt='none',
        color='black',
        capsize=5
    )
    
    for i, bar in enumerate(me_bars):
        value = me_results['ME Nash Probability'].iloc[i]
        if value > 0.01:  # Only show labels for values above threshold
            me_ax.text(
                value + 0.01,
                bar.get_y() + bar.get_height()/2,
                f'{value:.3f}',
                va='center',
                fontweight='bold' if value > 0.1 else 'normal'
            )
    
    me_ax.set_title('Max Entropy Nash Equilibrium Mixture', fontsize=14, fontweight='bold')
    me_ax.set_xlabel('Probability', fontweight='bold')
    me_ax.set_xlim(0, 1.1)  # Set x-axis limit from 0 to 1.1 to accommodate labels
    me_ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Plot RD Nash probabilities if available
    if has_rd:
        rd_ax = axes[1]
        rd_bars = rd_ax.barh(rd_results['Agent'], rd_results['RD Nash Probability'], color='lightgreen')
        
        # Add error bars for RD Nash
        rd_ax.errorbar(
            rd_results['RD Nash Probability'],
            rd_results['Agent'],
            xerr=[
                rd_results['RD Nash Probability'] - rd_results[f'Lower {confidence*100:.0f}% CI'],
                rd_results[f'Upper {confidence*100:.0f}% CI'] - rd_results['RD Nash Probability']
            ],
            fmt='none',
            color='black',
            capsize=5
        )
        
        # Add RD Nash value labels
        for i, bar in enumerate(rd_bars):
            value = rd_results['RD Nash Probability'].iloc[i]
            if value > 0.01:  # Only show labels for values above threshold
                rd_ax.text(
                    value + 0.01,
                    bar.get_y() + bar.get_height()/2,
                    f'{value:.3f}',
                    va='center',
                    fontweight='bold' if value > 0.1 else 'normal'
                )
        
        rd_ax.set_title('Replicator Dynamics Nash Equilibrium Mixture', fontsize=14, fontweight='bold')
        rd_ax.set_xlabel('Probability', fontweight='bold')
        rd_ax.set_yticklabels([])  # Hide y-axis labels since they're shown in the ME plot
        rd_ax.set_xlim(0, 1.1)  # Set x-axis limit from 0 to 1.1 to accommodate labels
        rd_ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Add a suptitle
    plt.suptitle(
        f'Average Nash Equilibrium Mixes ({confidence*100:.0f}% Confidence Intervals)',
        fontsize=16,
        fontweight='bold',
        y=0.98
    )
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # Make room for suptitle
    
    return fig