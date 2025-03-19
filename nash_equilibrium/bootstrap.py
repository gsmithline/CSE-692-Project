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
from nash_equilibrium.nash_solver import milp_max_sym_ent_2p

def bootstrap_performance_metrics(performance_matrix, num_bootstrap=10000, sample_with_replacement=True, 
                                 confidence=0.95, min_samples_warning=5, data_matrix=None):
    """
    Bootstrapping to estimate distributions of performance metrics with proper resampling
    
    Args:
        performance_matrix: DataFrame with agent performance data (average performance for each pair)
        num_bootstrap: Number of bootstrap replicas (default: 10,000)
        sample_with_replacement: Whether to sample with replacement
        confidence: Confidence level for intervals (default: 0.95)
        min_samples_warning: Threshold for warning about too few samples
        data_matrix: Optional dictionary of raw performance data {(agent1, agent2): [list of raw payoff values]}
                     If provided, bootstrap will directly resample from raw data
        
    Returns:
        Dictionary of bootstrap results with statistics and confidence intervals
    """
    all_agents = performance_matrix.index.tolist()
    num_agents = len(all_agents)
    
    # Check if any profiles have very few samples
    non_nan_counts = performance_matrix.notna().sum().sum()
    if non_nan_counts < min_samples_warning * num_agents:
        warnings.warn(f"Performance matrix has fewer than {min_samples_warning} samples per agent on average. "
                      f"Bootstrap intervals may be unreliable.", UserWarning)
    
    bootstrap_results = {
        'ne_regret': [],               # Relative performance (expected_utils - nash_value)
        'traditional_regret': [],      # Traditional regret (max_utils - expected_utils)
        'ne_strategy': [],
        'agent_expected_utility': [],
        'agent_max_utility': [],
        'nash_value': []               # Nash equilibrium value for each bootstrap
    }
    
    print(f"Generating {num_bootstrap} bootstrap samples...")
    
    # Determine bootstrapping approach based on available data
    has_raw_data = data_matrix is not None and len(data_matrix) > 0
    
    if has_raw_data:
        print("Using raw performance data for true non-parametric bootstrapping")
    else:
        print("No raw data provided. Using parametric bootstrapping with normal noise (5% of cell value)")
        # Store the non-NaN values in the original matrix for noise-based resampling
        valid_values = {}
        for agent1 in all_agents:
            for agent2 in all_agents:
                if not np.isnan(performance_matrix.loc[agent1, agent2]):
                    key = (agent1, agent2)
                    valid_values[key] = performance_matrix.loc[agent1, agent2]
    
    for i in range(num_bootstrap):
        if i % 1000 == 0 and i > 0:
            print(f"Processed {i} bootstrap samples...")
            
        # Create a bootstrap sample
        bootstrap_matrix = performance_matrix.copy()
        
        if sample_with_replacement:
            for agent1 in all_agents:
                for agent2 in all_agents:
                    key = (agent1, agent2)
                    
                    if has_raw_data and key in data_matrix and len(data_matrix[key]) > 0:
                        # True bootstrapping: resample with replacement from raw data
                        raw_values = data_matrix[key]
                        # Randomly select samples with replacement
                        bootstrap_samples = np.random.choice(raw_values, size=len(raw_values), replace=True)
                        # Use the mean of bootstrap samples as the new value
                        bootstrap_matrix.loc[agent1, agent2] = np.mean(bootstrap_samples)
                    elif not has_raw_data and key in valid_values:
                        # Parametric approximation when we only have single observations
                        original_value = valid_values[key]
                        noise_level = 0.05 * abs(original_value) if original_value != 0 else 0.01
                        bootstrap_matrix.loc[agent1, agent2] = np.random.normal(original_value, noise_level)
        
        # Convert to numpy array for computation
        game_matrix_np = bootstrap_matrix.to_numpy()
        
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
        
        try:
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
            
            # Convert to consistent data types
            nash_strategy = np.array(nash_strategy, dtype=np.float64)
            expected_utils = np.array(expected_utils, dtype=np.float64)
            max_utils = np.array(max_utils, dtype=np.float64)
            relative_regrets = np.array(relative_regrets, dtype=np.float64)
            traditional_regrets = np.array(traditional_regrets, dtype=np.float64)
            
            # Store results
            bootstrap_results['ne_regret'].append(relative_regrets)
            bootstrap_results['traditional_regret'].append(traditional_regrets)
            bootstrap_results['ne_strategy'].append(nash_strategy)
            bootstrap_results['agent_expected_utility'].append(expected_utils)
            bootstrap_results['agent_max_utility'].append(max_utils)
            bootstrap_results['nash_value'].append(nash_value)
        except Exception as e:
            print(f"Error in bootstrap sample {i}: {e}")
            continue
    
    # Ensure consistent types for all results
    bootstrap_results['ne_regret'] = [np.array(r, dtype=np.float64) for r in bootstrap_results['ne_regret']]
    bootstrap_results['traditional_regret'] = [np.array(r, dtype=np.float64) for r in bootstrap_results['traditional_regret']]
    bootstrap_results['ne_strategy'] = [np.array(s, dtype=np.float64) for s in bootstrap_results['ne_strategy']]
    bootstrap_results['agent_expected_utility'] = [np.array(u, dtype=np.float64) for u in bootstrap_results['agent_expected_utility']]
    bootstrap_results['agent_max_utility'] = [np.array(m, dtype=np.float64) for m in bootstrap_results['agent_max_utility']]
    bootstrap_results['nash_value'] = np.array(bootstrap_results['nash_value'], dtype=np.float64)
    
    # Compute percentile-based confidence intervals
    bootstrap_stats = analyze_bootstrap_results(bootstrap_results, all_agents, confidence)
    
    # Add the bootstrap samples to the results
    bootstrap_results['statistics'] = bootstrap_stats
    
    return bootstrap_results

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
    
    try:
        ne_regrets = np.stack(bootstrap_results['ne_regret'])
        traditional_regrets = np.stack(bootstrap_results['traditional_regret'])
        expected_utils = np.stack(bootstrap_results['agent_expected_utility'])
    except ValueError:
        print("Warning: Bootstrap samples have inconsistent shapes. Using a more flexible approach.")
        first_regret = bootstrap_results['ne_regret'][0]
        first_trad_regret = bootstrap_results['traditional_regret'][0]
        first_util = bootstrap_results['agent_expected_utility'][0]
        
        ne_regrets = np.zeros((len(bootstrap_results['ne_regret']), len(first_regret)), dtype=np.float64)
        traditional_regrets = np.zeros((len(bootstrap_results['traditional_regret']), len(first_trad_regret)), dtype=np.float64)
        expected_utils = np.zeros((len(bootstrap_results['agent_expected_utility']), len(first_util)), dtype=np.float64)
        
        for i, (regret, trad_regret, util) in enumerate(zip(bootstrap_results['ne_regret'], 
                                                           bootstrap_results['traditional_regret'],
                                                           bootstrap_results['agent_expected_utility'])):
            if len(regret) == len(first_regret):
                ne_regrets[i] = regret
            else:
                print(f"Warning: Skipping regret sample {i} due to shape mismatch")
                
            if len(trad_regret) == len(first_trad_regret):
                traditional_regrets[i] = trad_regret
            else:
                print(f"Warning: Skipping traditional regret sample {i} due to shape mismatch")
                
            if len(util) == len(first_util):
                expected_utils[i] = util
            else:
                print(f"Warning: Skipping utility sample {i} due to shape mismatch")
    
    # Calculate means
    mean_regrets = np.mean(ne_regrets, axis=0)
    mean_trad_regrets = np.mean(traditional_regrets, axis=0)
    mean_expected_utils = np.mean(expected_utils, axis=0)
    
    # Calculate standard deviations
    std_regrets = np.std(ne_regrets, axis=0)
    std_trad_regrets = np.std(traditional_regrets, axis=0)
    std_expected_utils = np.std(expected_utils, axis=0)
    
    # Calculate percentile-based confidence intervals
    alpha = 1 - confidence
    lower_percentile = alpha / 2 * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    # Calculate percentile-based confidence intervals
    lower_regrets = np.percentile(ne_regrets, lower_percentile, axis=0)
    upper_regrets = np.percentile(ne_regrets, upper_percentile, axis=0)
    
    lower_trad_regrets = np.percentile(traditional_regrets, lower_percentile, axis=0)
    upper_trad_regrets = np.percentile(traditional_regrets, upper_percentile, axis=0)
    
    lower_utils = np.percentile(expected_utils, lower_percentile, axis=0)
    upper_utils = np.percentile(expected_utils, upper_percentile, axis=0)
    
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

def visualize_dual_regret(bootstrap_results, agent_names, figsize=(18, 12)):
    """
    Visualize both traditional regret and relative performance metrics side by side
    Args:
        bootstrap_results: Dictionary with bootstrap samples
        agent_names: List of agent names
        figsize: Figure size tuple
        
    Returns:
        matplotlib figure
    """
    ne_regrets = np.stack(bootstrap_results['ne_regret'])
    traditional_regrets = np.stack(bootstrap_results['traditional_regret'])
    
    mean_ne_regrets = np.mean(ne_regrets, axis=0)
    mean_trad_regrets = np.mean(traditional_regrets, axis=0)
    
    # Create a DataFrame with both metrics
    dual_regret_df = pd.DataFrame({
        'Agent': agent_names,
        'Relative Performance': mean_ne_regrets,  # Higher is better
        'Traditional Regret': mean_trad_regrets   # Lower is better
    })
    
    # Sort agents by relative performance (descending)
    relative_df = dual_regret_df.sort_values('Relative Performance', ascending=False)
    
    # Sort agents by traditional regret (ascending)
    traditional_df = dual_regret_df.sort_values('Traditional Regret')
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=figsize)
    
    # 1. Side-by-side bar charts
    ax1 = plt.subplot2grid((2, 2), (0, 0))
    ax2 = plt.subplot2grid((2, 2), (0, 1))
    
    # Relative performance (higher is better)
    bars1 = ax1.barh(relative_df['Agent'], relative_df['Relative Performance'])
    ax1.set_title('Relative Performance vs Nash', fontsize=14, fontweight='bold')
    ax1.set_xlabel('expected_utility - nash_value (Higher is Better)', fontsize=12)
    ax1.axvline(x=0, color='red', linestyle='--', alpha=0.7)
    ax1.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Color bars based on value (positive=green, negative=red)
    for i, bar in enumerate(bars1):
        value = relative_df['Relative Performance'].iloc[i]
        bar.set_color('green' if value >= 0 else 'red')
        ax1.text(value + np.sign(value) * 0.01, i, f'{value:.2f}', 
                 va='center', fontsize=9,
                 color='darkgreen' if value >= 0 else 'darkred')
    
    # Traditional regret (lower is better)
    bars2 = ax2.barh(traditional_df['Agent'], traditional_df['Traditional Regret'])
    ax2.set_title('Traditional Regret', fontsize=14, fontweight='bold')
    ax2.set_xlabel('max_utility - expected_utility (Lower is Better)', fontsize=12)
    ax2.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Add value labels
    for i, bar in enumerate(bars2):
        value = traditional_df['Traditional Regret'].iloc[i]
        ax2.text(value + 0.01, i, f'{value:.2f}', va='center', fontsize=9)
    
    # 2. Scatterplot showing relationship between metrics
    ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=2)
    
    scatter = ax3.scatter(dual_regret_df['Traditional Regret'], 
                         dual_regret_df['Relative Performance'],
                         alpha=0.7, s=100)
    
    # Add agent labels to scatter points
    for i, agent in enumerate(dual_regret_df['Agent']):
        x = dual_regret_df['Traditional Regret'].iloc[i]
        y = dual_regret_df['Relative Performance'].iloc[i]
        ax3.annotate(agent, (x, y), fontsize=10, 
                    xytext=(5, 5), textcoords='offset points')
    
    ax3.set_xlabel('Traditional Regret', fontsize=12)
    ax3.set_ylabel('NE Regret Performance', fontsize=12)
    ax3.set_title('Relationship Between Regret Metrics', fontsize=14, fontweight='bold')
    ax3.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return fig

def plot_regret_distributions(bootstrap_results, agent_names, figsize=(12, 8), regret_type='ne_regret'):
    """
    Plot distributions of regrets for each agent
    
    Args:
        bootstrap_results: Dictionary with bootstrap samples
        agent_names: List of agent names
        figsize: Figure size
        regret_type: Type of regret to plot ('ne_regret' or 'traditional_regret')
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
    
    regret_name = "Relative Performance" if regret_type == 'ne_regret' else "Traditional Regret"
    
    for i, agent in enumerate(agent_names):
        if i < len(axs) and i < regrets.shape[1]:
            axs[i].hist(regrets[:, i], bins=20, alpha=0.7, 
                       color='green' if regret_type == 'ne_regret' else 'blue')
            axs[i].set_title(agent)
            axs[i].set_xlabel(regret_name)
            axs[i].set_ylabel('Frequency')
            
            # Add mean line
            mean_regret = np.mean(regrets[:, i])
            axs[i].axvline(mean_regret, color='r', linestyle='--', 
                           label=f'Mean: {mean_regret:.4f}')
            
            # Add 95% CI
            lower_ci = np.percentile(regrets[:, i], 2.5)
            upper_ci = np.percentile(regrets[:, i], 97.5)
            axs[i].axvline(lower_ci, color='g', linestyle=':')
            axs[i].axvline(upper_ci, color='g', linestyle=':', 
                           label=f'95% CI: [{lower_ci:.4f}, {upper_ci:.4f}]')
            
            if regret_type == 'ne_regret':
                axs[i].axvline(0, color='k', linestyle='-', alpha=0.5)
            
            axs[i].legend(fontsize='small')
    
    # Hide any unused subplots
    for j in range(min(i+1, len(axs))):
        if j >= regrets.shape[1]:
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
    
    # Initialize raw data dictionary
    raw_data = {}
    for agent1 in agent_names:
        for agent2 in agent_names:
            raw_data[(agent1, agent2)] = []
    
    # Create a performance matrix with the same dimensions
    performance_matrix = pd.DataFrame(index=agent_names, columns=agent_names)
    
    # Fill raw_data dictionary with values from the dataframe
    for _, row in game_df.iterrows():
        agent1 = agent_map.get(row['agent1'], row['agent1'])
        agent2 = agent_map.get(row['agent2'], row['agent2'])
        
        # Skip if either agent is not in our canonical list
        if agent1 not in agent_names or agent2 not in agent_names:
            continue
        
        # Store agent1's value when playing against agent2
        value1 = row.get(value_col)
        if value1 is not None and not pd.isna(value1):
            raw_data[(agent1, agent2)].append(float(value1))
        
        # If there's a corresponding agent2_value column, store that too
        agent2_value_col = value_col.replace('agent1', 'agent2') if 'agent1' in value_col else None
        if agent2_value_col and agent2_value_col in row:
            value2 = row.get(agent2_value_col)
            if value2 is not None and not pd.isna(value2):
                raw_data[(agent2, agent1)].append(float(value2))
    
    # Compute average values for the performance matrix
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
        confidence=confidence,
        data_matrix=raw_data
    )
    
    return bootstrap_results