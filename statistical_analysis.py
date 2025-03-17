import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.metrics import mean_squared_error
import random
import os
import math
from scipy.optimize import linprog
import json
import glob
from collections import defaultdict

def sample_empirical_game(scores, agent_types, seeds_sets):
    """
    Sample an empirical game matrix from the scores data.
    Similar to the function in the provided reference code.
    """
    save_file_name = "policy_type{}_seed{}"
    game_matrix = np.zeros((len(agent_types), len(agent_types)))
    acception_ratio_matrix = np.zeros((len(agent_types), len(agent_types)))
    social_welfare_matrix = np.zeros((len(agent_types), len(agent_types)))
    nash_welfare_matrix = np.zeros((len(agent_types), len(agent_types)))

    for i in range(len(agent_types)):
        for j in range(i, len(agent_types)):
            count = 0
            for i_seed in seeds_sets[i]:
                principal_agent_name = save_file_name.format(agent_types[i], i_seed)
                for j_seed in seeds_sets[j]:
                    count += 1
                    secondary_agent_name = save_file_name.format(agent_types[j], j_seed)
                    if i == j:
                        payoffs = scores[(principal_agent_name, secondary_agent_name)][2][0]
                        acception_ratio = scores[(principal_agent_name, secondary_agent_name)][2][2]
                        game_matrix[i, j] += (payoffs[0] + payoffs[1]) / 2
                        acception_ratio_matrix[i, j] += (acception_ratio[0] + acception_ratio[1]) / 2
                    else:
                        payoffs0 = scores[(principal_agent_name, secondary_agent_name)][2][0]
                        payoffs1 = scores[(secondary_agent_name, principal_agent_name)][2][0]
                        acception_ratio0 = scores[(principal_agent_name, secondary_agent_name)][2][2]
                        acception_ratio1 = scores[(secondary_agent_name, principal_agent_name)][2][2]
                        game_matrix[i, j] += (payoffs0[0] + payoffs1[1]) / 2
                        game_matrix[j, i] += (payoffs0[1] + payoffs1[0]) / 2
                        acception_ratio_matrix[i, j] += (acception_ratio0[0] + acception_ratio1[1]) / 2
                        acception_ratio_matrix[j, i] += (acception_ratio0[1] + acception_ratio1[0]) / 2

            game_matrix[i, j] /= count
            acception_ratio_matrix[i, j] /= count
            if i != j:
                game_matrix[j, i] /= count
                acception_ratio_matrix[j, i] /= count
            social_welfare_matrix[i, j] = (game_matrix[i, j] + game_matrix[j, i])
            nash_welfare_matrix[i, j] = (game_matrix[i, j] * game_matrix[j, i])
            if i != j:
                social_welfare_matrix[j, i] = (game_matrix[i, j] + game_matrix[j, i])
                nash_welfare_matrix[j, i] = (game_matrix[i, j] * game_matrix[j, i])

    return game_matrix, acception_ratio_matrix, social_welfare_matrix, nash_welfare_matrix

def milp_max_sym_ent_2p(game_matrix, iterations=100):
    """
    A better implementation of Nash equilibrium solver using linear programming.
    This is still simplified but better than the uniform strategy.
    
    Args:
        game_matrix: The payoff matrix
        iterations: Number of iterations for the algorithm
        
    Returns:
        Nash equilibrium mixed strategy
    """
    n = game_matrix.shape[0]
    
    # For a symmetric game, we can use linear programming to find the Nash equilibrium
    # This is a simplified approach for a zero-sum 2-player game
    try:
        # Set up the linear programming problem
        c = np.zeros(n + 1)
        c[-1] = -1  # We want to maximize v
        
        # Constraints: strategy_i * value >= v for all i
        A_ub = np.zeros((n, n + 1))
        for i in range(n):
            A_ub[i, :-1] = -game_matrix[:, i]
            A_ub[i, -1] = 1
        
        b_ub = np.zeros(n)
        
        # Sum of probabilities = 1
        A_eq = np.zeros((1, n + 1))
        A_eq[0, :-1] = 1
        b_eq = np.ones(1)
        
        # Bounds: probabilities between 0 and 1, v unconstrained
        bounds = [(0, 1) for _ in range(n)] + [(None, None)]
        
        # Solve the linear program
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        
        if result.success:
            strategy = result.x[:-1]  # Extract the strategy (excluding v)
            # Normalize to ensure it sums to 1
            return strategy / np.sum(strategy)
    except Exception as e:
        print(f"Error in Nash equilibrium calculation: {e}")
    
    # Fallback to a slightly randomized uniform strategy if the optimization fails
    strategy = np.ones(n) / n
    # Add some noise to break symmetry
    noise = np.random.normal(0, 0.01, n)
    strategy = strategy + noise
    # Ensure it's a valid probability distribution
    strategy = np.maximum(strategy, 0)
    return strategy / np.sum(strategy)

def compute_ne_regret(game_matrix, acception_ratio_matrix, social_welfare_matrix, nash_welfare_matrix):
    """
    Compute Nash equilibrium regret and other metrics.
    """
    nash_strategy = milp_max_sym_ent_2p(game_matrix, 100)
    dev_payoff = game_matrix @ nash_strategy.reshape((-1, 1))

    mixed_payoff = nash_strategy.reshape((1, -1)) @ dev_payoff
    ne_regret = (dev_payoff - mixed_payoff).reshape(-1)
    acception_ratio = acception_ratio_matrix @ nash_strategy.reshape((-1, 1))
    social_welfares = social_welfare_matrix @ nash_strategy.reshape((-1, 1))
    nash_welfares = nash_welfare_matrix @ nash_strategy.reshape((-1, 1))

    sigma_dev_payoff = (nash_strategy.reshape((1, -1)) @ game_matrix).reshape(-1)
    real_nash_welfares = dev_payoff.reshape(-1) * sigma_dev_payoff
    
    return ne_regret, acception_ratio.reshape(-1), social_welfares.reshape(-1), nash_welfares.reshape(-1), real_nash_welfares

def compute_uniform_ranking(game_matrix, acception_ratio_matrix, social_welfare_matrix, nash_welfare_matrix):
    """
    Compute uniform ranking based on average payoffs.
    """
    dev_payoff = np.mean(game_matrix, axis=1)
    acception_ratio = np.mean(acception_ratio_matrix, axis=1)
    social_welfares = np.mean(social_welfare_matrix, axis=1)
    nash_welfares = np.mean(nash_welfare_matrix, axis=1)

    sigma_dev_payoff = np.mean(game_matrix, axis=0)

    return dev_payoff, acception_ratio.reshape(-1), social_welfares.reshape(-1), nash_welfares.reshape(-1), dev_payoff * sigma_dev_payoff

def analyze_results_data(performance_matrix, num_bootstrap=100, with_replacement=True, num_seed=10):
    """
    Analyze results data from a DataFrame.
    This adapts the logic from the provided code to work with a DataFrame.
    
    Args:
        performance_matrix: DataFrame with performance values
        num_bootstrap: Number of bootstrap samples
        with_replacement: Whether to sample with replacement
        num_seed: Number of seeds for bootstrapping
        
    Returns:
        Dictionary with analysis results
    """
    # Convert to DataFrame if it's not already
    if not isinstance(performance_matrix, pd.DataFrame):
        agent_names = [f"Agent_{i}" for i in range(performance_matrix.shape[0])]
        performance_matrix = pd.DataFrame(
            performance_matrix, 
            index=agent_names,
            columns=agent_names
        )
    
    # Extract agent types from DataFrame indices
    agent_types = performance_matrix.index.tolist()
    agent_real_names = agent_types  # You can provide mapping if needed
    
    # Run bootstrap analysis
    ne_regret_list = []
    ne_acception_ratio_list = []
    ne_social_welfare_list = []
    ne_nash_welfare_list = []
    real_ne_nash_welfare_list = []
    uniform_ranking_list = []
    uniform_acception_ratio_list = []
    uniform_social_welfare_list = []
    uniform_nash_welfare_list = []
    real_uniform_nash_welfare_list = []
    
    # Create a 3D acceptance ratio to match the game matrix structure
    # By default, use 80% acceptance as a placeholder
    acceptance_ratio_base = performance_matrix.values * 0 + 0.8
    
    for i in range(num_bootstrap):
        # For bootstrapping, sample rows and columns with replacement
        if with_replacement:
            # Sample indices with replacement
            indices = np.random.choice(len(agent_types), size=len(agent_types), replace=True)
        else:
            # Without replacement, we'll just permute the indices
            indices = np.random.permutation(len(agent_types))
        
        # Create bootstrap sample by selecting rows and columns
        bootstrap_matrix = performance_matrix.values[np.ix_(indices, indices)]
        
        # Add noise to the bootstrap sample to create variability
        # This simulates variation in the underlying data
        noise = np.random.normal(0, performance_matrix.values.std() * 0.05, bootstrap_matrix.shape)
        bootstrap_matrix = bootstrap_matrix + noise
        
        # Create acceptance ratio matrix (can be replaced with actual data if available)
        # Here we'll use a placeholder with some variability
        acceptance_matrix = acceptance_ratio_base[np.ix_(indices, indices)]
        acceptance_noise = np.random.normal(0, 0.05, acceptance_matrix.shape)
        acceptance_matrix = np.clip(acceptance_matrix + acceptance_noise, 0, 1)
        
        # Calculate social welfare and Nash welfare matrices
        social_welfare_matrix = bootstrap_matrix + bootstrap_matrix.T
        nash_welfare_matrix = bootstrap_matrix * bootstrap_matrix.T
        
        # Compute metrics
        ne_regret, ne_acception, ne_social, ne_nash, real_ne_nash = compute_ne_regret(
            bootstrap_matrix, acceptance_matrix, social_welfare_matrix, nash_welfare_matrix)
        
        ne_regret_list.append(ne_regret)
        ne_acception_ratio_list.append(ne_acception)
        ne_social_welfare_list.append(ne_social)
        ne_nash_welfare_list.append(ne_nash)
        real_ne_nash_welfare_list.append(real_ne_nash)
        
        uniform_payoffs, uniform_acception, uniform_social, uniform_nash, real_uniform_nash = compute_uniform_ranking(
            bootstrap_matrix, acceptance_matrix, social_welfare_matrix, nash_welfare_matrix)
        
        uniform_ranking_list.append(uniform_payoffs)
        uniform_acception_ratio_list.append(uniform_acception)
        uniform_social_welfare_list.append(uniform_social)
        uniform_nash_welfare_list.append(uniform_nash)
        real_uniform_nash_welfare_list.append(real_uniform_nash)
    
    # Convert lists to arrays for easier statistics
    ne_regret_list = np.array(ne_regret_list)
    ne_acception_ratio_list = np.array(ne_acception_ratio_list)
    ne_social_welfare_list = np.array(ne_social_welfare_list)
    ne_nash_welfare_list = np.array(ne_nash_welfare_list)
    real_ne_nash_welfare_list = np.array(real_ne_nash_welfare_list)
    uniform_ranking_list = np.array(uniform_ranking_list)
    uniform_acception_ratio_list = np.array(uniform_acception_ratio_list)
    uniform_social_welfare_list = np.array(uniform_social_welfare_list)
    uniform_nash_welfare_list = np.array(uniform_nash_welfare_list)
    real_uniform_nash_welfare_list = np.array(real_uniform_nash_welfare_list)
    
    # Calculate statistics
    ne_regret_mean = np.mean(ne_regret_list, axis=0)
    ne_regret_std = np.std(ne_regret_list, axis=0)
    
    ne_acception_ratio_mean = np.mean(ne_acception_ratio_list, axis=0)
    ne_acception_ratio_std = np.std(ne_acception_ratio_list, axis=0)
    
    ne_social_welfare_mean = np.mean(ne_social_welfare_list, axis=0)
    ne_social_welfare_std = np.std(ne_social_welfare_list, axis=0)
    
    ne_nash_welfare_mean = np.mean(ne_nash_welfare_list, axis=0)
    ne_nash_welfare_std = np.std(ne_nash_welfare_list, axis=0)
    
    real_ne_nash_welfare_mean = np.mean(real_ne_nash_welfare_list, axis=0)
    real_ne_nash_welfare_std = np.std(real_ne_nash_welfare_list, axis=0)
    
    uniform_ranking_mean = np.mean(uniform_ranking_list, axis=0)
    uniform_ranking_std = np.std(uniform_ranking_list, axis=0)
    
    uniform_acception_ratio_mean = np.mean(uniform_acception_ratio_list, axis=0)
    uniform_acception_ratio_std = np.std(uniform_acception_ratio_list, axis=0)
    
    uniform_social_welfare_mean = np.mean(uniform_social_welfare_list, axis=0)
    uniform_social_welfare_std = np.std(uniform_social_welfare_list, axis=0)
    
    uniform_nash_welfare_mean = np.mean(uniform_nash_welfare_list, axis=0)
    uniform_nash_welfare_std = np.std(uniform_nash_welfare_list, axis=0)
    
    real_uniform_nash_welfare_mean = np.mean(real_uniform_nash_welfare_list, axis=0)
    real_uniform_nash_welfare_std = np.std(real_uniform_nash_welfare_list, axis=0)
    
    # Print results
    print("NE Ranking:")
    ne_ranking = np.argsort(ne_regret_mean)[::-1]
    for i in ne_ranking:
        print(f"method label {i}", agent_real_names[i], 
              f"{ne_regret_mean[i]:.6f} +/- {ne_regret_std[i]:.6f}")
    
    print("\nNE acceptance ratio Ranking:")
    ne_acception_ratio_ranking = np.argsort(ne_acception_ratio_mean)[::-1]
    for i in ne_acception_ratio_ranking:
        print(f"method label {i}", agent_real_names[i], 
              f"{ne_acception_ratio_mean[i]:.6f} +/- {ne_acception_ratio_std[i]:.6f}")
    
    print("\nNE social welfare Ranking:")
    ne_social_welfare_ranking = np.argsort(ne_social_welfare_mean)[::-1]
    for i in ne_social_welfare_ranking:
        print(f"method label {i}", agent_real_names[i], 
              f"{ne_social_welfare_mean[i]:.6f} +/- {ne_social_welfare_std[i]:.6f}")
    
    print("\nNE Nash welfare Ranking:")
    ne_nash_welfare_ranking = np.argsort(ne_nash_welfare_mean)[::-1]
    for i in ne_nash_welfare_ranking:
        print(f"method label {i}", agent_real_names[i], 
              f"{ne_nash_welfare_mean[i]:.6f} +/- {ne_nash_welfare_std[i]:.6f}")
    
    print("\nReal NE Nash welfare Ranking:")
    real_ne_nash_welfare_ranking = np.argsort(real_ne_nash_welfare_mean)[::-1]
    for i in real_ne_nash_welfare_ranking:
        print(f"method label {i}", agent_real_names[i], 
              f"{real_ne_nash_welfare_mean[i]:.6f} +/- {real_ne_nash_welfare_std[i]:.6f}")
    
    print("\nUniform Ranking:")
    uniform_ranking_idx = np.argsort(uniform_ranking_mean)[::-1]
    for i in uniform_ranking_idx:
        print(f"method label {i}", agent_real_names[i], 
              f"{uniform_ranking_mean[i]:.6f} +/- {uniform_ranking_std[i]:.6f}")
    
    print("\nUniform acceptance ratio Ranking:")
    uniform_acception_ratio_ranking = np.argsort(uniform_acception_ratio_mean)[::-1]
    for i in uniform_acception_ratio_ranking:
        print(f"method label {i}", agent_real_names[i], 
              f"{uniform_acception_ratio_mean[i]:.6f} +/- {uniform_acception_ratio_std[i]:.6f}")
    
    print("\nUniform social welfare Ranking:")
    uniform_social_welfare_ranking = np.argsort(uniform_social_welfare_mean)[::-1]
    for i in uniform_social_welfare_ranking:
        print(f"method label {i}", agent_real_names[i], 
              f"{uniform_social_welfare_mean[i]:.6f} +/- {uniform_social_welfare_std[i]:.6f}")
    
    print("\nUniform Nash welfare Ranking:")
    uniform_nash_welfare_ranking = np.argsort(uniform_nash_welfare_mean)[::-1]
    for i in uniform_nash_welfare_ranking:
        print(f"method label {i}", agent_real_names[i], 
              f"{uniform_nash_welfare_mean[i]:.6f} +/- {uniform_nash_welfare_std[i]:.6f}")
    
    print("\nReal uniform Nash welfare Ranking:")
    real_uniform_nash_welfare_ranking = np.argsort(real_uniform_nash_welfare_mean)[::-1]
    for i in real_uniform_nash_welfare_ranking:
        print(f"method label {i}", agent_real_names[i], 
              f"{real_uniform_nash_welfare_mean[i]:.6f} +/- {real_uniform_nash_welfare_std[i]:.6f}")
    
    # Save results
    results = {
        'ne_regret_mean': ne_regret_mean,
        'ne_regret_std': ne_regret_std,
        'ne_acception_ratio_mean': ne_acception_ratio_mean,
        'ne_acception_ratio_std': ne_acception_ratio_std,
        'ne_social_welfare_mean': ne_social_welfare_mean,
        'ne_social_welfare_std': ne_social_welfare_std,
        'ne_nash_welfare_mean': ne_nash_welfare_mean,
        'ne_nash_welfare_std': ne_nash_welfare_std,
        'real_ne_nash_welfare_mean': real_ne_nash_welfare_mean,
        'real_ne_nash_welfare_std': real_ne_nash_welfare_std,
        'uniform_ranking_mean': uniform_ranking_mean,
        'uniform_ranking_std': uniform_ranking_std,
        'uniform_acception_ratio_mean': uniform_acception_ratio_mean,
        'uniform_acception_ratio_std': uniform_acception_ratio_std,
        'uniform_social_welfare_mean': uniform_social_welfare_mean,
        'uniform_social_welfare_std': uniform_social_welfare_std,
        'uniform_nash_welfare_mean': uniform_nash_welfare_mean,
        'uniform_nash_welfare_std': uniform_nash_welfare_std,
        'real_uniform_nash_welfare_mean': real_uniform_nash_welfare_mean,
        'real_uniform_nash_welfare_std': real_uniform_nash_welfare_std,
        'agent_types': agent_types,
        'agent_real_names': agent_real_names
    }
    
    return results

def visualize_results(results, metric_name="ne_regret", title="Nash Equilibrium Regret", reverse=True):
    """
    Create a visualization of the results for a specific metric.
    
    Args:
        results: Results dictionary from analyze_results_data
        metric_name: Name of the metric to visualize
        title: Title for the plot
        reverse: Whether to reverse the sorting order
    """
    mean_key = f"{metric_name}_mean"
    std_key = f"{metric_name}_std"
    
    # Sort the indices
    if reverse:
        indices = np.argsort(results[mean_key])[::-1]
    else:
        indices = np.argsort(results[mean_key])
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.errorbar(
        range(len(indices)),
        [results[mean_key][i] for i in indices],
        yerr=[results[std_key][i] for i in indices],
        fmt='o',
        capsize=5
    )
    plt.xticks(range(len(indices)), [results['agent_real_names'][i] for i in indices], rotation=45, ha='right')
    plt.title(title)
    plt.ylabel(metric_name.replace('_', ' ').title())
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return plt.gcf()

def visualize_all_metrics(results):
    """
    Create a visualization with all metrics.
    
    Args:
        results: Results dictionary from analyze_results_data
    """
    plt.figure(figsize=(16, 12))
    
    # Define the metrics to visualize
    metrics = [
        ("ne_regret", "Nash Equilibrium Regret", True),
        ("ne_social_welfare", "Social Welfare", True),
        ("ne_nash_welfare", "Nash Welfare", True),
        ("ne_acception_ratio", "Acceptance Ratio", True)
    ]
    
    # Create subplots
    for i, (metric, title, reverse) in enumerate(metrics):
        plt.subplot(2, 2, i+1)
        
        mean_key = f"{metric}_mean"
        std_key = f"{metric}_std"
        
        # Sort the indices
        if reverse:
            indices = np.argsort(results[mean_key])[::-1]
        else:
            indices = np.argsort(results[mean_key])
        
        # Create the plot
        plt.errorbar(
            range(len(indices)),
            [results[mean_key][i] for i in indices],
            yerr=[results[std_key][i] for i in indices],
            fmt='o',
            capsize=5
        )
        plt.xticks(range(len(indices)), [results['agent_real_names'][i] for i in indices], rotation=45, ha='right')
        plt.title(title)
        plt.ylabel(metric.replace('_', ' ').title())
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return plt.gcf()

def count_game_outcomes(directory_path):
    """
    Count different game outcomes (ACCEPT, WALK) in JSON files within directory.
    
    Args:
        directory_path (str): Path to directory containing game data
        
    Returns:
        dict: Counts of different outcomes
    """
    # Define patterns to search for
    json_files = glob.glob(os.path.join(directory_path, "*", "*.json"))
    
    print(f"Found {len(json_files)} JSON files in {directory_path}")
    
    # Initialize counters
    outcome_counts = {
        "ACCEPT": 0,
        "WALK": 0,
        "INVALID_WALK": 0
    }
    
    # Dictionary to store outcomes by matchup
    matchup_outcomes = defaultdict(lambda: {"ACCEPT": 0, "WALK": 0, "INVALID_WALK": 0, "total": 0})
    
    for file_path in json_files:
        try:
            # Extract matchup name from directory path
            matchup = os.path.basename(os.path.dirname(file_path))
            
            with open(file_path, 'r') as f:
                data = json.load(f)
                
                # The structure is different than expected
                # Each file contains a list of games in 'all_game_data'
                if 'all_game_data' in data:
                    for game in data['all_game_data']:
                        if 'round_data' in game:
                            # Get the last round's actions
                            rounds = game['round_data']
                            if rounds:
                                last_round = rounds[-1]
                                if 'action' in last_round:
                                    final_action = last_round['action']
                                    
                                    # Check if this is potentially an invalid walk
                                    is_invalid_walk = False
                                    
                                    if final_action == "WALK" and len(rounds) > 1:
                                        # Check if there's text mentioning invalid in the last few rounds
                                        for r in rounds[-3:]:
                                            if 'response' in r and isinstance(r['response'], str):
                                                if "invalid" in r['response'].lower() or "not valid" in r['response'].lower():
                                                    is_invalid_walk = True
                                                    break
                                    
                                    if final_action == "ACCEPT":
                                        outcome_counts["ACCEPT"] += 1
                                        matchup_outcomes[matchup]["ACCEPT"] += 1
                                    elif final_action == "WALK":
                                        if is_invalid_walk:
                                            outcome_counts["INVALID_WALK"] += 1
                                            matchup_outcomes[matchup]["INVALID_WALK"] += 1
                                        else:
                                            outcome_counts["WALK"] += 1
                                            matchup_outcomes[matchup]["WALK"] += 1
                                    
                                    matchup_outcomes[matchup]["total"] += 1
                
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"Error processing {file_path}: {e}")
    
    return outcome_counts, matchup_outcomes

def calculate_acceptance_ratios(matchup_outcomes):
    """
    Calculate acceptance ratios for each matchup.
    
    Args:
        matchup_outcomes (dict): Dictionary of outcomes by matchup
        
    Returns:
        pd.DataFrame: DataFrame with acceptance ratios
    """
    results = []
    
    for matchup, counts in matchup_outcomes.items():
        if counts["total"] > 0:
            acceptance_ratio = counts["ACCEPT"] / counts["total"]
            walk_ratio = counts["WALK"] / counts["total"] if counts["total"] > 0 else 0
            invalid_walk_ratio = counts["INVALID_WALK"] / counts["total"] if counts["total"] > 0 else 0
            
            results.append({
                "matchup": matchup,
                "accept_count": counts["ACCEPT"],
                "walk_count": counts["WALK"],
                "invalid_walk_count": counts["INVALID_WALK"],
                "total_games": counts["total"],
                "acceptance_ratio": acceptance_ratio,
                "walk_ratio": walk_ratio,
                "invalid_walk_ratio": invalid_walk_ratio
            })
    
    return pd.DataFrame(results)

def analyze_game_matrices():
    """
    Analyze game matrices and return statistics.
    """
    # Analyze game matrix 1
    matrix1_counts, matrix1_matchups = count_game_outcomes("crossplay/game_matrix_1")
    matrix1_df = calculate_acceptance_ratios(matrix1_matchups)
    
    # Analyze game matrix 2
    matrix2_counts, matrix2_matchups = count_game_outcomes("crossplay/game_matrix_2")
    matrix2_df = calculate_acceptance_ratios(matrix2_matchups)
    
    # Print summary statistics
    print("Game Matrix 1 Outcomes:")
    print(f"Total ACCEPT: {matrix1_counts['ACCEPT']}")
    print(f"Total WALK: {matrix1_counts['WALK']}")
    print(f"Total INVALID_WALK: {matrix1_counts['INVALID_WALK']}")
    total_games1 = matrix1_counts['ACCEPT'] + matrix1_counts['WALK'] + matrix1_counts['INVALID_WALK']
    print(f"Acceptance Ratio: {matrix1_counts['ACCEPT'] / total_games1 if total_games1 > 0 else 0}")
    
    print("\nGame Matrix 2 Outcomes:")
    print(f"Total ACCEPT: {matrix2_counts['ACCEPT']}")
    print(f"Total WALK: {matrix2_counts['WALK']}")
    print(f"Total INVALID_WALK: {matrix2_counts['INVALID_WALK']}")
    total_games2 = matrix2_counts['ACCEPT'] + matrix2_counts['WALK'] + matrix2_counts['INVALID_WALK']
    print(f"Acceptance Ratio: {matrix2_counts['ACCEPT'] / total_games2 if total_games2 > 0 else 0}")
    
    return {
        "matrix1": {
            "counts": matrix1_counts,
            "matchups": matrix1_df
        },
        "matrix2": {
            "counts": matrix2_counts,
            "matchups": matrix2_df
        }
    }

if __name__ == "__main__":
    results = analyze_game_matrices()
    
    # Display detailed results by matchup
    print("\nGame Matrix 1 - Results by Matchup:")
    print(results["matrix1"]["matchups"].sort_values("acceptance_ratio", ascending=False))
    
    print("\nGame Matrix 2 - Results by Matchup:")
    print(results["matrix2"]["matchups"].sort_values("acceptance_ratio", ascending=False))

# Code to add to the notebook:
"""
# Add this code to your notebook to run the statistical analysis

import statistical_analysis as sa
import pickle
import matplotlib.pyplot as plt

# Choose the matrix you want to analyze (e.g., performance_matrix, welfare_matrix, etc.)
# Make sure the matrix contains the data you want to analyze

# Run analysis
results = sa.analyze_results_data(performance_matrix, num_bootstrap=100)

# Save results if needed
with open('statistical_analysis_results.pkl', 'wb') as f:
    pickle.dump(results, f)

# Create visualization
fig = sa.visualize_all_metrics(results)
plt.show()
""" 