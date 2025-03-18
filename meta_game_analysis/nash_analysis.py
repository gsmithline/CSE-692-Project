#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import sys
import os
import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nash_equilibrium import (
    bootstrap_performance_metrics,
    analyze_bootstrap_results,
    plot_regret_distributions,
    compute_acceptance_ratio_matrix
)

def run_nash_analysis(performance_matrix, num_bootstrap_samples=100, confidence_level=0.95):
    """
    Run Nash equilibrium analysis on the performance matrix.
    
    Args:
        performance_matrix: Performance matrix DataFrame
        num_bootstrap_samples: Number of bootstrap samples to use
        confidence_level: Confidence level for bootstrap intervals
        
    Returns:
        tuple: (bootstrap_results, bootstrap_stats, acceptance_matrix, ne_strategy_df)
    """
    # Get list of agents from performance matrix
    agents = performance_matrix.index.tolist()
    
    # Run bootstrapping
    print(f"Running bootstrapping with {num_bootstrap_samples} samples...")
    bootstrap_results = bootstrap_performance_metrics(
        performance_matrix, 
        num_bootstrap=num_bootstrap_samples, 
        sample_with_replacement=True
    )

    # Compute confidence intervals
    print(f"\nComputing {confidence_level*100:.0f}% confidence intervals...")
    bootstrap_stats = analyze_bootstrap_results(
        bootstrap_results, 
        agents, 
        confidence=confidence_level
    )
    
    # Calculate acceptance ratio matrix
    acceptance_matrix = None
    
    # Compute Nash equilibrium mixed strategy
    avg_ne_strategy = np.mean([s for s in bootstrap_results['ne_strategy']], axis=0)
    ne_strategy_df = pd.DataFrame({
        'Agent': agents,
        'Nash Probability': avg_ne_strategy
    }).sort_values(by='Nash Probability', ascending=False)
    
    return bootstrap_results, bootstrap_stats, acceptance_matrix, ne_strategy_df

def plot_nash_distributions(bootstrap_results, agents):
    """
    Plot the distributions of Nash equilibrium metrics.
    
    Args:
        bootstrap_results: Bootstrap results dictionary
        agents: List of agent names
        
    Returns:
        tuple: (regret_fig, trad_regret_fig, rel_perf_fig, dual_regret_fig)
    """
    # Plot Nash equilibrium regret distribution
    regret_fig = plot_regret_distributions(bootstrap_results, agents)
    
    # Plot traditional regret distribution
    trad_regret_fig = plot_regret_distributions(
        bootstrap_results, 
        agents,
        regret_type='traditional_regret'
    )
    
    # Plot relative performance distribution
    rel_perf_fig = plot_regret_distributions(
        bootstrap_results, 
        agents,
        regret_type='ne_regret'
    )
    
    # Plot dual regret visualization
    from nash_equilibrium.bootstrap import visualize_dual_regret
    dual_regret_fig = visualize_dual_regret(bootstrap_results, agents)
    
    return regret_fig, trad_regret_fig, rel_perf_fig, dual_regret_fig

def save_nash_plots(figures, save_dir):
    """
    Save Nash equilibrium plot figures to files.
    
    Args:
        figures: Dictionary of figure objects
        save_dir: Directory to save the plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    for name, fig in figures.items():
        filename = f"{name}.png"
        filepath = os.path.join(save_dir, filename)
        fig.savefig(filepath, bbox_inches='tight', dpi=300)
        plt.close(fig)

def print_nash_summary(bootstrap_stats, ne_strategy_df, bootstrap_results):
    """
    Print a summary of Nash equilibrium analysis results.
    
    Args:
        bootstrap_stats: DataFrame with bootstrap statistics
        ne_strategy_df: DataFrame with Nash equilibrium strategy probabilities
        bootstrap_results: Bootstrap results dictionary
    """
    print("\nNash Equilibrium Mixed Strategy (Probability Distribution):")
    print(ne_strategy_df)

    try:
        # Try to compute statistics from bootstrap results
        if isinstance(bootstrap_results['ne_regret'], list) and len(bootstrap_results['ne_regret']) > 0:
            # Check if we're dealing with raw bootstrap or standard bootstrap
            if isinstance(bootstrap_results['ne_regret'][0], np.ndarray):
                ne_regrets = np.array(bootstrap_results['ne_regret'])
                # Handle potential shape issues
                if ne_regrets.ndim == 3:  # This happens with some matrix formats
                    ne_regrets = ne_regrets.reshape(ne_regrets.shape[0], -1)
                mean_regrets = np.mean(ne_regrets, axis=0)
                std_regrets = np.std(ne_regrets, axis=0)
            else:
                # We already have computed statistics in bootstrap_stats
                mean_regrets = bootstrap_stats['Mean NE Regret'].values
                std_regrets = bootstrap_stats['Std NE Regret'].values
            
            print("\nStatistical Summary of Nash Equilibrium Analysis:")
            print(f"Average NE regret across all agents: {np.mean(mean_regrets):.4f}")
            print(f"Maximum average regret: {np.max(mean_regrets):.4f}")
            print(f"Minimum average regret: {np.min(mean_regrets):.4f}")
            print(f"Standard deviation of average regrets: {np.std(mean_regrets):.4f}")
        else:
            print("\nNo valid bootstrap regret data available.")
            return
    except Exception as e:
        print(f"\nError computing bootstrap statistics: {e}")
        print("Using pre-computed statistics from bootstrap_stats instead.")
        mean_regrets = bootstrap_stats['Mean NE Regret'].values
        std_regrets = bootstrap_stats['Std NE Regret'].values

    top_agents = bootstrap_stats.head(5)
    print("\nTop 5 agents by Nash Equilibrium analysis (lowest regret):")
    print(top_agents[['Agent', 'Mean NE Regret', 'Std NE Regret']])
    
    print("\nTop 5 agents by Traditional Regret (lower is better):")
    print(bootstrap_stats.sort_values('Mean Traditional Regret').head(5)[['Agent', 'Mean Traditional Regret', 'Std Traditional Regret']])

    print("\nTop 5 agents by Relative Performance (higher is better):")
    print(bootstrap_stats.sort_values('Mean NE Regret', ascending=False).head(5)[['Agent', 'Mean NE Regret', 'Std NE Regret']])

def calculate_acceptance_ratio(all_results, agents):
    """
    Calculate the acceptance ratio matrix for all agents.
    
    Args:
        all_results: List of all game results
        agents: List of agent names
        
    Returns:
        DataFrame: Acceptance ratio matrix
    """
    acceptance_matrix = compute_acceptance_ratio_matrix(all_results, agents)
    return acceptance_matrix

def run_raw_data_nash_analysis(all_results, num_bootstrap_samples=100, confidence_level=0.95):
    """
    Run Nash equilibrium analysis using non-parametric bootstrapping on raw game data.
    This is the preferred method for direct bootstrapping from individual game outcomes.
    
    Args:
        all_results: List of dictionaries containing raw game results
        num_bootstrap_samples: Number of bootstrap samples to use
        confidence_level: Confidence level for bootstrap intervals
        
    Returns:
        tuple: (bootstrap_results, bootstrap_stats, agent_names)
    """
    from meta_game_analysis.bootstrap_nonparametric import nonparametric_bootstrap_from_raw_data
    
    print(f"Running non-parametric bootstrapping with {num_bootstrap_samples} samples...")
    bootstrap_results, agent_names = nonparametric_bootstrap_from_raw_data(
        all_results,
        num_bootstrap=num_bootstrap_samples,
        confidence=confidence_level
    )
    
    bootstrap_stats = bootstrap_results['statistics']
    
    # Compute Nash equilibrium mixed strategy
    avg_ne_strategy = np.mean([s for s in bootstrap_results['ne_strategy']], axis=0)
    ne_strategy_df = pd.DataFrame({
        'Agent': agent_names,
        'Nash Probability': avg_ne_strategy
    }).sort_values(by='Nash Probability', ascending=False)
    
    print("\nNash Equilibrium Analysis Complete")
    print_nash_summary(bootstrap_stats, ne_strategy_df, bootstrap_results)
    
    return bootstrap_results, bootstrap_stats, ne_strategy_df, agent_names 