#!/usr/bin/env python
# coding: utf-8

import os
import sys
import argparse
import pandas as pd
import numpy as np

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from meta_game_analysis.data_processing import (
    process_all_games,
    get_display_name,
    compute_global_max_values
)
from meta_game_analysis.matrix_creation import (
    create_performance_matrices,
    create_welfare_matrices,
    clean_matrix_names,
    filter_matrices
)
from meta_game_analysis.visualization import (
    create_best_response_graph,
    create_complete_best_response_graph,
    visualize_welfare_matrices,
    visualize_nash_equilibrium,
    visualize_rd_regret_heatmaps,
    visualize_nash_comparison,
    save_results_to_csv
)
from meta_game_analysis.nash_analysis import (
    run_nash_analysis,
    run_raw_data_nash_analysis,
    plot_nash_distributions,
    save_nash_plots,
    print_nash_summary,
    calculate_acceptance_ratio,
    print_pure_nash_info,
    find_nash_with_replicator_dynamics,
    generate_all_nash_stats,
    print_rd_nash_summary,
    print_nash_comparison
)

def run_analysis(input_dir="crossplay/game_matrix_2", output_dir="meta_game_analysis/results", 
                 num_bootstrap=100, confidence=0.95, global_samples=1000, 
                 use_raw_bootstrap=False, discount_factor=0.98):
    """
    Run the full meta-game analysis pipeline.
    
    Args:
        input_dir: Directory containing game data files
        output_dir: Directory to save results
        num_bootstrap: Number of bootstrap samples for Nash analysis
        confidence: Confidence level for bootstrap intervals
        global_samples: Number of samples for computing global max values
        use_raw_bootstrap: Whether to use raw game data for non-parametric bootstrapping
        discount_factor: Discount factor for utility calculations (gamma)
    """
    print(f"Starting meta-game analysis on {input_dir}")
    print(f"Results will be saved to {output_dir}")
    print(f"Using discount factor (gamma): {discount_factor}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Process all games
    print("\nStep 1: Processing game data...")
    all_results, agent_performance, agent_final_rounds, agent_game_counts, agent_final_rounds_self_play = process_all_games(
        input_dir,
        discount_factor=discount_factor
    )
    
    # Step 2: Compute global maximum values
    print("\nStep 2: Computing global maximum values...")
    global_max_nash_welfare, global_standard_max = compute_global_max_values(num_samples=10000)
    print(f"Global max Nash welfare: {global_max_nash_welfare:.2f}")
    print(f"Global max social welfare: {global_standard_max:.2f}")
    
    # Step 3: Create performance matrices
    print("\nStep 3: Creating performance matrices...")
    performance_matrices = create_performance_matrices(all_results, agent_performance, agent_final_rounds)
    
    # Get list of all agents
    all_agents = sorted(list(performance_matrices['overall_agent_performance'].keys()))
    
    # Step 4: Create welfare matrices
    print("\nStep 4: Creating welfare matrices...")
    welfare_matrices = create_welfare_matrices(all_results, all_agents, global_max_nash_welfare)
    
    # Step 5: Clean matrix names
    print("\nStep 5: Cleaning matrix names...")
    cleaned_matrices = {}
    
    # Clean performance matrices
    performance_matrix_names = ['performance_matrix', 'std_dev_matrix', 'variance_matrix', 'scaled_performance_matrix', 'count_matrix']
    for name in performance_matrix_names:
        if name in performance_matrices:
            cleaned_matrices[name] = clean_matrix_names(performance_matrices[name], get_display_name)
    
    # Clean welfare matrices
    for name, matrix in welfare_matrices.items():
        cleaned_matrices[name] = clean_matrix_names(matrix, get_display_name)
    
    # Step 6: Filter matrices to exclude specific agents if needed
    print("\nStep 6: Filtering matrices...")
    exclude_agents = ['anthropic_3.7_sonnet_circle_4']  # Add any agents to exclude
    filtered_matrices = filter_matrices(cleaned_matrices, exclude_agents)
    
    # Step 7: Run Nash equilibrium analysis
    print("\nStep 7: Running Nash equilibrium analysis...")
    performance_matrix = filtered_matrices['performance_matrix']
    
    # Check for pure Nash equilibria
    print("\nChecking for pure Nash equilibria in the performance matrix:")
    print_pure_nash_info(performance_matrix)
    
    # Calculate Nash equilibrium using replicator dynamics
    print("\nFinding Nash equilibrium using replicator dynamics with multiple restarts:")
    rd_nash_df = find_nash_with_replicator_dynamics(
        performance_matrix, 
        num_restarts=10,
        num_iterations=2000,
        verbose=True
    )
    print("\nNash Equilibrium from Replicator Dynamics:")
    print(rd_nash_df)
    
    if use_raw_bootstrap:
        # Use non-parametric bootstrapping with raw game data
        print("\nUsing non-parametric bootstrapping with raw game data...")
        bootstrap_results, bootstrap_stats, ne_strategy_df, agent_names = run_raw_data_nash_analysis(
            all_results,
            num_bootstrap_samples=num_bootstrap,
            confidence_level=confidence
        )
    else:
        # Use traditional bootstrapping with the performance matrix
        print("\nUsing traditional bootstrapping with performance matrix...")
        bootstrap_results, bootstrap_stats, acceptance_matrix, ne_strategy_df = run_nash_analysis(
            performance_matrix,
            num_bootstrap_samples=num_bootstrap,
            confidence_level=confidence
        )
    
    print("\nMax Entropy Nash Equilibrium from bootstrapping:")
    print(ne_strategy_df)
    
    # Generate comprehensive Nash equilibrium statistics
    print("\nGenerating comprehensive Nash equilibrium statistics...")
    comparison_df, rd_regret_df, rd_nash_value = generate_all_nash_stats(
        performance_matrix, 
        bootstrap_stats, 
        ne_strategy_df, 
        rd_nash_df
    )
    
    # Print statistics for both Nash concepts
    print_rd_nash_summary(rd_regret_df, rd_nash_df, rd_nash_value)
    print_nash_comparison(comparison_df)
    
    # Calculate acceptance ratio if needed
    if 'acceptance_matrix' not in filtered_matrices and len(all_results) > 0:
        agents = performance_matrix.index.tolist()
        acceptance_matrix = calculate_acceptance_ratio(all_results, agents)
        filtered_matrices['acceptance_matrix'] = clean_matrix_names(acceptance_matrix, get_display_name)
    
    # Step 8: Create visualizations
    print("\nStep 8: Creating visualizations...")
    # Visualize welfare matrices
    welfare_figures = visualize_welfare_matrices(filtered_matrices, os.path.join(output_dir, 'heatmaps'))
    
    # Create best response graphs
    print("Creating best response graphs...")
    best_response_graph = create_best_response_graph(
        performance_matrix, 
        filename='best_response_graph', 
        save_dir=os.path.join(output_dir, 'graphs')
    )
    
    complete_best_response_graph = create_complete_best_response_graph(
        performance_matrix, 
        filename='complete_best_response', 
        save_dir=os.path.join(output_dir, 'graphs')
    )
    
    # Visualize Nash equilibrium results for Max Entropy Nash
    print("Creating Max Entropy Nash equilibrium visualizations...")
    max_entropy_nash_dir = os.path.join(output_dir, 'max_entropy_nash')
    os.makedirs(max_entropy_nash_dir, exist_ok=True)
    
    nash_figures = visualize_nash_equilibrium(
        bootstrap_stats, 
        ne_strategy_df, 
        save_dir=max_entropy_nash_dir
    )
    
    # Create Nash regret visualizations for RD Nash
    print("Creating Replicator Dynamics Nash equilibrium visualizations...")
    rd_nash_dir = os.path.join(output_dir, 'rd_nash')
    os.makedirs(rd_nash_dir, exist_ok=True)
    
    rd_regret_figures = visualize_rd_regret_heatmaps(
        performance_matrix,
        rd_regret_df,
        save_dir=rd_nash_dir
    )
    
    # Create Nash comparison visualizations
    print("Creating Nash comparison visualizations...")
    comparison_dir = os.path.join(output_dir, 'nash_comparison')
    os.makedirs(comparison_dir, exist_ok=True)
    
    comparison_figures = visualize_nash_comparison(
        comparison_df,
        save_dir=comparison_dir
    )
    
    # Plot Nash distributions
    print("Creating Nash distribution plots...")
    agents_list = performance_matrix.index.tolist()
    regret_fig, trad_regret_fig, rel_perf_fig, dual_regret_fig = plot_nash_distributions(
        bootstrap_results, 
        agents_list
    )
    
    # Save Nash plots
    nash_plot_figures = {
        'regret_distribution': regret_fig,
        'traditional_regret_distribution': trad_regret_fig,
        'relative_performance_distribution': rel_perf_fig,
        'dual_regret': dual_regret_fig
    }
    save_nash_plots(nash_plot_figures, max_entropy_nash_dir)
    
    # Step 9: Save results to CSV files
    print("\nStep 9: Saving results to CSV files...")
    # Save regular matrices
    save_results_to_csv(
        filtered_matrices, 
        bootstrap_stats, 
        ne_strategy_df, 
        os.path.join(output_dir, 'csv')
    )
    
    # Save RD Nash results and comparison
    csv_dir = os.path.join(output_dir, 'csv')
    os.makedirs(csv_dir, exist_ok=True)
    
    rd_regret_df.to_csv(os.path.join(csv_dir, 'rd_nash_regret.csv'), index=False)
    comparison_df.to_csv(os.path.join(csv_dir, 'nash_comparison.csv'), index=False)
    
    # Step 10: Print summary statistics
    print("\nStep 10: Summary of results:")
    print(f"Total games analyzed: {len(all_results)}")
    print(f"Unique agent types: {len(performance_matrices['overall_agent_performance'])}")
    
    print("\nAverage Final Rounds by Agent:")
    for agent, avg_round in sorted(performance_matrices['average_final_rounds'].items()):
        print(f"{get_display_name(agent)}: {avg_round:.2f}")
    
    # Print Nash equilibrium summary
    print_nash_summary(bootstrap_stats, ne_strategy_df, bootstrap_results)
    
    print("\nAnalysis complete. Results saved to:", output_dir)
    
    return {
        'all_results': all_results,
        'performance_matrices': performance_matrices,
        'welfare_matrices': welfare_matrices,
        'filtered_matrices': filtered_matrices,
        'bootstrap_results': bootstrap_results,
        'bootstrap_stats': bootstrap_stats,
        'ne_strategy_df': ne_strategy_df
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run meta-game analysis on negotiation data.")
    parser.add_argument("--input", default="crossplay/game_matrix_2", help="Input directory containing game data")
    parser.add_argument("--output", default="meta_game_analysis/results", help="Output directory for results")
    parser.add_argument("--bootstrap", type=int, default=100, help="Number of bootstrap samples")
    parser.add_argument("--confidence", type=float, default=0.95, help="Confidence level for intervals")
    parser.add_argument("--global-samples", type=int, default=1000, help="Number of samples for global max values")
    parser.add_argument("--raw-bootstrap", action="store_true", help="Use raw game data for non-parametric bootstrapping")
    parser.add_argument("--discount", type=float, default=0.98, help="Discount factor (gamma) for utilities")
    
    args = parser.parse_args()
    
    run_analysis(
        input_dir=args.input,
        output_dir=args.output,
        num_bootstrap=args.bootstrap,
        confidence=args.confidence,
        global_samples=args.global_samples,
        use_raw_bootstrap=args.raw_bootstrap,
        discount_factor=args.discount
    ) 