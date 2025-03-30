#!/usr/bin/env python
# coding: utf-8

import os
import sys
import argparse
import pandas as pd
import numpy as np

# Set matplotlib configuration directory to a writable location
# This avoids the warning about matplotlib config directory not being writable
os.environ['MPLCONFIGDIR'] = os.path.join(os.path.expanduser('~'), '.matplotlib_temp')
os.makedirs(os.environ['MPLCONFIGDIR'], exist_ok=True)

import matplotlib.pyplot as plt

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
from nash_equilibrium.nash_regret_viz import create_matrix_heatmap_with_nash_regret

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
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("\nStep 1: Processing game data...")
    all_results, agent_performance, agent_final_rounds, agent_game_counts, agent_final_rounds_self_play = process_all_games(
        input_dir,
        discount_factor=discount_factor
    )
    
    # Step 2: Compute global maximum values
    print("\nStep 2: Computing global maximum values...")
    global_max_nash_welfare, global_standard_max = compute_global_max_values(num_samples=1000)
    print(f"Global max Nash welfare: {global_max_nash_welfare:.2f}")
    print(f"Global max social welfare: {global_standard_max:.2f}")
    
    print("\nStep 3: Creating performance matrices...")
    performance_matrices = create_performance_matrices(all_results, agent_performance, agent_final_rounds)
    
    all_agents = sorted(list(performance_matrices['overall_agent_performance'].keys()))
    
    print("\nStep 4: Creating welfare matrices...")
    welfare_matrices = create_welfare_matrices(all_results, all_agents, global_max_nash_welfare)
    
    print("\nStep 5: Cleaning matrix names...")
    cleaned_matrices = {}
    
    performance_matrix_names = ['performance_matrix', 'std_dev_matrix', 'variance_matrix', 'scaled_performance_matrix', 'count_matrix']
    for name in performance_matrix_names:
        if name in performance_matrices:
            cleaned_matrices[name] = clean_matrix_names(performance_matrices[name], get_display_name)
    
    for name, matrix in welfare_matrices.items():
        cleaned_matrices[name] = clean_matrix_names(matrix, get_display_name)
    
    print("\nStep 6: Filtering matrices...")
    exclude_agents = ['anthropic_3.7_sonnet_circle_4']  
    filtered_matrices = filter_matrices(cleaned_matrices, exclude_agents)
    
    print("\nStep 7: Running Nash equilibrium analysis...")
    performance_matrix = filtered_matrices['performance_matrix']
    
    print("\nChecking for pure Nash equilibria in the performance matrix:")
    print_pure_nash_info(performance_matrix)
    
    #print("\nFinding Nash equilibrium using replicator dynamics with multiple restarts:")
    #rd_nash_df = find_nash_with_replicator_dynamics(
    #    performance_matrix, 
    #    num_restarts=10,
    #    num_iterations=2000,
    #    verbose=True
    #)
    print("\nFinding Nash equilibrium using MILP:")
    rd_nash_df = milp_nash_2p(performance_matrix, 0.05)
    print("\nNash Equilibrium from Replicator Dynamics:")
    print(rd_nash_df)
    
    print("\nCalculating Max Entropy Nash Equilibrium for the plain performance matrix...")
    from nash_equilibrium.nash_solver import milp_max_sym_ent_2p
    
    performance_matrix_np = performance_matrix.to_numpy()
    
    for i in range(performance_matrix_np.shape[0]):
        for j in range(performance_matrix_np.shape[1]):
            if np.isnan(performance_matrix_np[i, j]):
                col_mean = np.nanmean(performance_matrix_np[:, j])
                if not np.isnan(col_mean):
                    performance_matrix_np[i, j] = col_mean
                else:
                    row_mean = np.nanmean(performance_matrix_np[i, :])
                    performance_matrix_np[i, j] = row_mean if not np.isnan(row_mean) else 0
    
    me_nash_strategy = milp_max_sym_ent_2p(performance_matrix_np)
    
    agents = performance_matrix.index.tolist()
    me_strategy_df = pd.DataFrame({
        'Agent': agents,
        'Nash Probability': me_nash_strategy
    }).sort_values(by='Nash Probability', ascending=False)
    
    print("\nMax Entropy Nash Equilibrium for plain performance matrix:")
    print(me_strategy_df)
    
    if use_raw_bootstrap:

        print("\nUsing non-parametric bootstrapping with raw game data...")
        bootstrap_results, bootstrap_stats, ne_strategy_df, agent_names = run_raw_data_nash_analysis(
            all_results,
            num_bootstrap_samples=num_bootstrap,
            confidence_level=confidence
        )
        
        # Create bootstrap analysis directory in the output dir
        bootstrap_dir = os.path.join(output_dir, 'bootstrap_analysis')
        os.makedirs(bootstrap_dir, exist_ok=True)
        
        # Generate distribution plots for non-parametric bootstrap results
        print("\nGenerating bootstrap distribution plots...")
        from meta_game_analysis.bootstrap_nonparametric import plot_regret_distributions, plot_ci_size_evolution
        
        # Extract regrets from bootstrap results
        ne_regrets = bootstrap_results['ne_regret']
        rd_regrets = bootstrap_results['rd_regret']
        agent_expected_utils = bootstrap_results.get('agent_expected_utility', [])

        # Add explicit calls to generate CI evolution plots
        print("\nGenerating confidence interval evolution plots...")
        try:
            # Create CI evolution plots for NE regrets
            if ne_regrets:
                ne_ci_plot = plot_ci_size_evolution(
                    {'ne_regret': ne_regrets}, 'ne_regret', agent_names, bootstrap_dir
                )
                print(f"Created NE regret CI evolution plot in {bootstrap_dir}")
            
            # Create CI evolution plots for RD regrets
            if rd_regrets:
                rd_ci_plot = plot_ci_size_evolution(
                    {'rd_regret': rd_regrets}, 'rd_regret', agent_names, bootstrap_dir
                )
                print(f"Created RD regret CI evolution plot in {bootstrap_dir}")
            
            # Create CI evolution plots for expected utilities
            if agent_expected_utils:
                eu_ci_plot = plot_ci_size_evolution(
                    {'agent_expected_utility': agent_expected_utils}, 
                    'agent_expected_utility', agent_names, bootstrap_dir
                )
                print(f"Created expected utility CI evolution plot in {bootstrap_dir}")
        except Exception as e:
            print(f"Error generating CI evolution plots: {str(e)}")
            import traceback
            traceback.print_exc()
        
        # Create distribution plots
        try:
            ne_dist_fig = plot_regret_distributions(ne_regrets, agent_names, 
                                                title="Nash Equilibrium Regret Distribution")
            if ne_dist_fig is not None:
                ne_dist_fig.savefig(os.path.join(bootstrap_dir, 'ne_regret_distribution.png'))
            else:
                print("Warning: Failed to create Nash Equilibrium Regret Distribution figure")
        except Exception as e:
            print(f"Error creating regret distribution plot: {str(e)}")
        
        try:
            rd_dist_fig = plot_regret_distributions(rd_regrets, agent_names,
                                                title="Replicator Dynamics Regret Distribution")
            if rd_dist_fig is not None:
                rd_dist_fig.savefig(os.path.join(bootstrap_dir, 'rd_regret_distribution.png'))
            else:
                print("Warning: Failed to create Replicator Dynamics Regret Distribution figure")
        except Exception as e:
            print(f"Error creating RD regret distribution plot: {str(e)}")
                
        # Generate box plots
        try:
            ne_box_fig = plot_regret_distributions(ne_regrets, agent_names, 
                                                title="Nash Equilibrium Regret Box Plot", 
                                                plot_type="box")
            if ne_box_fig is not None:
                ne_box_fig.savefig(os.path.join(bootstrap_dir, 'ne_regret_boxplot.png'))
            else:
                print("Warning: Failed to create Nash Equilibrium Regret Box Plot figure")
        except Exception as e:
            print(f"Error creating NE regret box plot: {str(e)}")
            
        try:
            rd_box_fig = plot_regret_distributions(rd_regrets, agent_names,
                                                title="Replicator Dynamics Regret Box Plot", 
                                                plot_type="box")
            if rd_box_fig is not None:
                rd_box_fig.savefig(os.path.join(bootstrap_dir, 'rd_regret_boxplot.png'))
            else:
                print("Warning: Failed to create Replicator Dynamics Regret Box Plot figure")
        except Exception as e:
            print(f"Error creating RD regret box plot: {str(e)}")
            
        # Generate running mean plots
        try:
            ne_running_fig = plot_regret_distributions(ne_regrets, agent_names, 
                                                    title="Nash Equilibrium Regret Running Mean", 
                                                    plot_type="running_mean")
            if ne_running_fig is not None:
                ne_running_fig.savefig(os.path.join(bootstrap_dir, 'ne_regret_running_mean.png'))
            else:
                print("Warning: Failed to create Nash Equilibrium Regret Running Mean figure")
        except Exception as e:
            print(f"Error creating NE regret running mean plot: {str(e)}")
            
        try:
            rd_running_fig = plot_regret_distributions(rd_regrets, agent_names,
                                                    title="Replicator Dynamics Regret Running Mean", 
                                                    plot_type="running_mean")
            if rd_running_fig is not None:
                rd_running_fig.savefig(os.path.join(bootstrap_dir, 'rd_regret_running_mean.png'))
            else:
                print("Warning: Failed to create Replicator Dynamics Regret Running Mean figure")
        except Exception as e:
            print(f"Error creating RD regret running mean plot: {str(e)}")
            
        # Additionally run the explicit convergence analysis to ensure all plots are created
        print("\nRunning explicit bootstrap convergence analysis to generate all plots...")
        try:
            from meta_game_analysis.bootstrap_nonparametric import analyze_bootstrap_results_for_convergence
            analyze_bootstrap_results_for_convergence(bootstrap_results, agent_names, bootstrap_dir)
            print(f"Completed explicit convergence analysis, check {bootstrap_dir} for all plots")
        except Exception as e:
            print(f"Error in explicit convergence analysis: {str(e)}")
            import traceback
            traceback.print_exc()
    else:
        print("\nUsing traditional bootstrapping with performance matrix...")
        bootstrap_results, bootstrap_stats, acceptance_matrix, ne_strategy_df = run_nash_analysis(
            performance_matrix,
            num_bootstrap_samples=num_bootstrap,
            confidence_level=confidence
        )
    
    print("\nMax Entropy Nash Equilibrium from bootstrapping:")
    print(ne_strategy_df)
    
    print("\nGenerating comprehensive Nash equilibrium statistics...")
    comparison_df, rd_regret_df, rd_nash_value = generate_all_nash_stats(
        performance_matrix, 
        bootstrap_stats, 
        ne_strategy_df, 
        rd_nash_df
    )
    
    print_rd_nash_summary(rd_regret_df, rd_nash_df, rd_nash_value)
    print_nash_comparison(comparison_df)
    
    if 'acceptance_matrix' not in filtered_matrices and len(all_results) > 0:
        agents = performance_matrix.index.tolist()
        acceptance_matrix = calculate_acceptance_ratio(all_results, agents)
        filtered_matrices['acceptance_matrix'] = clean_matrix_names(acceptance_matrix, get_display_name)
    
    print("\nStep 8: Creating visualizations...")
    welfare_figures = visualize_welfare_matrices(filtered_matrices, os.path.join(output_dir, 'heatmaps'))
    
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
    
    # Check if bootstrap_results contains RD regret data from non-parametric bootstrapping
    has_rd_regrets = False
    if use_raw_bootstrap and 'rd_regret' in bootstrap_results and bootstrap_results['rd_regret']:
        has_rd_regrets = True
        print("Found RD regret data from non-parametric bootstrapping. Including in visualizations.")
    
    # Get Nash equilibrium visualizations
    nash_plot_figures = plot_nash_distributions(
        bootstrap_results, 
        agents_list,
        include_rd_regrets=has_rd_regrets
    )
    
    # Save Nash plots
    save_nash_plots(nash_plot_figures, max_entropy_nash_dir)
    
    # Create and save normal regret visualization if available
    normal_regret_dir = os.path.join(output_dir, "normal_regret")
    os.makedirs(normal_regret_dir, exist_ok=True)
    
    # Extract normal regret plots if they exist
    normal_regret_figures = {
        k: v for k, v in nash_plot_figures.items() 
        if 'normal_regret' in k or 'normal_comparison' in k
    }
    
    if normal_regret_figures:
        print("Saving normal regret visualizations...")
        save_nash_plots(normal_regret_figures, normal_regret_dir)
    
    # Create performance matrix with Nash regret visualization
    print("Creating performance matrix with Nash regret visualization...")
    performance_with_regret_fig = create_matrix_heatmap_with_nash_regret(
        performance_matrix,
        bootstrap_stats,
        title="Performance Matrix with Nash Equilibrium Regret"
    )
    
    if performance_with_regret_fig is not None:
        performance_with_regret_path = os.path.join(output_dir, "performance_matrix_with_regret.png")
        performance_with_regret_fig.savefig(performance_with_regret_path, bbox_inches='tight', dpi=300)
        plt.close(performance_with_regret_fig)
    
    # Also save RD plots to the RD Nash directory if available
    if has_rd_regrets:
        rd_plot_figures = {
            'rd_ne_regret': nash_plot_figures.get('rd_ne_regret'),
            'dual_ne_regret': nash_plot_figures.get('dual_ne_regret')
        }
        # Only include plots that exist
        rd_plot_figures = {k: v for k, v in rd_plot_figures.items() if v is not None}
        if rd_plot_figures:  # Only save if we have any valid figures
            save_nash_plots(rd_plot_figures, rd_nash_dir)
    
    # Step 9: Save results to CSV files
    print("\nStep 9: Saving results to CSV files...")
    save_results_to_csv(
        filtered_matrices, 
        bootstrap_stats, 
        ne_strategy_df, 
        os.path.join(output_dir, 'csv')
    )
    
    csv_dir = os.path.join(output_dir, 'csv')
    os.makedirs(csv_dir, exist_ok=True)
    
    rd_regret_df.to_csv(os.path.join(csv_dir, 'rd_nash_regret.csv'), index=False)
    comparison_df.to_csv(os.path.join(csv_dir, 'nash_comparison.csv'), index=False)
    
    print("\nStep 10: Summary of results:")
    print(f"Total games analyzed: {len(all_results)}")
    print(f"Unique agent types: {len(performance_matrices['overall_agent_performance'])}")
    
    print("\nAverage Final Rounds by Agent:")
    for agent, avg_round in sorted(performance_matrices['average_final_rounds'].items()):
        print(f"{get_display_name(agent)}: {avg_round:.2f}")
    
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

def main():
    """Main function to run bootstrap analysis on a performance matrix."""
    # Load the performance matrix
    performance_matrix = pd.read_csv('performance_matrix.csv', index_col=0)
    
    # Run bootstrap analysis
    from meta_game_analysis.bootstrap_nonparametric import run_bootstrap_analysis, analyze_bootstrap_convergence, plot_regret_distributions
    from meta_game_analysis.nash_analysis import analyze_bootstrap_results
    
    # Create output directory
    output_dir = 'bootstrap_analysis'
    os.makedirs(output_dir, exist_ok=True)
    
    # Run bootstrap analysis
    bootstrap_results = run_bootstrap_analysis(performance_matrix)
    
    # Analyze bootstrap results
    agent_names = performance_matrix.index.tolist()
    bootstrap_stats = analyze_bootstrap_results(bootstrap_results, agent_names)
    
    # Save bootstrap statistics
    bootstrap_stats.to_csv(os.path.join(output_dir, 'bootstrap_statistics.csv'))
    
    # Generate convergence analysis and plots
    print("\nGenerating convergence analysis plots...")
    convergence_analysis = analyze_bootstrap_convergence(bootstrap_results, agent_names)
    
    # Extract regrets from bootstrap results for distribution plots
    ne_regrets = [result['ne_regrets'] for result in bootstrap_results]
    rd_regrets = [result['rd_regrets'] for result in bootstrap_results]
    
    # Create and save distribution plots
    print("\nGenerating regret distribution plots...")
    ne_dist_fig = plot_regret_distributions(ne_regrets, agent_names, 
                                          title="Nash Equilibrium Regret Distribution")
    if ne_dist_fig is not None:
        ne_dist_fig.savefig(os.path.join(output_dir, 'ne_regret_distribution.png'))
    else:
        print("Warning: Failed to create Nash Equilibrium Regret Distribution figure")
    
    rd_dist_fig = plot_regret_distributions(rd_regrets, agent_names,
                                          title="Replicator Dynamics Regret Distribution")
    if rd_dist_fig is not None:
        rd_dist_fig.savefig(os.path.join(output_dir, 'rd_regret_distribution.png'))
    else:
        print("Warning: Failed to create Replicator Dynamics Regret Distribution figure")
    
    print("\nGenerating regret box plots...")
    ne_box_fig = plot_regret_distributions(ne_regrets, agent_names, 
                                          title="Nash Equilibrium Regret Box Plot", 
                                          plot_type="box")
    if ne_box_fig is not None:
        ne_box_fig.savefig(os.path.join(output_dir, 'ne_regret_boxplot.png'))
    else:
        print("Warning: Failed to create Nash Equilibrium Regret Box Plot figure")
    
    rd_box_fig = plot_regret_distributions(rd_regrets, agent_names,
                                          title="Replicator Dynamics Regret Box Plot", 
                                          plot_type="box")
    if rd_box_fig is not None:
        rd_box_fig.savefig(os.path.join(output_dir, 'rd_regret_boxplot.png'))
    else:
        print("Warning: Failed to create Replicator Dynamics Regret Box Plot figure")
    
    # Generate running mean plots
    print("\nGenerating running mean plots...")
    ne_running_fig = plot_regret_distributions(ne_regrets, agent_names, 
                                             title="Nash Equilibrium Regret Running Mean", 
                                             plot_type="running_mean")
    if ne_running_fig is not None:
        ne_running_fig.savefig(os.path.join(output_dir, 'ne_regret_running_mean.png'))
    else:
        print("Warning: Failed to create Nash Equilibrium Regret Running Mean figure")
    
    rd_running_fig = plot_regret_distributions(rd_regrets, agent_names,
                                             title="Replicator Dynamics Regret Running Mean", 
                                             plot_type="running_mean")
    if rd_running_fig is not None:
        rd_running_fig.savefig(os.path.join(output_dir, 'rd_regret_running_mean.png'))
    else:
        print("Warning: Failed to create Replicator Dynamics Regret Running Mean figure")
    
    # Print summary statistics
    print("\nBootstrap Analysis Summary:")
    print("=" * 50)
    print("\nNash Equilibrium Statistics:")
    print("-" * 30)
    print(f"Mean NE Regret: {bootstrap_stats['Mean NE Regret'].mean():.6f}")
    print(f"Std NE Regret: {bootstrap_stats['Std NE Regret'].mean():.6f}")
    print(f"95% CI NE Regret: [{bootstrap_stats['CI Lower NE Regret'].mean():.6f}, {bootstrap_stats['CI Upper NE Regret'].mean():.6f}]")
    
    print("\nReplicator Dynamics Statistics:")
    print("-" * 30)
    print(f"Mean RD Regret: {bootstrap_stats['Mean RD Regret'].mean():.6f}")
    print(f"Std RD Regret: {bootstrap_stats['Std RD Regret'].mean():.6f}")
    print(f"95% CI RD Regret: [{bootstrap_stats['CI Lower RD Regret'].mean():.6f}, {bootstrap_stats['CI Upper RD Regret'].mean():.6f}]")
    
    print("\nExpected Utility Statistics:")
    print("-" * 30)
    print(f"Mean Expected Utility: {bootstrap_stats['Mean Expected Utility'].mean():.6f}")
    print(f"Std Expected Utility: {bootstrap_stats['Std Expected Utility'].mean():.6f}")
    print(f"95% CI Expected Utility: [{bootstrap_stats['CI Lower Expected Utility'].mean():.6f}, {bootstrap_stats['CI Upper Expected Utility'].mean():.6f}]")
    
    print("\nConvergence Analysis:")
    print("-" * 30)
    print(f"NE Regrets Converged: {'Yes' if convergence_analysis['ne_converged'] else 'No'}")
    print(f"Expected Utilities Converged: {'Yes' if convergence_analysis['eu_converged'] else 'No'}")
    print(f"RD Regrets Converged: {'Yes' if convergence_analysis['rd_converged'] else 'No'}")
    
    if not (convergence_analysis['ne_converged'] and convergence_analysis['eu_converged'] and convergence_analysis['rd_converged']):
        print("\nWARNING: Some statistics have not converged. Consider increasing the number of bootstrap samples.")
    
    print("\nAll plots and statistics have been saved to the 'bootstrap_analysis' directory.")

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