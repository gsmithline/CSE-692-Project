"""
Meta-game analysis tools for AI negotiation.
"""

from meta_game_analysis.data_processing import (
    parse_values,
    parse_outside_offer,
    calculate_value,
    compute_max_nash_welfare,
    analyze_single_game,
    get_canonical_name,
    get_display_name,
    process_all_games,
    compute_global_max_values
)

from meta_game_analysis.matrix_creation import (
    create_performance_matrices,
    create_welfare_matrices,
    clean_matrix_names,
    filter_matrices
)

from meta_game_analysis.visualization import (
    create_matrix_heatmap_with_marginals,
    create_best_response_graph,
    create_complete_best_response_graph,
    visualize_welfare_matrices,
    visualize_nash_equilibrium,
    save_results_to_csv,
    create_matrix_heatmap_with_rd_regret,
    visualize_rd_regret_heatmaps,
    visualize_nash_comparison
)

from meta_game_analysis.nash_regret_viz import (
    create_matrix_heatmap_with_nash_regret,
    visualize_regret_heatmaps
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
    calculate_regrets_against_replicator_nash,
    generate_all_nash_stats,
    print_rd_nash_summary,
    print_nash_comparison
)

from meta_game_analysis.bootstrap_nonparametric import (
    nonparametric_bootstrap_from_raw_data,
    analyze_bootstrap_results
)

__all__ = [
    # data_processing
    'parse_values', 'parse_outside_offer', 'calculate_value', 'compute_max_nash_welfare',
    'analyze_single_game', 'get_canonical_name', 'get_display_name', 'process_all_games', 
    'compute_global_max_values',
    
    # matrix_creation
    'create_performance_matrices', 'create_welfare_matrices', 'clean_matrix_names', 'filter_matrices',
    
    # visualization
    'create_matrix_heatmap_with_marginals', 'create_best_response_graph', 
    'create_complete_best_response_graph', 'visualize_welfare_matrices',
    'visualize_nash_equilibrium', 'save_results_to_csv',
    'create_matrix_heatmap_with_rd_regret', 'visualize_rd_regret_heatmaps', 
    'visualize_nash_comparison',
    
    # nash_regret_viz
    'create_matrix_heatmap_with_nash_regret', 'visualize_regret_heatmaps',
    
    # nash_analysis
    'run_nash_analysis', 'run_raw_data_nash_analysis', 'plot_nash_distributions', 'save_nash_plots',
    'print_nash_summary', 'calculate_acceptance_ratio', 'print_pure_nash_info', 'find_nash_with_replicator_dynamics',
    'calculate_regrets_against_replicator_nash', 'generate_all_nash_stats', 'print_rd_nash_summary',
    'print_nash_comparison',
    
    # bootstrap_nonparametric
    'nonparametric_bootstrap_from_raw_data', 'analyze_bootstrap_results'
] 