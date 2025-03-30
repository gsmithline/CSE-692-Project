"""
Nash equilibrium and bootstrapping module for game theoretical analysis.
"""

from nash_equilibrium.nash_solver import (
    milp_max_sym_ent_2p,
    replicator_dynamics_nash,
    calculate_max_regret,
    minimize_max_regret, 
    milp_nash_2p, 
    compute_regret
)

from nash_equilibrium.nash_regret_viz import (
    create_matrix_heatmap_with_nash_regret,
    create_matrix_with_relative_performance,
    create_dual_regret_visualization
)

from nash_equilibrium.bootstrap import (
    bootstrap_performance_metrics,
    analyze_bootstrap_results,
    visualize_dual_regret,
    plot_regret_distributions,
    compute_acceptance_ratio_matrix,
    prepare_raw_data_from_df,
    bootstrap_from_dataframe,
    plot_nash_distributions,
    visualize_nash_mixture_with_ci,
    plot_normal_regret_distributions,
    visualize_normal_regret_comparison
)

__all__ = [
    'milp_max_sym_ent_2p',
    'replicator_dynamics_nash',
    'calculate_max_regret',
    'minimize_max_regret',
    'create_matrix_heatmap_with_nash_regret',
    'create_matrix_with_relative_performance',
    'create_dual_regret_visualization',
    'bootstrap_performance_metrics',
    'analyze_bootstrap_results',
    'visualize_dual_regret',
    'plot_regret_distributions',
    'compute_acceptance_ratio_matrix',
    'prepare_raw_data_from_df',
    'bootstrap_from_dataframe',
    'plot_nash_distributions',
    'visualize_nash_mixture_with_ci',
    'plot_normal_regret_distributions',
    'visualize_normal_regret_comparison',
    'milp_nash_2p',
    'compute_regret'
] 