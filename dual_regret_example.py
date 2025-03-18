"""
Example script demonstrating how to use the dual regret visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nash_equilibrium.bootstrap import bootstrap_performance_metrics, analyze_bootstrap_results, visualize_dual_regret, plot_regret_distributions
from nash_equilibrium.nash_regret_viz import create_dual_regret_heatmap

# Add this code to your notebook to visualize both regret metrics

# Assuming you already have your performance_matrix defined
# and have run your bootstrap analysis:

# Example usage in your notebook:
"""
# After running your bootstrapping code:

# First, visualize the relationship between the two metrics
dual_regret_fig = visualize_dual_regret(
    bootstrap_results, 
    agents_for_bootstrap
)
plt.show()

# Then show the distribution of traditional regret
trad_regret_fig = plot_regret_distributions(
    bootstrap_results, 
    agents_for_bootstrap,
    regret_type='traditional_regret'
)
plt.show()

# And the distribution of relative performance (ne_regret)
rel_perf_fig = plot_regret_distributions(
    bootstrap_results, 
    agents_for_bootstrap,
    regret_type='ne_regret'
)
plt.show()

# Finally, create a heatmap showing both metrics together
dual_heatmap = create_dual_regret_heatmap(
    performance_matrix, 
    bootstrap_stats,
    title="Performance Matrix with Dual Regret Analysis"
)
plt.show()

# Analyze which agents perform best by different metrics
# Display top agents by traditional regret (lower is better)
print("\nTop 5 agents by Traditional Regret (lower is better):")
display(bootstrap_stats.sort_values('Mean Traditional Regret').head(5)[['Agent', 'Mean Traditional Regret', 'Std Traditional Regret']])

# Display top agents by relative performance (higher is better)
print("\nTop 5 agents by Relative Performance (higher is better):")
display(bootstrap_stats.sort_values('Mean NE Regret', ascending=False).head(5)[['Agent', 'Mean NE Regret', 'Std NE Regret']])
"""

# You can interpret the results as follows:
"""
1. Traditional Regret (max_utils - expected_utils)
   - Lower values = better agent performance
   - Measures how close each agent is to its own optimal strategy
   - Always non-negative

2. Relative Performance (expected_utils - nash_value)
   - Higher values = better agent performance
   - Measures how well agent performs against the Nash mixture compared to Nash vs itself
   - Can be positive (outperforms Nash) or negative (underperforms Nash)

3. Agents that have low traditional regret but negative relative performance:
   - Robust performers but not effective against the Nash mixture
   
4. Agents that have high traditional regret but positive relative performance:
   - Specialized strategies that counter the Nash mixture effectively
"""
