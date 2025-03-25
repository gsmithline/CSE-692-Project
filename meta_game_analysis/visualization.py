#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from graphviz import Digraph
import os
import sys

def create_matrix_heatmap_with_marginals(matrix, title="Matrix Heatmap", cmap="coolwarm", 
                                         fmt=".2f", figsize=(14, 12), 
                                         vmin=None, vmax=None, center=None,
                                         annot=True, normalize_marginals=False):
    """
    Create a heatmap visualization of a matrix with row and column marginals.
    
    Parameters:
    - matrix: pandas DataFrame containing the matrix data
    - title: title for the plot
    - cmap: colormap to use
    - fmt: format string for annotations
    - figsize: figure size (width, height)
    - vmin, vmax, center: color scale parameters
    - annot: whether to annotate cells
    - normalize_marginals: whether to normalize marginals (mean=1)
    
    Returns:
    - fig: the matplotlib figure
    """
    
    numeric_matrix = matrix.apply(pd.to_numeric, errors='coerce')
    
    mask = np.isnan(numeric_matrix)
    
    # Calculate row and column means (ignoring NaN values)
    row_means = numeric_matrix.mean(axis=1, skipna=True)
    col_means = numeric_matrix.mean(axis=0, skipna=True)
    
    # Normalize if requested
    if normalize_marginals:
        row_mean_value = row_means.mean()
        col_mean_value = col_means.mean()
        if not np.isnan(row_mean_value) and row_mean_value != 0:
            row_means = row_means / row_mean_value
        if not np.isnan(col_mean_value) and col_mean_value != 0:
            col_means = col_means / col_mean_value
    
    # Create a figure with gridspec for layout
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(7, 6)  # Increased to 7 rows to accommodate column means at bottom
    
    # Main heatmap (central 4x4 grid)
    ax_heatmap = fig.add_subplot(gs[1:5, 0:4])
    
    # Row marginal (right side)
    ax_row = fig.add_subplot(gs[1:5, 4:6])
    
    # Column marginal (bottom)
    ax_col = fig.add_subplot(gs[5:6, 0:4])  # Moved to bottom
    
    # Title area
    ax_title = fig.add_subplot(gs[0:1, 0:6])  # Expanded to full width
    ax_title.axis('off')
    
    # Colorbar area
    ax_cbar = fig.add_subplot(gs[6:7, :])
    ax_cbar.axis('off')
    
    # Draw the heatmap with mask for NaN values
    hm = sns.heatmap(numeric_matrix, ax=ax_heatmap, cmap=cmap, annot=annot, fmt=fmt,
                cbar=False, vmin=vmin, vmax=vmax, center=center,
                linewidths=0.5, mask=mask)
    
    ax_heatmap.xaxis.tick_top()
    
    ax_heatmap.set_xticklabels(
        numeric_matrix.columns, 
        rotation=80, 
        ha='center', 
        rotation_mode='anchor', 
        fontweight='bold',
    )
    
    ax_heatmap.tick_params(axis='x', pad=90)  # Changed from 15 to 25 to move labels higher

    
    ax_heatmap.set_yticklabels(
        ax_heatmap.get_yticklabels(),
        fontweight='bold'
    )
    
    if not row_means.isna().all(): 
        valid_row_means = row_means.dropna()
        if len(valid_row_means) > 0 and valid_row_means.max() > valid_row_means.min():
            color_norm = (valid_row_means - valid_row_means.min()) / (valid_row_means.max() - valid_row_means.min())
            colors = plt.cm.get_cmap(cmap)(color_norm)
        else:
            colors = plt.cm.get_cmap(cmap)(0.5 * np.ones_like(valid_row_means))
        
        y_pos = np.arange(len(row_means)) + 0.5  # Center on cells
        ax_row.barh(y_pos, row_means, color=colors, edgecolor='k', linewidth=0.5, height=0.8)
        
        for i, value in enumerate(row_means):
            if not np.isnan(value):
                ax_row.text(value + (row_means.max() * 0.02), i + 0.5, f"{value:.2f}", 
                         va='center', fontsize=9)
    
    ax_row.set_yticks([])
    ax_row.set_xlabel('Row Mean')
    ax_row.spines['top'].set_visible(False)
    ax_row.spines['right'].set_visible(False)
    ax_row.spines['left'].set_visible(False)
    ax_row.set_ylim(ax_heatmap.get_ylim())  # Match heatmap y-limits

    ax_title.text(0.95, 0.5, title, 
              horizontalalignment='right',
              verticalalignment='center', 
              fontsize=16, 
              fontweight='bold',
              transform=ax_title.transAxes)
    
    plt.tight_layout()
    return fig

def create_best_response_graph(performance_matrix, filename='best_response_graph_graphviz', save_dir=None):
    """
    Create a best response graph visualization.
    
    Args:
        performance_matrix: Performance matrix DataFrame
        filename: Base name for the output file
        save_dir: Directory to save the output file
        
    Returns:
        graphviz.Digraph: The Graphviz graph object
    """
    dot = Digraph(comment='NW Response Graph', format='png')
    dot.attr(rankdir='LR', size='12,8', dpi='300', bgcolor='white')
    dot.attr('node', shape='box', style='filled,rounded', fillcolor='lightblue', 
             fontname='Arial', fontsize='14', margin='0.4,0.3')

    strategy_names = performance_matrix.columns.tolist()

    for strategy in strategy_names:
        dot.node(strategy, strategy, width='2.5', height='0.8')

    for strategy in strategy_names:
        column_values = performance_matrix[strategy]
        
        best_response = column_values.idxmax()
        best_value = column_values.max()
        
        dot.edge(strategy, best_response, label=f" {best_value:.2f}", color='#3366FF')

    if save_dir:
        # Create the directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        full_path = os.path.join(save_dir, filename)
    else:
        full_path = filename
        
    dot.render(full_path, view=False, cleanup=True)
    
    return dot

def create_complete_best_response_graph(performance_matrix, filename='complete_best_response', save_dir=None):
    """
    Create a complete best response graph showing both player perspectives.
    
    Args:
        performance_matrix: Performance matrix DataFrame
        filename: Base name for the output file
        save_dir: Directory to save the output file
        
    Returns:
        graphviz.Digraph: The Graphviz graph object
    """
    dot = Digraph(comment='Complete Best Response Graph', format='png')
    dot.attr(rankdir='LR', size='14,10', dpi='300', bgcolor='white')
    dot.attr('node', shape='box', style='filled,rounded', fontname='Arial', fontsize='14', margin='0.4,0.3')
    
    strategy_names = performance_matrix.columns.tolist()

    for strategy in strategy_names:
        dot.node(strategy, strategy, width='2.5', height='0.8', fillcolor='lightblue')
    
    for p2_strategy in strategy_names:
        p1_payoffs = performance_matrix[p2_strategy]
        
        p1_best_response = p1_payoffs.idxmax()
        p1_best_value = p1_payoffs.max()
        
        dot.edge(p2_strategy, p1_best_response, 
                label=f" P1:{p1_best_value:.2f}", 
                color='#3366FF', 
                fontcolor='#3366FF')
    
 
    p2_matrix = performance_matrix.transpose()  
    
    for p1_strategy in strategy_names:
        p2_payoffs = p2_matrix[p1_strategy]
        
        p2_best_response = p2_payoffs.idxmax()
        p2_best_value = p2_payoffs.max()
        
        dot.edge(p1_strategy, p2_best_response, 
                label=f" P2:{p2_best_value:.2f}", 
                color='#FF6347', 
                fontcolor='#FF6347')
    
    if save_dir:
        # Create the directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        full_path = os.path.join(save_dir, filename)
    else:
        full_path = filename
        
    dot.render(full_path, view=False, cleanup=True)
    
    return dot

def visualize_welfare_matrices(matrices, save_dir=None):
    """
    Create heatmap visualizations for all welfare and performance matrices.
    
    Args:
        matrices: Dictionary of matrices to visualize
        save_dir: Directory to save the plots
        
    Returns:
        dict: Dictionary of figure objects
    """
    figures = {}
    
    # Create output directory if needed
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        
    matrix_configs = {
        "Performance Matrix": {"matrix": matrices.get("performance_matrix"), "cmap": "viridis", "center": None},
        "Scaled Performance Matrix": {"matrix": matrices.get("scaled_performance_matrix"), "cmap": "viridis", "center": None},
        "Nash Welfare Matrix": {"matrix": matrices.get("nash_welfare_matrix"), "cmap": "viridis", "center": None},
        "Utilitarian Welfare Matrix": {"matrix": matrices.get("utilitarian_welfare_matrix"), "cmap": "viridis", "center": None},
        "Rawls Welfare Matrix": {"matrix": matrices.get("rawls_welfare_matrix"), "cmap": "viridis", "center": None},
        "Gini Coefficient Matrix": {"matrix": matrices.get("gini_matrix"), "cmap": "coolwarm_r", "center": 0.5},
        "Jain's Fairness Index Matrix": {"matrix": matrices.get("jain_matrix"), "cmap": "viridis", "center": None},
        "Envy-Free Matrix": {"matrix": matrices.get("envy_free_matrix"), "cmap": "viridis", "center": None},
        "EF1 Matrix": {"matrix": matrices.get("ef1_matrix"), "cmap": "viridis", "center": None},
        "Pareto-Front-Count": {"matrix": matrices.get("pareto_matrix"), "cmap": "viridis", "center": None}
    }

    for name, config in matrix_configs.items():
        matrix = config["matrix"]
        if matrix is not None:
            fig = create_matrix_heatmap_with_marginals(
                matrix, 
                title=name,
                cmap=config["cmap"],
                center=config["center"]
            )
            figures[name] = fig
            
            if save_dir:
                filename = name.replace(" ", "_").replace("'", "").lower() + ".png"
                filepath = os.path.join(save_dir, filename)
                fig.savefig(filepath, bbox_inches='tight', dpi=300)
                plt.close(fig)
                
    return figures

def visualize_nash_equilibrium(bootstrap_stats, ne_strategy_df, save_dir=None):
    """
    Create visualizations for Nash equilibrium results.
    
    Args:
        bootstrap_stats: DataFrame with bootstrap statistics
        ne_strategy_df: DataFrame with Nash equilibrium strategy probabilities
        save_dir: Directory to save the plots
        
    Returns:
        dict: Dictionary of figure objects
    """
    figures = {}
    
    # Create output directory if needed
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # Nash equilibrium strategy bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(ne_strategy_df['Agent'], ne_strategy_df['Nash Probability'])
    ax.set_xticklabels(ne_strategy_df['Agent'], rotation=90)
    ax.set_title('Nash Equilibrium Mixed Strategy')
    ax.set_ylabel('Probability')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    figures['nash_strategy'] = fig
    
    if save_dir:
        filepath = os.path.join(save_dir, 'nash_equilibrium_strategy.png')
        fig.savefig(filepath, bbox_inches='tight', dpi=300)
        plt.close(fig)
    
    # Agent regret bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    bootstrap_stats_sorted = bootstrap_stats.sort_values('Mean NE Regret')
    ax.barh(bootstrap_stats_sorted['Agent'], bootstrap_stats_sorted['Mean NE Regret'])
    ax.set_title('Agent Nash Equilibrium Regret (Lower is Better)')
    ax.set_xlabel('Mean NE Regret')
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    figures['agent_regret'] = fig
    
    if save_dir:
        filepath = os.path.join(save_dir, 'agent_nash_regret.png')
        fig.savefig(filepath, bbox_inches='tight', dpi=300)
        plt.close(fig)
        
    return figures

def save_results_to_csv(matrices, bootstrap_stats, ne_strategy_df, save_dir):
    """
    Save all results to CSV files.
    
    Args:
        matrices: Dictionary of matrices
        bootstrap_stats: DataFrame with bootstrap statistics
        ne_strategy_df: DataFrame with Nash equilibrium strategy probabilities
        save_dir: Directory to save the CSV files
    """
    # Create the directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Save matrices
    for name, matrix in matrices.items():
        if matrix is not None:
            filename = name.replace(' ', '_').lower() + '.csv'
            filepath = os.path.join(save_dir, filename)
            matrix.to_csv(filepath)
    
    # Save bootstrap statistics
    bootstrap_stats.to_csv(os.path.join(save_dir, 'bootstrap_statistics.csv'))
    
    # Save Nash equilibrium strategy
    ne_strategy_df.to_csv(os.path.join(save_dir, 'nash_equilibrium_strategy.csv'), index=False)

def create_matrix_heatmap_with_rd_regret(performance_matrix, rd_regret_df, title="Performance Matrix with RD Nash Regret", 
                                      cmap="coolwarm", figsize=(16, 12)):
    """
    Create a heatmap visualization of a performance matrix with Replicator Dynamics Nash regret values.
    
    Parameters:
    - performance_matrix: pandas DataFrame containing the performance data
    - rd_regret_df: DataFrame containing regret values calculated against RD Nash
    - title: title for the plot
    - cmap: colormap to use
    - figsize: figure size (width, height)
    
    Returns:
    - fig: the matplotlib figure
    """
    # Convert performance matrix to numeric values
    numeric_matrix = performance_matrix.copy()
    
    # Convert all values to numeric, coercing errors to NaN
    for col in numeric_matrix.columns:
        numeric_matrix[col] = pd.to_numeric(numeric_matrix[col], errors='coerce')
    
    # Create mask for NaN values
    mask = np.isnan(numeric_matrix)
    
    # Create a figure with grid layout
    fig = plt.figure(figsize=figsize)
    gs = plt.GridSpec(1, 5, figure=fig, width_ratios=[4, 1, 0.2, 0.2, 0.2])
    
    # Main heatmap
    ax_heatmap = fig.add_subplot(gs[0, 0])
    
    # Nash regret subplot
    ax_regret = fig.add_subplot(gs[0, 1])
    
    # Draw the main heatmap
    sns.heatmap(numeric_matrix, ax=ax_heatmap, cmap=cmap, annot=True, fmt=".2f",
                cbar=False, linewidths=0.5, mask=mask)
    
    ax_heatmap.set_title(title, fontsize=16, fontweight='bold')
    ax_heatmap.set_xticklabels(ax_heatmap.get_xticklabels(), rotation=45, ha='right', fontweight='bold')
    ax_heatmap.set_yticklabels(ax_heatmap.get_yticklabels(), fontweight='bold')
    
    # Create a Series of regrets aligned with performance matrix index
    regrets = pd.Series(index=numeric_matrix.index)
    
    # Extract RD Nash regret values from rd_regret_df
    regret_type = 'RD Nash Regret'
    for agent in numeric_matrix.index:
        idx = rd_regret_df[rd_regret_df['Agent'] == agent].index
        if len(idx) > 0:
            regrets[agent] = rd_regret_df.loc[idx[0], regret_type]
        else:
            regrets[agent] = np.nan
    
    # Drop NaN values but maintain original order from performance matrix
    regrets = regrets.dropna()
    
    if len(regrets) > 0:
        # Check for any positive regrets
        epsilon = 1e-8
        if (regrets > epsilon).any():
            print(f"Warning: Some RD Nash regrets are positive: {regrets[regrets > epsilon].values}")
            print("These positive regrets indicate the Nash equilibrium calculation may not have fully converged.")
            print("Displaying actual regret values without modification.")
            
        # Sort by regret (closer to 0 is better)
        regrets = regrets.sort_values(ascending=False)
            
        # Determine the normalization range based on actual values
        min_val = regrets.min()
        max_val = max(0, regrets.max())  # Use 0 as upper bound if all regrets are negative
        
        # If we have positive regrets, adjust the colormap to show positive as worse (red)
        if max_val > 0:
            # Use a diverging colormap centered at 0
            norm = plt.Normalize(min_val, max_val) 
            colors = plt.cm.RdYlGn_r(norm(regrets.values))  # Reversed so negative (good) is green
        else:
            # All regrets are negative, use standard colormap
            norm = plt.Normalize(min_val, 0)  # 0 is the upper bound
            colors = plt.cm.RdYlGn(norm(regrets.values))
        
        # Plot bars in the same order as the heatmap rows
        for i, (agent, value) in enumerate(regrets.items()):
            y_pos = list(numeric_matrix.index).index(agent)
            ax_regret.barh(y_pos, value, color=colors[i], height=0.8)
            
            # Position text based on value
            text_offset = abs(regrets.min()) * 0.02 if regrets.min() < 0 else 0.01
            ax_regret.text(value - text_offset, y_pos, f"{value:.4f}", 
                          va='center', fontsize=9)
        
        # Set y-tick labels to agent names (same as heatmap)
        ax_regret.set_yticks(range(len(numeric_matrix.index)))
        ax_regret.set_yticklabels([])  # No need for labels, they're already on the heatmap
        
        # Set labels and add reference line at 0
        ax_regret.set_xlabel('RD Nash Regret\n(closer to 0 is better)', fontweight='bold')
        ax_regret.set_title('RD Nash Regret\n(should be ≤ 0)', fontweight='bold')
        ax_regret.axvline(x=0, color='black', linestyle='--', alpha=0.7)
        
        # Set x-axis limits to show all bars clearly with appropriate buffers
        min_val = min(regrets.min() * 1.1, -0.001)  # Buffer for negative values
        # Use the actual max value instead of a fixed limit to show all positive regrets
        max_val = max(regrets.max() * 1.1, 0.001)  # Buffer for positive values or small default
        ax_regret.set_xlim(min_val, max_val)
        
        # Set y-axis limits to match heatmap
        ax_regret.set_ylim(ax_heatmap.get_ylim())
        
        # Remove unnecessary spines
        ax_regret.spines['top'].set_visible(False)
        ax_regret.spines['right'].set_visible(False)
        ax_regret.spines['left'].set_visible(False)
    else:
        ax_regret.text(0.5, 0.5, "No valid regret data", 
                      ha='center', va='center', fontsize=12)
        ax_regret.axis('off')
    
    plt.tight_layout()
    return fig

def visualize_rd_regret_heatmaps(performance_matrix, rd_regret_df, save_dir=None):
    """
    Create heatmap visualizations with RD Nash regret marginals.
    
    Args:
        performance_matrix: Performance matrix DataFrame
        rd_regret_df: DataFrame with regret values calculated against RD Nash
        save_dir: Directory to save the plots
        
    Returns:
        dict: Dictionary of figure objects
    """
    figures = {}
    
    # Create output directory if needed
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # Create heatmap with RD Nash regret marginal
    fig_nash = create_matrix_heatmap_with_rd_regret(
        performance_matrix,
        rd_regret_df,
        title="Performance Matrix with RD Nash Regret"
    )
    figures['performance_with_rd_nash_regret'] = fig_nash
    
    if save_dir:
        filepath = os.path.join(save_dir, 'performance_with_rd_nash_regret.png')
        fig_nash.savefig(filepath, bbox_inches='tight', dpi=300)
        plt.close(fig_nash)
    
    return figures

def visualize_nash_comparison(comparison_df, save_dir=None):
    """
    Create visualizations comparing the two Nash equilibrium concepts.
    
    Args:
        comparison_df: DataFrame with comparison of both Nash approaches
        save_dir: Directory to save the plots
        
    Returns:
        dict: Dictionary of figure objects
    """
    figures = {}
    
    # Create output directory if needed
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # Bar chart comparing Nash probabilities
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Sort by Max Entropy Nash probability
    ordered_df = comparison_df.sort_values('ME Nash Probability', ascending=False)
    agents = ordered_df['Agent']
    
    x = np.arange(len(agents))
    width = 0.35
    
    ax.bar(x - width/2, ordered_df['ME Nash Probability'], width, label='Max Entropy Nash')
    ax.bar(x + width/2, ordered_df['RD Nash Probability'], width, label='Replicator Dynamics Nash')
    
    ax.set_ylabel('Nash Probability')
    ax.set_title('Comparison of Nash Equilibrium Probabilities')
    ax.set_xticks(x)
    ax.set_xticklabels(agents, rotation=45, ha='right')
    ax.legend()
    
    plt.tight_layout()
    
    figures['nash_probability_comparison'] = fig
    
    if save_dir:
        filepath = os.path.join(save_dir, 'nash_probability_comparison.png')
        fig.savefig(filepath, bbox_inches='tight', dpi=300)
        plt.close(fig)
    
    # Scatter plot of ME vs RD Nash Regret
    fig, ax = plt.subplots(figsize=(10, 10))
    
    ax.scatter(comparison_df['ME Nash Regret'], comparison_df['RD Nash Regret'], s=80, alpha=0.7)
    
    # Add agent labels
    for i, row in comparison_df.iterrows():
        ax.annotate(row['Agent'], 
                   (row['ME Nash Regret'], row['RD Nash Regret']),
                   fontsize=9,
                   xytext=(5, 5),
                   textcoords='offset points')
    
    # Add diagonal line
    min_val = min(comparison_df['ME Nash Regret'].min(), comparison_df['RD Nash Regret'].min())
    max_val = max(comparison_df['ME Nash Regret'].max(), comparison_df['RD Nash Regret'].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.3)
    
    ax.set_xlabel('Max Entropy Nash Regret')
    ax.set_ylabel('Replicator Dynamics Nash Regret')
    ax.set_title('Comparison of Nash Regret Metrics')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    figures['nash_regret_comparison'] = fig
    
    if save_dir:
        filepath = os.path.join(save_dir, 'nash_regret_comparison.png')
        fig.savefig(filepath, bbox_inches='tight', dpi=300)
        plt.close(fig)
    
    return figures

def plot_regret_distributions(regrets, agent_names, title="Nash Equilibrium Regret Distribution", figsize=(12, 8)):
    """
    Plot distributions of Nash equilibrium regrets for each agent with extreme memory optimization
    
    Args:
        regrets: List of regret vectors from bootstrap samples
        agent_names: List of agent names
        title: Plot title
        figsize: Figure size
    """
    # Print regret range for debugging
    # Check regret range without loading all into memory at once
    min_overall = float('inf')
    max_overall = float('-inf')
    
    # Sample a subset of regrets to estimate range
    sample_size = min(100, len(regrets))
    sample_indices = np.linspace(0, len(regrets)-1, sample_size, dtype=int)
    
    for i in sample_indices:
        r = np.array(regrets[i], dtype=np.float32)  # Use float32 to reduce memory
        if r.size > 0:
            min_val = np.min(r)
            max_val = np.max(r)
            min_overall = min(min_overall, min_val)
            max_overall = max(max_overall, max_val)
    
    print(f"Regret range for {title}: [{min_overall:.8f}, {max_overall:.8f}]")
    
    # Check for extreme range that might cause memory issues
    range_magnitude = abs(min_overall - max_overall)
    
    # For extreme ranges, use aggressive downsampling and fewer bins
    MAX_BOOTSTRAP_SAMPLES = 500 if range_magnitude > 100 else 1000
    
    # Process regrets in batches if there are many agents to reduce memory usage
    n_agents = len(agent_names)
    
    # Use smaller batch size for extreme ranges
    batch_size = 6 if range_magnitude > 100 else min(9, n_agents)  # Use 2x3 grid for extreme ranges
    
    # Track positive regrets across all batches
    overall_positive_count = 0
    overall_total_regrets = 0
    
    first_fig = None
    
    # Process one agent at a time for extreme ranges
    for batch_start in range(0, n_agents, batch_size):
        batch_end = min(batch_start + batch_size, n_agents)
        batch_agent_names = agent_names[batch_start:batch_end]
        batch_size_actual = len(batch_agent_names)
        
        # Create figure outside the agent loop
        fig, axs = plt.subplots(int(np.ceil(batch_size_actual/3)), 3, figsize=figsize)
        axs = axs.flatten()
        
        # Track positive regrets for this batch
        batch_positive_count = 0
        batch_total_regrets = 0
        
        # Process each agent separately to minimize memory usage
        for i, agent_idx in enumerate(range(batch_start, batch_end)):
            if agent_idx >= len(agent_names):
                continue
                
            agent = agent_names[agent_idx]
            
            # Calculate statistics in a memory-efficient way
            agent_regrets = []
            positive_count = 0
            
            # Use downsampling for large datasets
            if len(regrets) > MAX_BOOTSTRAP_SAMPLES:
                # Select samples evenly distributed throughout the dataset
                sample_indices = np.linspace(0, len(regrets)-1, MAX_BOOTSTRAP_SAMPLES, dtype=int)
                regrets_to_process = [regrets[j] for j in sample_indices]
            else:
                regrets_to_process = regrets
                
            for r in regrets_to_process:
                try:
                    # Extract just this agent's regret
                    if isinstance(r, np.ndarray) and agent_idx < r.shape[0]:
                        val = r[agent_idx]
                        agent_regrets.append(val)
                        if val > 1e-6:  # Check for positive regrets
                            positive_count += 1
                except (IndexError, TypeError) as e:
                    continue  # Skip problematic samples
            
            # Skip if no valid regrets
            if not agent_regrets:
                continue
                
            # Convert to numpy array for faster calculations
            agent_regrets = np.array(agent_regrets, dtype=np.float32)  # Use float32 to reduce memory
            total_samples = len(agent_regrets)
            
            # Update counts
            batch_positive_count += positive_count
            batch_total_regrets += total_samples
            
            # Calculate basic statistics
            min_regret = np.min(agent_regrets)
            max_regret = np.max(agent_regrets)
            mean_regret = np.mean(agent_regrets)
            
            # Calculate percentiles directly
            lower_ci = np.percentile(agent_regrets, 2.5)
            upper_ci = np.percentile(agent_regrets, 97.5)
            
            # Determine bin count based on range magnitude - more aggressive reduction
            regret_range = max_regret - min_regret
            if regret_range > 100:
                bin_count = 10  # Very few bins for extreme ranges
            elif regret_range > 50:
                bin_count = 15  # Fewer bins for large ranges
            elif regret_range > 10:
                bin_count = 20  # Medium bins for medium ranges
            else:
                bin_count = 30  # More bins for small ranges
            
            # Calculate histogram in memory-efficient way
            hist, bin_edges = np.histogram(agent_regrets, bins=bin_count)
            
            # Plot histogram
            axs[i].bar(bin_edges[:-1], hist, width=bin_edges[1]-bin_edges[0], alpha=0.7, color='darkgreen')
            axs[i].set_title(agent)
            axs[i].set_xlabel('Nash Equilibrium Regret')
            axs[i].set_ylabel('Frequency')
            
            # Add mean line
            axs[i].axvline(mean_regret, color='r', linestyle='--', 
                         label=f'Mean: {mean_regret:.2f}')  # Further reduce precision
            
            # Add 95% CI
            axs[i].axvline(lower_ci, color='orange', linestyle=':')
            axs[i].axvline(upper_ci, color='orange', linestyle=':', 
                         label=f'95% CI: [{lower_ci:.2f}, {upper_ci:.2f}]')  # Further reduce precision
            
            # Add a reference line at 0
            axs[i].axvline(0, color='black', linestyle='-', alpha=0.7, 
                         label='Zero (equilibrium)')
            
            # Set x-axis limit with smaller buffer for extreme ranges
            buffer_factor = 0.05 if regret_range > 50 else 0.1
            left_buffer = abs(min_regret) * buffer_factor
            right_buffer = abs(max_regret) * buffer_factor
            axs[i].set_xlim(min_regret - left_buffer, max_regret + right_buffer)
            
            # Add text with key statistics
            axs[i].text(0.5, 0.97, "Regret ≤ 0 at equilibrium", 
                      transform=axs[i].transAxes, ha='center', va='top',
                      bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'),
                      fontsize=8)
            
            axs[i].text(0.5, 0.89, f"Range: [{min_regret:.1f}, {max_regret:.1f}]", 
                      transform=axs[i].transAxes, ha='center', va='top',
                      bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'),
                      fontsize=8)
            
            # If there are positive regrets for this agent, add info
            if positive_count > 0:
                axs[i].text(0.5, 0.82, 
                          f"{positive_count}/{total_samples} samples > 0", 
                          transform=axs[i].transAxes, ha='center', va='top',
                          color='red', bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'),
                          fontsize=8)
            
            # Add legend but with fewer items for memory efficiency
            axs[i].legend(fontsize='x-small', loc='lower left')
            
            # Explicit garbage collection to free memory
            del agent_regrets
            del hist
        
        # Update overall counts
        overall_positive_count += batch_positive_count
        overall_total_regrets += batch_total_regrets
        
        # Add batch summary title
        positive_note = f"\n({batch_positive_count}/{batch_total_regrets} samples have positive regrets)" if batch_positive_count > 0 else ""
        fig.suptitle(f"{title} (Agents {batch_start+1}-{batch_end})\n(Values should be ≤ 0 at equilibrium){positive_note}", 
                    fontsize=14, fontweight='bold')
        
        # Hide any unused subplots
        for j in range(batch_size_actual, len(axs)):
            axs[j].axis('off')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)  # Make room for the suptitle
        
        # Save the first figure for return
        if batch_start == 0:
            first_fig = fig
        
        # Force garbage collection after each batch
        import gc
        gc.collect()
    
    print(f"Summary for {title}: {overall_positive_count}/{overall_total_regrets} samples had positive regrets")
    
    return first_fig if first_fig is not None else None 