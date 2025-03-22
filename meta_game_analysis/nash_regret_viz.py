#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import os

def create_matrix_heatmap_with_nash_regret(performance_matrix, nash_regrets, regret_type, title="Performance Matrix with Nash Regret", 
                                         cmap="coolwarm", figsize=(16, 12)):
    """
    Create a heatmap visualization of a performance matrix with Nash regret values.
    Nash regret values should be at most 0 in a proper Nash equilibrium.
    
    Parameters:
    - performance_matrix: pandas DataFrame containing the performance data
    - nash_regrets: Series or DataFrame containing Nash regret values for each agent
    - regret_type: Type of regret to display ('Mean NE Regret')
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
    
    # Process nash_regrets to get values
    if isinstance(nash_regrets, pd.DataFrame):
        if regret_type in nash_regrets.columns and 'Agent' in nash_regrets.columns:
            # Convert DataFrame to Series indexed by Agent
            regrets_dict = dict(zip(nash_regrets['Agent'], nash_regrets[regret_type]))
            for agent in numeric_matrix.index:
                if agent in regrets_dict:
                    regrets[agent] = regrets_dict[agent]
                else:
                    regrets[agent] = np.nan
        else:
            # Try to use the DataFrame directly
            for agent in numeric_matrix.index:
                if agent in nash_regrets.index:
                    # Try to find a suitable column for regret values
                    for col in nash_regrets.columns:
                        if 'regret' in col.lower():
                            regrets[agent] = nash_regrets.loc[agent, col]
                            break
                    # If no regret column found, use the first numeric column
                    if np.isnan(regrets[agent]):
                        for col in nash_regrets.columns:
                            if pd.api.types.is_numeric_dtype(nash_regrets[col]):
                                regrets[agent] = nash_regrets.loc[agent, col]
                                break
    elif isinstance(nash_regrets, pd.Series):
        # Use Series directly
        for agent in numeric_matrix.index:
            if agent in nash_regrets.index:
                regrets[agent] = nash_regrets[agent]
    
    # Drop NaN values but maintain original order from performance matrix
    regrets = regrets.dropna()
    
    # For Nash Equilibrium regrets, ensure they are non-positive
    if 'NE Regret' in regret_type:
        epsilon = 1e-8  # Numerical tolerance
        if (regrets > epsilon).any():
            print(f"Warning: Some NE regrets are positive: {regrets[regrets > epsilon].values}")
            print("Setting these regrets to 0 for visualization correctness.")
            regrets[regrets > epsilon] = 0.0
        
        # For NE regrets, less negative is better (closer to 0)
        regrets = regrets.sort_values(ascending=False)
    else:
        # For any other metric, use standard sorting
        regrets = regrets.sort_values(ascending=True)
    
    if len(regrets) > 0:
        # Use appropriate colormap based on regret type
        if 'NE Regret' in regret_type:
            # For NE regrets, which should be ≤ 0, use a colormap that shows closer to 0 as better
            norm = plt.Normalize(regrets.min(), 0)  # 0 is the upper bound for NE regrets
            colors = plt.cm.RdYlGn(norm(regrets.values))  # Red (bad) to Yellow to Green (good)
        else:
            # For other metrics, use standard colormap
            norm = plt.Normalize(regrets.min(), regrets.max())
            colors = plt.cm.viridis_r(norm(regrets.values))  # Lower is better
        
        # Plot bars in the same order as determined by the sorting
        for i, (agent, value) in enumerate(regrets.items()):
            y_pos = list(numeric_matrix.index).index(agent)
            ax_regret.barh(y_pos, value, color=colors[i], height=0.8)
            
            # Add text to show the value with appropriate positioning
            if 'NE Regret' in regret_type:
                # For Nash regrets, adjust text position based on value
                text_offset = abs(regrets.min()) * 0.02 if regrets.min() < 0 else 0.01
                ax_regret.text(value - text_offset, y_pos, f"{value:.4f}", 
                              va='center', fontsize=9)
            else:
                # For other metrics, position text to the right of the bar
                ax_regret.text(value + (regrets.max() * 0.02), y_pos, f"{value:.2f}", 
                              va='center', fontsize=9)
        
        # Set y-tick labels to agent names (same as heatmap)
        ax_regret.set_yticks(range(len(numeric_matrix.index)))
        ax_regret.set_yticklabels([])  # No need for labels, they're already on the heatmap
        
        # Set labels based on regret type
        if 'NE Regret' in regret_type:
            ax_regret.set_xlabel('NE Regret\n(closer to 0 is better)', fontweight='bold')
            ax_regret.set_title('NE Regret\n(should be ≤ 0)', fontweight='bold')
            
            # Add a horizontal line at 0 for NE regrets
            ax_regret.axvline(x=0, color='black', linestyle='--', alpha=0.7)
            
            # Set x-axis limits to show all bars clearly
            ax_regret.set_xlim(min(regrets.min() * 1.1, -0.001), 0.001)  # Small buffer around 0
        else:
            ax_regret.set_xlabel(regret_type, fontweight='bold')
            ax_regret.set_title(regret_type, fontweight='bold')
        
        # Set y-axis limits to match heatmap
        ax_regret.set_ylim(ax_heatmap.get_ylim())
        
        # Remove unnecessary spines
        ax_regret.spines['top'].set_visible(False)
        ax_regret.spines['right'].set_visible(False)
        ax_regret.spines['left'].set_visible(False)
    else:
        ax_regret.text(0.5, 0.5, "No valid Nash regret data", 
                      ha='center', va='center', fontsize=12)
        ax_regret.axis('off')
    
    plt.tight_layout()
    return fig 

def visualize_regret_heatmaps(performance_matrix, bootstrap_stats, save_dir=None):
    """
    Create heatmap visualizations with Nash regret marginals.
    
    Args:
        performance_matrix: Performance matrix DataFrame
        bootstrap_stats: DataFrame with bootstrap statistics
        save_dir: Directory to save the plots
        
    Returns:
        dict: Dictionary of figure objects
    """
    figures = {}
    
    # Create output directory if needed
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # Create heatmap with Nash regret marginal
    fig_nash = create_matrix_heatmap_with_nash_regret(
        performance_matrix,
        bootstrap_stats,
        regret_type='Mean NE Regret',
        title="Performance Matrix with Nash Regret"
    )
    figures['performance_with_nash_regret'] = fig_nash
    
    if save_dir:
        filepath = os.path.join(save_dir, 'performance_with_nash_regret.png')
        fig_nash.savefig(filepath, bbox_inches='tight', dpi=300)
        plt.close(fig_nash)
    
    return figures 