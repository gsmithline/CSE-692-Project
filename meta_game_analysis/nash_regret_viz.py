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
            print("These positive regrets indicate the Nash equilibrium calculation may not have fully converged.")
            print("Displaying actual regret values without modification.")
        
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
            ax_regret.set_xlim(min(regrets.min() * 1.1, -0.001), 0.001)  
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

def plot_regret_distributions(regrets, agent_names, title="Nash Equilibrium Regret Distribution", figsize=(12, 8)):
    """
    Plot distributions of Nash equilibrium regrets for each agent
    
    Args:
        regrets: List of regret vectors from bootstrap samples
        agent_names: List of agent names
        title: Plot title
        figsize: Figure size
    """
    try:
        regrets = np.stack(regrets)
    except ValueError:
        print("Warning: Bootstrap samples have inconsistent shapes. Using a more flexible approach.")
        first_regret = regrets[0]
        regrets_array = np.zeros((len(regrets), len(first_regret)))
        for i, regret in enumerate(regrets):
            if len(regret) == len(first_regret):
                regrets_array[i] = regret
            else:
                print(f"Warning: Skipping regret sample {i} due to shape mismatch")
        regrets = regrets_array
    
    n_agents = len(agent_names)
    
    fig, axs = plt.subplots(int(np.ceil(n_agents/3)), 3, figsize=figsize)
    axs = axs.flatten()
    
    # Check for any positive regrets
    epsilon = 1e-6  # Small threshold for numerical precision
    positive_regrets = (regrets > epsilon)
    
    if np.any(positive_regrets):
        count_positive = np.sum(positive_regrets)
        total_regrets = regrets.size
        percent_positive = (count_positive / total_regrets) * 100
        
        # Count regrets above epsilon by agent
        agent_violation_counts = np.sum(positive_regrets, axis=0)
        worst_agent_idx = np.argmax(agent_violation_counts)
        worst_agent = agent_names[worst_agent_idx]
        samples_per_agent = regrets.shape[0]
        
        print(f"Warning: {count_positive}/{total_regrets} regret values ({percent_positive:.2f}%) are above epsilon={epsilon:.2e}.")
        print(f"         Worst agent: '{worst_agent}' with {agent_violation_counts[worst_agent_idx]}/{samples_per_agent} samples ({(agent_violation_counts[worst_agent_idx]/samples_per_agent)*100:.2f}%) showing positive regrets.")
    
    # Add a note to the title about positive regrets if they exist
    positive_note = f"\n({count_positive}/{total_regrets} samples have positive regrets)" if np.any(positive_regrets) else ""
    fig.suptitle(f"{title}\n(Values should be ≤ 0 at equilibrium){positive_note}", 
                fontsize=16, fontweight='bold')
    
    for i, agent in enumerate(agent_names):
        if i < len(axs):
            agent_regrets = regrets[:, i]
            
            # Calculate bin edges to cover the entire range of data
            min_regret = min(agent_regrets)
            max_regret = max(agent_regrets)
            # Ensure we have enough bins to represent the full distribution
            # Add a small padding to the min/max to guarantee no values are outside the range
            bin_edges = np.linspace(min_regret - abs(min_regret)*0.01, 
                                  max_regret + abs(max_regret)*0.01, 
                                  40)  # Use 40 bins for higher resolution
            
            # Plot histogram with original (uncapped) regrets
            axs[i].hist(agent_regrets, bins=bin_edges, alpha=0.7, color='darkgreen')
            axs[i].set_title(agent)
            axs[i].set_xlabel('Nash Equilibrium Regret')
            axs[i].set_ylabel('Frequency')
            
            # Add mean line
            mean_regret = np.mean(agent_regrets)
            axs[i].axvline(mean_regret, color='r', linestyle='--', 
                          label=f'Mean: {mean_regret:.6f}')
            
            # Add 95% CI
            lower_ci = np.percentile(agent_regrets, 2.5)
            upper_ci = np.percentile(agent_regrets, 97.5)
            axs[i].axvline(lower_ci, color='orange', linestyle=':')
            axs[i].axvline(upper_ci, color='orange', linestyle=':', 
                          label=f'95% CI: [{lower_ci:.6f}, {upper_ci:.6f}]')
            
            # Add a reference line at 0 with explicit label
            axs[i].axvline(0, color='black', linestyle='-', alpha=0.7, 
                          label='Zero regret (equilibrium)')
            
            # Set x-axis limit to ensure all data points are visible
            left_buffer = abs(min_regret) * 0.1  # Add 10% buffer on left side
            right_buffer = abs(max_regret) * 0.1  # Add 10% buffer on right side
            axs[i].set_xlim(min_regret - left_buffer, max_regret + right_buffer)
            
            # Add a text annotation explaining what the plot shows
            axs[i].text(0.5, 0.97, "Regret must be ≤ 0 at equilibrium", 
                      transform=axs[i].transAxes, ha='center', va='top',
                      bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'),
                      fontsize=8)
            
            # Add text showing the full range of regret values
            axs[i].text(0.5, 0.89, f"Full range: [{min_regret:.2f}, {max_regret:.2f}]", 
                      transform=axs[i].transAxes, ha='center', va='top',
                      bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'),
                      fontsize=8)
            
            # If there are positive regrets for this agent, add info
            agent_pos_count = np.sum(agent_regrets > epsilon)
            if agent_pos_count > 0:
                axs[i].text(0.5, 0.82, 
                          f"{agent_pos_count}/{len(agent_regrets)} samples ({(agent_pos_count/len(agent_regrets))*100:.1f}%) above epsilon", 
                          transform=axs[i].transAxes, ha='center', va='top',
                          color='red', bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'),
                          fontsize=8)
            
            # Move legend to bottom left to avoid blocking data
            axs[i].legend(fontsize='small', loc='lower left')
    
    # Hide any unused subplots
    for j in range(i+1, len(axs)):
        axs[j].axis('off')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # Make room for the suptitle
    
    return fig 