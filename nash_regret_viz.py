import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

def create_matrix_heatmap_with_nash_regret(performance_matrix, nash_regrets, title="Performance Matrix with Nash Regret", 
                                         cmap="coolwarm", figsize=(16, 12)):
    """
    Create a heatmap visualization of a performance matrix with Nash regret values.
    
    Parameters:
    - performance_matrix: pandas DataFrame containing the performance data
    - nash_regrets: Series or DataFrame containing Nash regret values for each agent
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
    
    # Create a Series of Nash regrets aligned with performance matrix index
    regrets = pd.Series(index=numeric_matrix.index)
    
    # Process nash_regrets to get values
    if isinstance(nash_regrets, pd.DataFrame):
        if 'Mean NE Regret' in nash_regrets.columns and 'Agent' in nash_regrets.columns:
            # Convert DataFrame to Series indexed by Agent
            regrets_dict = dict(zip(nash_regrets['Agent'], nash_regrets['Mean NE Regret']))
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
    
    if len(regrets) > 0:
        # Get colors based on regret values (lower is better - use reversed colormap)
        norm = plt.Normalize(regrets.min(), regrets.max())
        colors = plt.cm.viridis_r(norm(regrets.values))
        
        # Create a mapping from index to position
        y_positions = {agent: i for i, agent in enumerate(numeric_matrix.index) if agent in regrets.index}
        
        # Plot bars in the same order as the heatmap rows
        for i, (agent, value) in enumerate(regrets.items()):
            y_pos = list(numeric_matrix.index).index(agent)
            ax_regret.barh(y_pos, value, color=colors[i], height=0.8)
            ax_regret.text(value + (regrets.max() * 0.02), y_pos, f"{value:.2f}", 
                          va='center', fontsize=9)
        
        # Set y-tick labels to agent names (same as heatmap)
        ax_regret.set_yticks(range(len(numeric_matrix.index)))
        ax_regret.set_yticklabels([])  # No need for labels, they're already on the heatmap
        
        # Set labels
        ax_regret.set_xlabel('Nash Equilibrium Regret', fontweight='bold')
        ax_regret.set_title('NE Regret', fontweight='bold')
        
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