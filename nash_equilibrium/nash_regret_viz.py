import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec

def create_matrix_heatmap_with_nash_regret(performance_matrix, nash_regrets, title="Performance Matrix with Nash Regret", 
                                         cmap="coolwarm", figsize=(16, 12)):
    """
    Create a heatmap visualization of a performance matrix with Nash regret values.
    Nash regret values should be at most 0 in a proper Nash equilibrium.
    
    Parameters:
    - performance_matrix: pandas DataFrame containing the performance data
    - nash_regrets: Series or DataFrame containing Nash regret values for each agent
    - title: title for the plot
    - cmap: colormap to use
    - figsize: figure size (width, height)
    
    Returns:
    - fig: the matplotlib figure
    """
    numeric_matrix = performance_matrix.copy()
    
    for col in numeric_matrix.columns:
        numeric_matrix[col] = pd.to_numeric(numeric_matrix[col], errors='coerce')
    
    mask = np.isnan(numeric_matrix)
    
    fig = plt.figure(figsize=figsize)
    gs = plt.GridSpec(1, 5, figure=fig, width_ratios=[4, 1, 0.2, 0.2, 0.2])
    
    ax_heatmap = fig.add_subplot(gs[0, 0])
    
    ax_regret = fig.add_subplot(gs[0, 1])
    
    sns.heatmap(numeric_matrix, ax=ax_heatmap, cmap=cmap, annot=True, fmt=".2f",
                cbar=False, linewidths=0.5, mask=mask)
    
    ax_heatmap.set_title(title, fontsize=16, fontweight='bold')
    ax_heatmap.set_xticklabels(ax_heatmap.get_xticklabels(), rotation=45, ha='right', fontweight='bold')
    ax_heatmap.set_yticklabels(ax_heatmap.get_yticklabels(), fontweight='bold')
    
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
    
    regrets = regrets.dropna()
    
    # Ensure all regrets are non-positive (using a small tolerance)
    '''
    epsilon = 1e-4
    if (regrets > epsilon).any():
        print(f"Warning: Some regrets are positive: {regrets[regrets > epsilon].values}")
        print("Setting these regrets to 0 for visualization correctness.")
        regrets[regrets > epsilon] = 0.0
    '''
    # Sort regrets by value (in Nash equilibrium, less negative is better)
    regrets = regrets.sort_values(ascending=False)
    
    if len(regrets) > 0:
        # Use a colormap that shows close-to-zero regrets as green and more negative regrets as red
        # Since all regrets should be ≤ 0, we'll create a diverging colormap focused on negative values
        norm = plt.Normalize(regrets.min(), 0)  # 0 is the upper bound for regrets
        colors = plt.cm.RdYlGn(norm(regrets.values))  # Red (bad) to Yellow to Green (good)
        
        bars = ax_regret.barh(range(len(regrets)), regrets.values, color=colors)
        
        for i, value in enumerate(regrets.values):
            # Add text to show the value
            # For Nash regrets, closer to 0 is better (less exploitability)
            text_color = 'black'
            ax_regret.text(value - (abs(regrets.min()) * 0.02), i, f"{value:.4f}", 
                         va='center', fontsize=9, color=text_color)
        
        ax_regret.set_yticks(range(len(regrets)))
        ax_regret.set_yticklabels(regrets.index, fontweight='bold')
        
        ax_regret.set_xlabel('Nash Equilibrium Regret\n(closer to 0 is better)', fontweight='bold')
        ax_regret.set_title('NE Regret\n(should be ≤ 0)', fontweight='bold')
        
        # Add a horizontal line at 0
        ax_regret.axvline(x=0, color='black', linestyle='--', alpha=0.7)
        
        # Set x-axis limits to show all bars clearly
        ax_regret.set_xlim(min(regrets.min() * 1.1, -0.001), 0.001)  # Small buffer around 0
        
        ax_regret.spines['top'].set_visible(False)
        ax_regret.spines['right'].set_visible(False)
    else:
        ax_regret.text(0.5, 0.5, "No valid Nash regret data", 
                      ha='center', va='center', fontsize=12)
        ax_regret.axis('off')
    
    plt.tight_layout()
    return fig

def create_matrix_with_relative_performance(performance_matrix, bootstrap_stats, 
                                          title="Performance Matrix with Relative Performance", 
                                          cmap="coolwarm", figsize=(16, 12)):
    """
    Create a heatmap visualization of performance matrix with relative performance metric
    (expected_utility - nash_value)
    
    Parameters:
    - performance_matrix: pandas DataFrame containing the performance data
    - bootstrap_stats: DataFrame with regret statistics
    - title: title for the plot
    - cmap: colormap to use
    - figsize: figure size (width, height)
    
    Returns:
    - fig: the matplotlib figure
    """
    numeric_matrix = performance_matrix.copy()
    
    for col in numeric_matrix.columns:
        numeric_matrix[col] = pd.to_numeric(numeric_matrix[col], errors='coerce')
    
    mask = np.isnan(numeric_matrix)
    
    fig = plt.figure(figsize=figsize)
    gs = plt.GridSpec(1, 5, figure=fig, width_ratios=[4, 1, 0.2, 0.2, 0.2])
    
    ax_heatmap = fig.add_subplot(gs[0, 0])
    ax_regret = fig.add_subplot(gs[0, 1])
    
    # Heatmap of performance matrix
    sns.heatmap(numeric_matrix, ax=ax_heatmap, cmap=cmap, annot=True, fmt=".2f",
                cbar=False, linewidths=0.5, mask=mask)
    
    ax_heatmap.set_title(title, fontsize=16, fontweight='bold')
    ax_heatmap.set_xticklabels(ax_heatmap.get_xticklabels(), rotation=45, ha='right', fontweight='bold')
    ax_heatmap.set_yticklabels(ax_heatmap.get_yticklabels(), fontweight='bold')
    
    # Get relative performance values
    relative_regrets = pd.Series(index=numeric_matrix.index)
    
    if isinstance(bootstrap_stats, pd.DataFrame):
        if 'Mean NE Regret' in bootstrap_stats.columns and 'Agent' in bootstrap_stats.columns:
            regrets_dict = dict(zip(bootstrap_stats['Agent'], bootstrap_stats['Mean NE Regret']))
            for agent in numeric_matrix.index:
                if agent in regrets_dict:
                    relative_regrets[agent] = regrets_dict[agent]
                else:
                    relative_regrets[agent] = np.nan
    
    relative_regrets = relative_regrets.dropna()
    relative_regrets = relative_regrets.sort_values(ascending=False)  # Higher is better
    
    # Plot relative performance (higher is better)
    if len(relative_regrets) > 0:
        rel_colors = ['green' if val >= 0 else 'red' for val in relative_regrets.values]
        bars = ax_regret.barh(range(len(relative_regrets)), relative_regrets.values, color=rel_colors)
        
        for i, value in enumerate(relative_regrets.values):
            text_color = 'darkgreen' if value >= 0 else 'darkred'
            ax_regret.text(value + np.sign(value) * (max(abs(min(relative_regrets.values)), 
                                                         abs(max(relative_regrets.values))) * 0.02), 
                          i, f"{value:.2f}", va='center', fontsize=9, color=text_color)
        
        ax_regret.set_yticks(range(len(relative_regrets)))
        ax_regret.set_yticklabels(relative_regrets.index, fontweight='bold')
        
        ax_regret.set_xlabel('Relative Performance\n(expected_utility - nash_value)', fontweight='bold')
        ax_regret.set_title('Relative Performance\n(Higher is Better)', fontweight='bold')
        
        ax_regret.axvline(x=0, color='r', linestyle='--', alpha=0.7)
        ax_regret.spines['top'].set_visible(False)
        ax_regret.spines['right'].set_visible(False)
    else:
        ax_regret.text(0.5, 0.5, "No valid relative performance data", 
                      ha='center', va='center', fontsize=12)
        ax_regret.axis('off')
    
    plt.tight_layout()
    return fig

def create_dual_regret_visualization(performance_matrix, bootstrap_stats, title="Nash Regret Analysis", 
                                   figsize=(18, 14)):
    """
    Create a visualization with both Nash equilibrium regret and relative performance.
    
    Parameters:
    - performance_matrix: pandas DataFrame containing the performance data
    - bootstrap_stats: DataFrame with bootstrap statistics
    - title: title for the plot
    - figsize: figure size (width, height)
    
    Returns:
    - fig: the matplotlib figure
    """
    numeric_matrix = performance_matrix.copy()
    
    for col in numeric_matrix.columns:
        numeric_matrix[col] = pd.to_numeric(numeric_matrix[col], errors='coerce')
    
    mask = np.isnan(numeric_matrix)
    
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 3, height_ratios=[2, 1], width_ratios=[3, 1, 1])
    
    # Main heatmap
    ax_matrix = plt.subplot(gs[0, 0])
    
    # Nash regret subplot
    ax_nash = plt.subplot(gs[0, 1])
    
    # Scatter plot at bottom
    ax_scatter = plt.subplot(gs[1, :])
    
    # Draw the main heatmap
    sns.heatmap(numeric_matrix, ax=ax_matrix, cmap="YlGnBu", annot=True, fmt=".2f",
                cbar=True, linewidths=0.5, mask=mask, 
                cbar_kws={"label": "Performance Score", "shrink": 0.8})
    
    ax_matrix.set_title("Performance Matrix", fontsize=16, fontweight='bold')
    ax_matrix.set_xticklabels(ax_matrix.get_xticklabels(), rotation=45, ha='right', fontweight='bold')
    ax_matrix.set_yticklabels(ax_matrix.get_yticklabels(), fontweight='bold')
    
    # Prepare data
    ne_regrets = pd.Series(index=numeric_matrix.index)
    
    # Get Nash regret if available
    if 'Mean NE Regret' in bootstrap_stats.columns:
        ne_dict = dict(zip(bootstrap_stats['Agent'], bootstrap_stats['Mean NE Regret']))
        for agent in numeric_matrix.index:
            if agent in ne_dict:
                ne_regrets[agent] = ne_dict[agent]
            else:
                ne_regrets[agent] = np.nan
    
    # Sort agents by Nash regret
    ne_regrets = ne_regrets.dropna()
    ne_regrets = ne_regrets.sort_values()  # Lower is better for NE regret
    
    # Plot Nash regret (lower is better)
    if len(ne_regrets) > 0:
        # Normalize regrets for colormap
        norm = plt.Normalize(ne_regrets.min(), ne_regrets.max())
        colors = plt.cm.viridis_r(norm(ne_regrets.values))  # Reversed so lower is better
        
        bars = ax_nash.barh(range(len(ne_regrets)), ne_regrets.values, color=colors)
        
        for i, value in enumerate(ne_regrets.values):
            ax_nash.text(value + (ne_regrets.max() * 0.02), i, f"{value:.2f}", 
                        va='center', fontsize=9)
        
        ax_nash.set_yticks(range(len(ne_regrets)))
        ax_nash.set_yticklabels(ne_regrets.index, fontweight='bold')
        
        ax_nash.set_xlabel('Nash Equilibrium Regret', fontweight='bold')
        ax_nash.set_title('NE Regret\n(Lower is Better)', fontweight='bold')
        
        ax_nash.spines['top'].set_visible(False)
        ax_nash.spines['right'].set_visible(False)
    else:
        ax_nash.text(0.5, 0.5, "No valid Nash regret data", 
                    ha='center', va='center', fontsize=12)
        ax_nash.axis('off')
    
    # Prepare scatter plot data
    scatter_data = []
    for agent in numeric_matrix.index:
        if agent in ne_regrets:
            ne_regret = ne_regrets[agent]
            
            # Calculate mean performance as average of row
            mean_performance = numeric_matrix.loc[agent].mean()
            
            scatter_data.append({
                'Agent': agent,
                'NE Regret': ne_regret,
                'Mean Performance': mean_performance
            })
    
    if scatter_data:
        scatter_df = pd.DataFrame(scatter_data)
        
        # Create scatter plot
        sns.scatterplot(
            x='NE Regret', 
            y='Mean Performance', 
            data=scatter_df,
            s=100,
            alpha=0.7,
            ax=ax_scatter
        )
        
        # Add agent labels to points
        for i, row in scatter_df.iterrows():
            ax_scatter.annotate(
                row['Agent'],
                (row['NE Regret'], row['Mean Performance']),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=10
            )
        
        ax_scatter.set_title('NE Regret vs. Mean Performance', fontsize=14, fontweight='bold')
        ax_scatter.set_xlabel('Nash Equilibrium Regret (Lower is Better)', fontweight='bold')
        ax_scatter.set_ylabel('Mean Performance (Higher is Better)', fontweight='bold')
        
        # Add grid
        ax_scatter.grid(True, alpha=0.3)
    else:
        ax_scatter.text(0.5, 0.5, "Insufficient data for scatter plot", 
                      ha='center', va='center', fontsize=12)
        ax_scatter.axis('off')
    
    plt.tight_layout()
    return fig 