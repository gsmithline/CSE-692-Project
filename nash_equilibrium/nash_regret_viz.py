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
    
    regrets = regrets.sort_values(ascending=True)
    
    if len(regrets) > 0:
        colors = plt.cm.viridis(np.linspace(0, 1, len(regrets)))
        bars = ax_regret.barh(range(len(regrets)), regrets.values, color=colors)
        
        for i, value in enumerate(regrets.values):
            ax_regret.text(value + (max(regrets.values) * 0.02), i, f"{value:.2f}", 
                          va='center', fontsize=9)
        
        ax_regret.set_yticks(range(len(regrets)))
        ax_regret.set_yticklabels(regrets.index, fontweight='bold')
        
        ax_regret.set_xlabel('Nash Equilibrium Regret', fontweight='bold')
        ax_regret.set_title('NE Regret', fontweight='bold')
        
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

def create_matrix_with_traditional_regret(performance_matrix, bootstrap_stats, title="Performance Matrix with Traditional Regret", figsize=(12, 10)):
    """
    Create a heatmap of the performance matrix with traditional regret
    
    Args:
        performance_matrix: DataFrame with performance data
        bootstrap_stats: DataFrame with bootstrap statistics
        title: Plot title
        figsize: Figure size
        
    Returns:
        matplotlib figure
    """
    # Sort agents by traditional regret (ascending - lower is better)
    sorted_agents = bootstrap_stats.sort_values("Mean Traditional Regret")["Agent"].tolist()
    
    # Reindex the performance matrix
    sorted_matrix = performance_matrix.loc[sorted_agents, sorted_agents]
    
    # Create the figure
    fig = plt.figure(figsize=figsize)
    
    # Add regret bar on the right
    gs = gridspec.GridSpec(1, 2, width_ratios=[4, 1])
    
    # Matrix heatmap
    ax1 = plt.subplot(gs[0])
    matrix_heatmap = sns.heatmap(
        sorted_matrix, 
        annot=True, 
        fmt=".2f", 
        cmap="YlGnBu",
        cbar_kws={"label": "Performance"}, 
        ax=ax1
    )
    ax1.set_title("Performance Matrix\n(Agents ordered by Traditional Regret)", fontsize=14)
    
    # Regret bar chart
    ax2 = plt.subplot(gs[1])
    
    # Extract regret values for sorted agents
    regret_values = bootstrap_stats.set_index("Agent").loc[sorted_agents, "Mean Traditional Regret"]
    
    regret_bars = ax2.barh(
        range(len(sorted_agents)), 
        regret_values,
        color='skyblue',
        alpha=0.8
    )
    
    ax2.set_yticks(range(len(sorted_agents)))
    ax2.set_yticklabels([])  # Hide agent names since they're shown in the matrix
    ax2.set_xlabel("Traditional Regret\n(Lower is Better)", fontsize=12)
    ax2.set_title("Traditional Regret", fontsize=14)
    
    # Add value labels
    for i, bar in enumerate(regret_bars):
        value = regret_values.iloc[i]
        ax2.text(value + 0.01, i, f'{value:.3f}', va='center', fontsize=9)
    
    plt.suptitle(title, fontsize=16, y=1.02)
    plt.tight_layout()
    
    return fig

def create_matrix_with_relative_performance(performance_matrix, bootstrap_stats, title="Performance Matrix with Relative Performance", figsize=(12, 10)):
    """
    Create a heatmap of the performance matrix with relative performance
    
    Args:
        performance_matrix: DataFrame with performance data
        bootstrap_stats: DataFrame with bootstrap statistics
        title: Plot title
        figsize: Figure size
        
    Returns:
        matplotlib figure
    """
    # Sort agents by relative performance (descending - higher is better)
    sorted_agents = bootstrap_stats.sort_values("Mean NE Regret", ascending=False)["Agent"].tolist()
    
    # Reindex the performance matrix
    sorted_matrix = performance_matrix.loc[sorted_agents, sorted_agents]
    
    # Create the figure
    fig = plt.figure(figsize=figsize)
    
    # Add regret bar on the right
    gs = gridspec.GridSpec(1, 2, width_ratios=[4, 1])
    
    # Matrix heatmap
    ax1 = plt.subplot(gs[0])
    matrix_heatmap = sns.heatmap(
        sorted_matrix, 
        annot=True, 
        fmt=".2f", 
        cmap="YlGnBu",
        cbar_kws={"label": "Performance"}, 
        ax=ax1
    )
    ax1.set_title("Performance Matrix\n(Agents ordered by Relative Performance)", fontsize=14)
    
    # Regret bar chart
    ax2 = plt.subplot(gs[1])
    
    # Extract regret values for sorted agents
    regret_values = bootstrap_stats.set_index("Agent").loc[sorted_agents, "Mean NE Regret"]
    
    regret_bars = ax2.barh(
        range(len(sorted_agents)), 
        regret_values,
        color=['green' if x >= 0 else 'red' for x in regret_values],
        alpha=0.8
    )
    
    ax2.set_yticks(range(len(sorted_agents)))
    ax2.set_yticklabels([])  # Hide agent names since they're shown in the matrix
    ax2.set_xlabel("Relative Performance\n(Higher is Better)", fontsize=12)
    ax2.set_title("Relative Performance", fontsize=14)
    ax2.axvline(x=0, color='black', linestyle='--', alpha=0.7)
    
    # Add value labels
    for i, bar in enumerate(regret_bars):
        value = regret_values.iloc[i]
        text_offset = np.sign(value) * 0.01
        ax2.text(value + text_offset, i, f'{value:.3f}', va='center', fontsize=9, 
                color='darkgreen' if value >= 0 else 'darkred')
    
    plt.suptitle(title, fontsize=16, y=1.02)
    plt.tight_layout()
    
    return fig

def create_dual_regret_heatmap(performance_matrix, bootstrap_stats, title="Performance Matrix with Dual Regret Metrics", 
                              cmap="coolwarm", figsize=(20, 12)):
    """
    Create a heatmap visualization of a performance matrix with both traditional regret and 
    relative performance (NE regret) metrics.
    
    Parameters:
    - performance_matrix: pandas DataFrame containing the performance data
    - bootstrap_stats: DataFrame containing both regret metrics for each agent
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
    gs = plt.GridSpec(1, 6, figure=fig, width_ratios=[4, 1, 1, 0.2, 0.2, 0.2])
    
    ax_heatmap = fig.add_subplot(gs[0, 0])
    ax_relative = fig.add_subplot(gs[0, 1])
    ax_traditional = fig.add_subplot(gs[0, 2])
    
    # Heatmap of performance matrix
    sns.heatmap(numeric_matrix, ax=ax_heatmap, cmap=cmap, annot=True, fmt=".2f",
                cbar=False, linewidths=0.5, mask=mask)
    
    ax_heatmap.set_title(title, fontsize=16, fontweight='bold')
    ax_heatmap.set_xticklabels(ax_heatmap.get_xticklabels(), rotation=45, ha='right', fontweight='bold')
    ax_heatmap.set_yticklabels(ax_heatmap.get_yticklabels(), fontweight='bold')
    
    # Process bootstrap_stats to get both regret metrics
    relative_regrets = pd.Series(index=numeric_matrix.index)
    traditional_regrets = pd.Series(index=numeric_matrix.index)
    
    if isinstance(bootstrap_stats, pd.DataFrame):
        if 'Mean NE Regret' in bootstrap_stats.columns and 'Agent' in bootstrap_stats.columns:
            # Get relative regret (NE regret)
            regrets_dict = dict(zip(bootstrap_stats['Agent'], bootstrap_stats['Mean NE Regret']))
            for agent in numeric_matrix.index:
                if agent in regrets_dict:
                    relative_regrets[agent] = regrets_dict[agent]
                else:
                    relative_regrets[agent] = np.nan
            
            # Get traditional regret if available
            if 'Mean Traditional Regret' in bootstrap_stats.columns:
                trad_dict = dict(zip(bootstrap_stats['Agent'], bootstrap_stats['Mean Traditional Regret']))
                for agent in numeric_matrix.index:
                    if agent in trad_dict:
                        traditional_regrets[agent] = trad_dict[agent]
                    else:
                        traditional_regrets[agent] = np.nan
    
    relative_regrets = relative_regrets.dropna()
    traditional_regrets = traditional_regrets.dropna()
    
    # Sort both regret series
    relative_regrets = relative_regrets.sort_values(ascending=False)  # Higher is better
    traditional_regrets = traditional_regrets.sort_values(ascending=True)  # Lower is better
    
    # Plot relative performance (higher is better)
    if len(relative_regrets) > 0:
        rel_colors = ['green' if val >= 0 else 'red' for val in relative_regrets.values]
        bars1 = ax_relative.barh(range(len(relative_regrets)), relative_regrets.values, color=rel_colors)
        
        for i, value in enumerate(relative_regrets.values):
            text_color = 'darkgreen' if value >= 0 else 'darkred'
            ax_relative.text(value + np.sign(value) * (max(abs(min(relative_regrets.values)), 
                                                          abs(max(relative_regrets.values))) * 0.02), 
                           i, f"{value:.2f}", va='center', fontsize=9, color=text_color)
        
        ax_relative.set_yticks(range(len(relative_regrets)))
        ax_relative.set_yticklabels(relative_regrets.index, fontweight='bold')
        
        ax_relative.set_xlabel('Relative Performance\n(expected_utility - nash_value)', fontweight='bold')
        ax_relative.set_title('Relative Performance\n(Higher is Better)', fontweight='bold')
        
        ax_relative.axvline(x=0, color='r', linestyle='--', alpha=0.7)
        ax_relative.spines['top'].set_visible(False)
        ax_relative.spines['right'].set_visible(False)
    else:
        ax_relative.text(0.5, 0.5, "No valid relative performance data", 
                      ha='center', va='center', fontsize=12)
        ax_relative.axis('off')
    
    # Plot traditional regret (lower is better)
    if len(traditional_regrets) > 0:
        bars2 = ax_traditional.barh(range(len(traditional_regrets)), traditional_regrets.values, color='blue')
        
        for i, value in enumerate(traditional_regrets.values):
            ax_traditional.text(value + (max(traditional_regrets.values) * 0.02), 
                              i, f"{value:.2f}", va='center', fontsize=9)
        
        ax_traditional.set_yticks(range(len(traditional_regrets)))
        ax_traditional.set_yticklabels(traditional_regrets.index, fontweight='bold')
        
        ax_traditional.set_xlabel('Traditional Regret\n(max_utility - expected_utility)', fontweight='bold')
        ax_traditional.set_title('Traditional Regret\n(Lower is Better)', fontweight='bold')
        
        ax_traditional.spines['top'].set_visible(False)
        ax_traditional.spines['right'].set_visible(False)
    else:
        ax_traditional.text(0.5, 0.5, "No valid traditional regret data", 
                         ha='center', va='center', fontsize=12)
        ax_traditional.axis('off')
    
    plt.tight_layout()
    return fig 