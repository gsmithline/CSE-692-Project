import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from matplotlib.colors import Normalize

EPSILON = 1e-8  

def create_matrix_heatmap_with_nash_regret(performance_matrix, bootstrap_stats=None, title=None, figsize=(12, 9)):
    """
    Create a heatmap visualization of a performance matrix with Nash regret values.
    
    Args:
        performance_matrix: DataFrame with performance matrix data
        bootstrap_stats: DataFrame with bootstrap statistics
        title: Optional title for the plot
        figsize: Figure size tuple (width, height)
        
    Returns:
        matplotlib.figure.Figure: The generated plot
    """
    # Create figure with two subplots - heatmap and regret bars
    fig, (ax_heatmap, ax_regret) = plt.subplots(1, 2, figsize=figsize, 
                                                 gridspec_kw={'width_ratios': [3, 1]})
    
    # Ensure performance_matrix is numeric for the heatmap
    numeric_matrix = performance_matrix.apply(pd.to_numeric, errors='coerce')
    
    # First, create the heatmap
    sns.heatmap(numeric_matrix, annot=True, cmap="YlGnBu", ax=ax_heatmap,
                fmt=".1f", cbar=True, cbar_kws={"shrink": 0.5})
    
    if title:
        ax_heatmap.set_title(title)
    else:
        ax_heatmap.set_title("Performance Matrix")
    
    ax_heatmap.set_xlabel("Column Player (Opponent)")
    ax_heatmap.set_ylabel("Row Player (Self)")
    
    # Second subplot - Nash regret bars
    if bootstrap_stats is not None:
        # Extract regret values and order them for plotting
        regrets = bootstrap_stats[['Agent', 'Mean NE Regret']].copy()
        
        # Drop NaN values but maintain original order from performance matrix
        regrets = regrets.dropna()
        
        # For Nash Equilibrium regrets, inform about the range
        if 'Mean NE Regret' in regrets.columns:
            epsilon = 1e-8  # Numerical tolerance
            if (regrets['Mean NE Regret'] > epsilon).any():
                print(f"Warning: Some NE regrets are positive: {regrets[regrets['Mean NE Regret'] > epsilon]['Mean NE Regret'].values}")
                print("These positive regrets indicate the Nash equilibrium calculation may not have fully converged.")
            
            # For NE regrets, less negative is better (closer to 0)
            regrets = regrets.sort_values('Mean NE Regret', ascending=False)
        else:
            # For any other metric, use standard sorting
            regrets = regrets.sort_values('Mean NE Regret', ascending=True)
            
        # Use a colormap that shows closer to 0 as better (green)
        min_val = regrets['Mean NE Regret'].min()
        max_val = regrets['Mean NE Regret'].max()  # Don't cap at 0, show actual values
        
        # Ensure regret values are properly converted to float
        try:
            # Try to convert the column to float
            regrets['Mean NE Regret'] = regrets['Mean NE Regret'].astype(float)
        except (ValueError, TypeError) as e:
            print(f"Warning: Error converting regret values to float: {e}")
            # Try a more robust conversion method
            regrets['Mean NE Regret'] = pd.to_numeric(regrets['Mean NE Regret'], errors='coerce')
            # Remove any rows with NaN values after conversion
            regrets = regrets.dropna(subset=['Mean NE Regret'])
            if regrets.empty:
                print("Error: No valid regret values after conversion.")
                return fig
            # Recalculate min and max after cleaning
            min_val = regrets['Mean NE Regret'].min()
            max_val = regrets['Mean NE Regret'].max()
        
        # Create a diverging colormap centered at 0
        norm = plt.Normalize(min_val, max_val) 
        try:
            if min_val < 0 and max_val > 0:
                # Use diverging colormap for mixed positive/negative values
                regret_values = regrets['Mean NE Regret'].values
                colors = plt.cm.RdYlGn_r(norm(regret_values))
            elif max_val <= 0:
                # All regrets are negative, use standard colormap
                regret_values = regrets['Mean NE Regret'].values
                colors = plt.cm.RdYlGn(norm(regret_values))
            else:
                # All regrets are positive
                regret_values = regrets['Mean NE Regret'].values
                colors = plt.cm.RdYlGn_r(norm(regret_values))
        except Exception as e:
            print(f"Warning: Error creating color mapping: {e}")
            # Use a simple default color as fallback
            colors = ['steelblue'] * len(regrets)
        
        # Create horizontal bar chart
        y_pos = np.arange(len(regrets))
        
        for i, (_, row) in enumerate(regrets.iterrows()):
            value = row['Mean NE Regret']
            try:
                # Safely create the horizontal bar
                if i < len(colors):
                    ax_regret.barh(y_pos[i], value, color=colors[i], height=0.8)
                else:
                    # Use a default color if index is out of range
                    ax_regret.barh(y_pos[i], value, color='steelblue', height=0.8)
            except Exception as e:
                print(f"Warning: Error creating bar for agent {row['Agent']}: {e}")
                # Try with a simpler approach
                ax_regret.barh(y_pos[i], value, color='steelblue', height=0.8)
        
        # Add value labels
        for i, (_, row) in enumerate(regrets.iterrows()):
            try:
                value = float(row['Mean NE Regret'])
                # Position the text slightly to the right of the bar for negative values,
                # and slightly to the left for positive values
                text_x = value + 0.1 * (min_val if value <= 0 else -value) 
                ax_regret.text(text_x, y_pos[i], f"{value:.2f}", 
                             va='center', ha='left' if value <= 0 else 'right',
                             fontsize=9, color='black')
            except Exception as e:
                print(f"Warning: Error adding label for agent {row['Agent']}: {e}")
        
        # Set y-ticks to agent names
        ax_regret.set_yticks(y_pos)
        ax_regret.set_yticklabels(regrets['Agent'])
        
        # Set axis labels and title
        ax_regret.set_xlabel("Nash Equilibrium Regret")
        ax_regret.set_title("Nash Equilibrium Regrets")
        
        # Add a vertical line at x=0 to indicate zero regret
        ax_regret.axvline(x=0, color='black', linestyle='--', alpha=0.7)
        
        # Set x-axis limit slightly beyond the min and max regret
        buffer_min = abs(min_val) * 0.15  # Add 15% buffer on min side
        buffer_max = abs(max_val) * 0.15  # Add 15% buffer on max side
        ax_regret.set_xlim(min_val - buffer_min, max_val + buffer_max)
    
    plt.tight_layout()
    return fig

def create_histogram_with_regret_values(bootstrap_stats, full_range=True, figsize=(12, 9)):
    """
    Create a set of histograms showing the distribution of regret values for each agent.
    
    Args:
        bootstrap_stats: DataFrame with bootstrap statistics
        full_range: Whether to use the full range of regret values or zoom in
        figsize: Figure size tuple (width, height)
        
    Returns:
        matplotlib.figure.Figure: The generated plot
    """
    if bootstrap_stats is None or len(bootstrap_stats) == 0:
        print("No bootstrap statistics available for histogram.")
        return None
    
    # Get the agents and their regret statistics
    agents = bootstrap_stats['Agent'].tolist()
    n_agents = len(agents)
    
    # Calculate grid size for subplots
    n_cols = min(3, n_agents)
    n_rows = (n_agents + n_cols - 1) // n_cols  # Ceiling division
    
    fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    # Handle single subplot case
    if n_agents == 1:
        axs = np.array([axs])
    
    # Flatten axs for easier indexing
    if n_rows > 1 and n_cols > 1:
        axs = axs.flatten()
    
    # Find global min and max for consistent scaling
    global_min = bootstrap_stats['Lower 95% CI (NE Regret)'].min()
    global_max = bootstrap_stats['Upper 95% CI (NE Regret)'].max()
    
    # Create histograms for each agent
    for i, agent in enumerate(agents):
        # Skip if we have more axes than agents
        if i >= len(axs):
            break
            
        agent_row = bootstrap_stats[bootstrap_stats['Agent'] == agent]
        
        # Get regret statistics
        mean_regret = agent_row['Mean NE Regret'].values[0]
        lower_ci = agent_row['Lower 95% CI (NE Regret)'].values[0]
        upper_ci = agent_row['Upper 95% CI (NE Regret)'].values[0]
        
        # For histograms, we need to simulate the distribution based on stats
        # Here we'll create a normal distribution with the given mean and std
        std_regret = agent_row['Std NE Regret'].values[0]
        
        # Create normal distribution sample (this is just for visualization)
        agent_regrets = np.random.normal(mean_regret, std_regret, 1000)
        
        # Find range for plotting
        min_regret = min(global_min, lower_ci)
        max_regret = max(global_max, upper_ci)
        
        # Create bins that cover the full range
        bins = np.linspace(
            min_regret - abs(min_regret)*0.01, 
            max_regret + abs(max_regret)*0.01,
            30
        )
        
        # Plot histogram
        axs[i].hist(agent_regrets, bins=bins, alpha=0.7, color='darkgreen')
        
        # Add vertical lines for mean and CI
        axs[i].axvline(mean_regret, color='red', linestyle='-', label='Mean')
        axs[i].axvline(lower_ci, color='orange', linestyle='--', label='95% CI')
        axs[i].axvline(upper_ci, color='orange', linestyle='--')
        
        # Add a line at zero for reference
        axs[i].axvline(0, color='black', linestyle='-', alpha=0.3)
        
        # Set title and labels
        axs[i].set_title(f"{agent}")
        axs[i].set_xlabel("Nash Equilibrium Regret")
        axs[i].set_ylabel("Frequency")
        
        # Set consistent x-axis limits if using full range
        if full_range:
            # Add some padding to the limits
            left_buffer = abs(min_regret) * 0.05
            right_buffer = abs(max_regret) * 0.1  # Add 10% buffer on right side
            axs[i].set_xlim(min_regret - left_buffer, max_regret + right_buffer)
        else:
            # Zoom in to around the mean +/- 3*std to focus on distribution
            zoom_range = 3 * std_regret
            axs[i].set_xlim(mean_regret - zoom_range, mean_regret + zoom_range)
        
        # Add legend to first plot only
        if i == 0:
            axs[i].legend()
            
        # Add annotations with statistics
        axs[i].text(0.5, 0.89, f"Mean: {mean_regret:.2f}",
                  transform=axs[i].transAxes, ha='center', bbox=dict(facecolor='white', alpha=0.8))
        axs[i].text(0.5, 0.82, f"95% CI: [{lower_ci:.2f}, {upper_ci:.2f}]",
                  transform=axs[i].transAxes, ha='center', bbox=dict(facecolor='white', alpha=0.8))
        axs[i].text(0.5, 0.75, f"Full range: [{min_regret:.2f}, {max_regret:.2f}]",
                  transform=axs[i].transAxes, ha='center', bbox=dict(facecolor='white', alpha=0.8))
    
    # Hide unused subplots
    for i in range(n_agents, len(axs)):
        axs[i].axis('off')
    
    plt.tight_layout()
    return fig

def visualize_positive_regrets(bootstrap_stats, threshold=1e-5, figsize=(12, 6)):
    """
    Visualize the frequency of positive regrets in bootstrap samples.
    
    Args:
        bootstrap_stats: DataFrame with bootstrap statistics
        threshold: Threshold for considering regrets as positive
        figsize: Figure size tuple (width, height)
        
    Returns:
        matplotlib.figure.Figure: The generated plot
    """
    if 'Upper 95% CI (NE Regret)' not in bootstrap_stats.columns:
        print("No regret confidence intervals found in bootstrap stats.")
        return None
    
    # Identify agents with positive regrets
    positive_regrets = bootstrap_stats['Upper 95% CI (NE Regret)'] > threshold
    
    if not positive_regrets.any():
        print("No agents have positive regret upper bounds.")
        return None
    
    # Count positive regrets per agent
    agent_violation_counts = np.sum(positive_regrets, axis=0)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot bar chart of violations
    agents_with_violations = bootstrap_stats.loc[positive_regrets, 'Agent']
    violation_values = bootstrap_stats.loc[positive_regrets, 'Upper 95% CI (NE Regret)']
    
    bars = ax.bar(agents_with_violations, violation_values)
    
    # Add value labels
    for bar, value in zip(bars, violation_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.4f}', ha='center', va='bottom')
    
    ax.set_ylabel('Upper 95% CI Regret Value')
    ax.set_title('Agents with Positive Regret Upper Bounds')
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    return fig

def create_overlapping_histograms(bootstrap_results, agents, regret_type='ne_regret'):
    """
    Create overlapping histograms of regret values for multiple agents.
    
    Args:
        bootstrap_results: Dictionary with bootstrap results
        agents: List of agent names
        regret_type: Type of regret to plot ('ne_regret' or 'rd_regret')
        
    Returns:
        matplotlib.figure.Figure: The generated plot
    """
    try:
        if regret_type not in bootstrap_results:
            print(f"No {regret_type} data found in bootstrap results.")
            return None
        
        # Get regret values
        regrets = bootstrap_results[regret_type]
        
        # Convert to numpy array for easier manipulation
        regrets_array = np.array(regrets)
        
        # Create a figure with a subplot for each agent
        fig, axs = plt.subplots(len(agents), 1, figsize=(12, 3*len(agents)), sharex=True)
        
        # Handle single agent case
        if len(agents) == 1:
            axs = [axs]
        
        regret_label = "Max Entropy Nash Equilibrium" if regret_type == 'ne_regret' else "Replicator Dynamics Nash Equilibrium"
        
        for i, agent in enumerate(agents):
            ax = axs[i]
            
            # Extract regret values for this agent
            agent_regrets = regrets_array[:, i]
            
            # Get statistics
            mean_regret = np.mean(agent_regrets)
            std_regret = np.std(agent_regrets)
            lower_ci = np.percentile(agent_regrets, 2.5)
            upper_ci = np.percentile(agent_regrets, 97.5)
            min_regret = np.min(agent_regrets)
            max_regret = np.max(agent_regrets)
            
            # Create bins for histogram
            bins = np.linspace(min_regret - abs(min_regret)*0.01, 
                               max_regret + abs(max_regret)*0.01, 30)
            
            # Plot histogram
            ax.hist(agent_regrets, bins=bins, alpha=0.7, color='darkgreen')
            
            # Add vertical lines for mean and CI
            ax.axvline(mean_regret, color='red', linestyle='-', label='Mean')
            ax.axvline(lower_ci, color='orange', linestyle='--', label='95% CI')
            ax.axvline(upper_ci, color='orange', linestyle='--')
            
            # Add a line at zero for reference
            ax.axvline(0, color='black', linestyle='-', alpha=0.3)
            
            # Set title and labels
            ax.set_title(f"{agent} - {regret_label} Regret Distribution")
            ax.set_ylabel("Frequency")
            
            # Set x limits with a small buffer
            left_buffer = abs(min_regret) * 0.05
            right_buffer = abs(max_regret) * 0.1  # Add 10% buffer on right side
            ax.set_xlim(min_regret - left_buffer, max_regret + right_buffer)
            
            # Add legend to first plot only
            if i == 0:
                ax.legend()
                
            # Add annotations with statistics
            ax.text(0.05, 0.92, f"Mean: {mean_regret:.2f}", transform=ax.transAxes, 
                   bbox=dict(facecolor='white', alpha=0.8))
            ax.text(0.05, 0.85, f"95% CI: [{lower_ci:.2f}, {upper_ci:.2f}]", 
                   transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.8))
            ax.text(0.05, 0.78, f"Full range: [{min_regret:.2f}, {max_regret:.2f}]",
                   transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.8))
        
        # Add x-label to the bottom subplot only
        axs[-1].set_xlabel(f"{regret_label} Regret")
        
        plt.tight_layout()
        return fig
    
    except Exception as e:
        print(f"Error creating overlapping histograms: {e}")
        import traceback
        traceback.print_exc()
        return None

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
        # Limit the number of samples to prevent memory issues
        max_samples = 1000
        if len(regrets) > max_samples:
            print(f"Warning: Limiting regret samples from {len(regrets)} to {max_samples} to prevent memory issues")
            import random
            random.seed(42)  # For reproducibility
            regrets = random.sample(regrets, max_samples)
        
        # Convert to numpy array with proper error handling
        try:
            regrets = np.stack(regrets)
        except ValueError:
            print("Warning: Bootstrap samples have inconsistent shapes. Using a more flexible approach.")
            # Get the shape of the first element if possible
            if hasattr(regrets[0], 'shape'):
                first_shape = regrets[0].shape
            else:
                # Try to infer shape by length
                try:
                    first_shape = (len(regrets[0]),)
                except:
                    first_shape = (1,)
            
            # Create a properly sized array
            if len(first_shape) == 0:  # Handle scalar values
                regrets_array = np.zeros((len(regrets), 1), dtype=np.float64)
                for i, regret in enumerate(regrets):
                    try:
                        regrets_array[i, 0] = float(regret)
                    except:
                        pass
            else:
                regrets_array = np.zeros((len(regrets), first_shape[0]), dtype=np.float64)
                for i, regret in enumerate(regrets):
                    try:
                        if hasattr(regret, '__len__') and len(regret) == first_shape[0]:
                            regrets_array[i] = np.array(regret, dtype=np.float64)
                        else:
                            print(f"Warning: Skipping regret sample {i} due to shape mismatch")
                    except Exception as e:
                        print(f"Warning: Error processing regret sample {i}: {e}")
            
            regrets = regrets_array
    
        # Remove any NaN values and replace with zeros
        regrets = np.nan_to_num(regrets, nan=0.0)
        
        # MEMORY SAFEGUARD: Check array size before proceeding
        array_size_bytes = regrets.size * regrets.itemsize
        max_allowed_size = 1 * 1024 * 1024 * 1024  # 1 GB
        if array_size_bytes > max_allowed_size:
            print(f"Error: Regret array too large ({array_size_bytes/1024/1024/1024:.2f} GB). Reducing to prevent memory error.")
            # Further reduce samples if needed
            reduction_factor = int(np.ceil(array_size_bytes / max_allowed_size))
            sample_step = max(1, reduction_factor)
            regrets = regrets[::sample_step]
            print(f"Reduced to {len(regrets)} samples")
        
        n_agents = min(len(agent_names), regrets.shape[1])
        
        # Create a more memory-efficient figure by processing one agent at a time
        n_cols = min(3, n_agents)
        n_rows = int(np.ceil(n_agents/n_cols))
        fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize)
        
        # Handle single subplot case
        if n_rows == 1 and n_cols == 1:
            axs = np.array([[axs]])
        elif n_rows == 1 or n_cols == 1:
            axs = np.array([axs]).reshape(n_rows, n_cols)
        
        # Check for any positive regrets
        epsilon = 1e-6  # Small threshold for numerical precision
        positive_regrets = (regrets > epsilon)
        
        if np.any(positive_regrets):
            count_positive = np.sum(positive_regrets)
            total_regrets = regrets.size
            percent_positive = (count_positive / total_regrets) * 100
            
            # Count regrets above epsilon by agent
            agent_violation_counts = np.sum(positive_regrets, axis=0)
            if len(agent_violation_counts) > 0:
                worst_agent_idx = np.argmax(agent_violation_counts)
                worst_agent = agent_names[worst_agent_idx] if worst_agent_idx < len(agent_names) else f"Agent {worst_agent_idx}"
                samples_per_agent = regrets.shape[0]
                
                print(f"Warning: {count_positive}/{total_regrets} regret values ({percent_positive:.2f}%) are above epsilon={epsilon:.2e}.")
                print(f"         Worst agent: '{worst_agent}' with {agent_violation_counts[worst_agent_idx]}/{samples_per_agent} samples ({(agent_violation_counts[worst_agent_idx]/samples_per_agent)*100:.2f}%) showing positive regrets.")
        
        # Process one agent at a time to save memory
        for i in range(n_agents):
            row, col = i // n_cols, i % n_cols
            ax = axs[row, col]
            
            agent = agent_names[i] if i < len(agent_names) else f"Agent {i}"
            agent_regrets = regrets[:, i]
            
            # Calculate statistics
            try:
                min_regret = float(np.min(agent_regrets))
                max_regret = float(np.max(agent_regrets))
                mean_regret = float(np.mean(agent_regrets))
                lower_ci = float(np.percentile(agent_regrets, 2.5))
                upper_ci = float(np.percentile(agent_regrets, 97.5))
                
                # MEMORY SAFEGUARD: Use fewer bins and check before creating histogram
                num_bins = min(30, max(10, len(agent_regrets) // 100))
                
                # Create bin edges to cover the entire range of data - use simple approach
                bin_edges = np.linspace(
                    min_regret - abs(min_regret)*0.01, 
                    max_regret + abs(max_regret)*0.01, 
                    num_bins  # Use fewer bins to save memory
                )
                
                # Plot histogram with original (uncapped) regrets
                # Use numpy to pre-compute histogram before plotting to save memory
                try:
                    hist, _ = np.histogram(agent_regrets, bins=bin_edges)
                    ax.bar(bin_edges[:-1], hist, width=np.diff(bin_edges), alpha=0.7, color='darkgreen')
                except MemoryError:
                    # Fallback to simple plotting if histogram fails
                    ax.text(0.5, 0.5, f"Memory error plotting histogram for {agent}", 
                          transform=ax.transAxes, ha='center', va='center')
                
                ax.set_title(agent)
                ax.set_xlabel('Nash Equilibrium Regret')
                ax.set_ylabel('Frequency')
                
                # Add mean line
                ax.axvline(mean_regret, color='r', linestyle='--', 
                          label=f'Mean: {mean_regret:.2f}')
                
                # Add 95% CI
                ax.axvline(lower_ci, color='orange', linestyle=':')
                ax.axvline(upper_ci, color='orange', linestyle=':', 
                          label=f'95% CI: [{lower_ci:.2f}, {upper_ci:.2f}]')
                
                # Add a reference line at 0 with explicit label
                ax.axvline(0, color='black', linestyle='-', alpha=0.7, 
                          label='Zero regret (equilibrium)')
                
                # Add a text annotation with statistics instead of using up memory on detailed plots
                ax.text(0.5, 0.97, f"Mean: {mean_regret:.2f}, 95% CI: [{lower_ci:.2f}, {upper_ci:.2f}]", 
                      transform=ax.transAxes, ha='center', va='top',
                      bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'),
                      fontsize=8)
                
                ax.text(0.5, 0.9, f"Range: [{min_regret:.2f}, {max_regret:.2f}]", 
                      transform=ax.transAxes, ha='center', va='top',
                      bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'),
                      fontsize=8)
                
                # Move legend to bottom left to avoid blocking data
                ax.legend(fontsize='small', loc='lower left')
                
            except Exception as e:
                print(f"Error processing agent {agent}: {e}")
                ax.text(0.5, 0.5, f"Error plotting data for {agent}", 
                      transform=ax.transAxes, ha='center', va='center')
                ax.axis('on')
        
        # Hide any unused subplots
        for j in range(n_agents, n_rows * n_cols):
            row, col = j // n_cols, j % n_cols
            axs[row, col].axis('off')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)  # Make room for the suptitle
        
        return fig
    
    except MemoryError as e:
        print(f"Memory error creating distribution plot: {e}")
        # Create a simple error figure that uses minimal memory
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "Memory error: Unable to create detailed plot.\nConsider reducing bootstrap samples or agent count.", 
              transform=ax.transAxes, ha='center', va='center')
        ax.axis('off')
        return fig
        
    except Exception as e:
        print(f"Error creating regret distribution plot: {e}")
        # Create a simple error figure
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, f"Error creating plot: {str(e)[:200]}...", 
              transform=ax.transAxes, ha='center', va='center')
        ax.axis('off')
        return fig 