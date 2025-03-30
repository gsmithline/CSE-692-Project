"""
.
These methods help determine if more simulator data is needed by analyzing:
1. Monte Carlo error relative to the estimates follwoing: https://faculty.washington.edu/yenchic/17Sp_302/R11.pdf
2. Convergence of bootstrap statistics across samples 
3. Stability of confidence intervals
4. Distribution of bootstrap statistics
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from collections import defaultdict
import matplotlib.gridspec as gridspec
import warnings

def assess_simulator_data_needs(bootstrap_results, agent_names, output_dir=None, 
                               relative_error_threshold=0.05, ci_stability_threshold=0.2,
                               plot_diagnostics=True):
    """
    Assess whether more simulator data is needed based on bootstrap diagnostics.
    Implements the methodology from the bootstrap paper to determine if the current
    simulator data is sufficient for reliable conclusions.
    
    Args:
        bootstrap_results: Dictionary containing bootstrap samples
        agent_names: List of agent names
        output_dir: Directory to save diagnostic plots
        relative_error_threshold: Threshold for acceptable relative Monte Carlo error
        ci_stability_threshold: Threshold for acceptable CI stability
        plot_diagnostics: Whether to create diagnostic plots
        
    Returns:
        DataFrame with assessment results and recommendations
    """
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    if not bootstrap_results or not all(k in bootstrap_results for k in ['ne_regret', 'agent_expected_utility']):
        raise ValueError("Invalid bootstrap_results: Must contain 'ne_regret' and 'agent_expected_utility'")
    
    # Extract key statistics
    results = {}
    statistics = {}
    
    for stat_name, stat_key in [
        ('Nash Equilibrium Regret', 'ne_regret'),
        ('Expected Utility', 'agent_expected_utility'),
        ('RD Nash Regret', 'rd_regret')
    ]:
        if stat_key in bootstrap_results and bootstrap_results[stat_key]:
            try:
                # Convert to numpy array
                data_array = np.array(bootstrap_results[stat_key])
                
                # Basic statistics
                n_samples = len(data_array)
                final_means = np.mean(data_array, axis=0)
                final_stds = np.std(data_array, axis=0, ddof=1)
                
                # Monte Carlo errors (SE = std / sqrt(n))
                monte_carlo_errors = final_stds / np.sqrt(n_samples)
                
                # Relative errors (error / |mean|) - key metric from the paper
                # Use a small epsilon to avoid division by zero
                epsilon = 1e-10
                relative_errors = np.zeros_like(monte_carlo_errors)
                for i in range(len(monte_carlo_errors)):
                    if abs(final_means[i]) > epsilon:
                        relative_errors[i] = abs(monte_carlo_errors[i] / final_means[i])
                    else:
                        # When mean is very small, use std as reference
                        relative_errors[i] = monte_carlo_errors[i] / max(final_stds[i], epsilon)
                
                # Confidence intervals
                lower_ci = np.percentile(data_array, 2.5, axis=0)
                upper_ci = np.percentile(data_array, 97.5, axis=0)
                ci_width = upper_ci - lower_ci
                
                # CI relative width (width / |mean|) - another key metric
                ci_relative_width = np.zeros_like(ci_width)
                for i in range(len(ci_width)):
                    if abs(final_means[i]) > epsilon:
                        ci_relative_width[i] = ci_width[i] / abs(final_means[i])
                    else:
                        ci_relative_width[i] = ci_width[i] / max(final_stds[i], epsilon)
                
                # Calculate convergence diagnosis metrics
                convergence_metrics = calculate_convergence_metrics(data_array, agent_names)
                
                # Store results
                statistics[stat_name] = {
                    'data_array': data_array,
                    'means': final_means,
                    'stds': final_stds,
                    'monte_carlo_errors': monte_carlo_errors,
                    'relative_errors': relative_errors,
                    'ci_lower': lower_ci,
                    'ci_upper': upper_ci,
                    'ci_width': ci_width,
                    'ci_relative_width': ci_relative_width,
                    'convergence': convergence_metrics
                }
                
                # Create assessment summary
                for i, agent in enumerate(agent_names):
                    if i >= len(final_means):
                        continue
                        
                    if agent not in results:
                        results[agent] = {}
                    
                    # Store main metrics
                    results[agent][f'{stat_name} Mean'] = final_means[i]
                    results[agent][f'{stat_name} Std'] = final_stds[i]
                    results[agent][f'{stat_name} MCE'] = monte_carlo_errors[i]
                    results[agent][f'{stat_name} Relative Error'] = relative_errors[i]
                    results[agent][f'{stat_name} CI Width'] = ci_width[i]
                    results[agent][f'{stat_name} CI Relative Width'] = ci_relative_width[i]
                    
                    # Assess simulator data needs based on relative error
                    # This is a key metric from the paper
                    if relative_errors[i] < 0.01:
                        results[agent][f'{stat_name} Assessment'] = "Excellent - No more data needed"
                    elif relative_errors[i] < relative_error_threshold:
                        results[agent][f'{stat_name} Assessment'] = "Good - Sufficient data"
                    elif relative_errors[i] < 0.10:
                        results[agent][f'{stat_name} Assessment'] = "Fair - Consider more data"
                    else:
                        results[agent][f'{stat_name} Assessment'] = "Poor - More simulator data needed"
                    
                    # Store convergence metrics for this agent
                    if 'convergence' in convergence_metrics:
                        for metric, values in convergence_metrics['convergence'].items():
                            if i < len(values):
                                results[agent][f'{stat_name} {metric}'] = values[i]
            
            except Exception as e:
                print(f"Error processing {stat_name}: {e}")
    
    # Create overall assessment
    assessment_df = pd.DataFrame.from_dict(results, orient='index').reset_index()
    assessment_df.rename(columns={'index': 'Agent'}, inplace=True)
    
    # Add overall recommendation column
    assessment_df['Overall Recommendation'] = assessment_df.apply(
        lambda row: generate_overall_recommendation(row), axis=1)
    
    # Generate diagnostic plots if requested
    if plot_diagnostics and output_dir:
        for stat_name, stat_data in statistics.items():
            try:
                # Create diagnostic plots
                create_bootstrap_diagnostic_plots(
                    stat_data['data_array'], 
                    agent_names, 
                    stat_name, 
                    output_dir,
                    stat_data
                )
            except Exception as e:
                print(f"Error creating diagnostic plots for {stat_name}: {e}")
    
    return assessment_df

def calculate_convergence_metrics(data_array, agent_names):
    """
    Calculate various convergence metrics for bootstrap samples.
    These help determine if more simulator data is needed.
    
    Args:
        data_array: Array of bootstrap samples, shape (n_samples, n_agents)
        agent_names: List of agent names
        
    Returns:
        Dictionary of convergence metrics
    """
    try:
        n_samples = data_array.shape[0]
        n_agents = min(data_array.shape[1], len(agent_names))
        
        # Only calculate if we have enough samples
        if n_samples < 5:
            return {'error': 'Too few samples for convergence metrics'}
        
        # Calculate bootstrap iteration stability
        # From the paper: "Examine bootstrap iteration plots to see if statistics have stabilized"
        # We'll use a few windows to check consistency
        window_sizes = [max(1, n_samples // 10), max(1, n_samples // 5), max(1, n_samples // 3)]
        stability_metrics = defaultdict(list)
        
        for window_size in window_sizes:
            if window_size < 2:
                continue
                
            # Calculate statistics for final window vs full dataset
            window_means = np.mean(data_array[-window_size:], axis=0)
            full_means = np.mean(data_array, axis=0)
            
            # Calculate relative difference
            abs_diff = np.abs(window_means - full_means)
            epsilon = 1e-10  # Small epsilon to avoid division by zero
            rel_diff = np.array([
                abs_diff[i] / max(abs(full_means[i]), epsilon) 
                for i in range(len(full_means))
            ])
            
            # Store the relative differences
            stability_metrics[f'Window {window_size} Stability'].extend(rel_diff)
        
        # Calculate CI stability across increasing sample sizes
        # From the paper: "Review CI stability plots to see if confidence intervals have stabilized"
        ci_stability = []
        sample_fractions = [0.25, 0.5, 0.75, 1.0]
        prev_ci_width = None
        
        for fraction in sample_fractions:
            n_subset = max(10, int(n_samples * fraction))
            subset = data_array[:n_subset]
            
            lower_ci = np.percentile(subset, 2.5, axis=0)
            upper_ci = np.percentile(subset, 97.5, axis=0)
            ci_width = upper_ci - lower_ci
            
            if prev_ci_width is not None:
                # Calculate change in CI width
                ci_width_change = np.abs(ci_width - prev_ci_width)
                # Normalize by the previous width to get relative change
                ci_width_rel_change = np.array([
                    ci_width_change[i] / max(prev_ci_width[i], epsilon) 
                    for i in range(len(ci_width))
                ])
                ci_stability.append(ci_width_rel_change)
            
            prev_ci_width = ci_width
        
        # Average CI stability across sample fractions
        if ci_stability:
            mean_ci_stability = np.mean(ci_stability, axis=0)
        else:
            mean_ci_stability = np.zeros(n_agents)
        
        # Final convergence assessment
        convergence_assessment = []
        for i in range(n_agents):
            # Get mean stability across windows
            mean_stability = np.mean([stability_metrics[k][i] for k in stability_metrics if i < len(stability_metrics[k])])
            
            # Combine with CI stability for overall assessment
            if i < len(mean_ci_stability):
                overall_stability = (mean_stability + mean_ci_stability[i]) / 2
            else:
                overall_stability = mean_stability
            
            # Assessment based on stability
            if overall_stability < 0.01:
                assessment = "Excellent"
            elif overall_stability < 0.05:
                assessment = "Good"
            elif overall_stability < 0.10:
                assessment = "Fair"
            else:
                assessment = "Poor - More data needed"
            
            convergence_assessment.append(assessment)
        
        return {
            'convergence': {
                'Mean Stability': [np.mean([stability_metrics[k][i] for k in stability_metrics if i < len(stability_metrics[k])]) 
                                for i in range(n_agents)],
                'CI Stability': mean_ci_stability if len(mean_ci_stability) == n_agents else [0] * n_agents,
                'Convergence Assessment': convergence_assessment
            }
        }
    except Exception as e:
        return {'error': f"Error calculating convergence metrics: {e}"}

def generate_overall_recommendation(row):
    """
    Generate an overall recommendation based on all metrics.
    This function implements the paper's methodology for determining
    if more simulator data is needed.
    
    Args:
        row: Row from the assessment DataFrame
        
    Returns:
        String with overall recommendation
    """
    # Count assessments by category
    assessment_columns = [col for col in row.index if ' Assessment' in col]
    if not assessment_columns:
        return "Unable to assess - insufficient data"
    
    assessment_counts = defaultdict(int)
    for col in assessment_columns:
        assessment = row[col]
        if 'Excellent' in assessment:
            assessment_counts['excellent'] += 1
        elif 'Good' in assessment:
            assessment_counts['good'] += 1
        elif 'Fair' in assessment:
            assessment_counts['fair'] += 1
        elif 'Poor' in assessment:
            assessment_counts['poor'] += 1
    
    total = sum(assessment_counts.values())
    if total == 0:
        return "Unable to assess - insufficient data"
    
    # Calculate proportions
    poor_ratio = assessment_counts['poor'] / total
    fair_ratio = assessment_counts['fair'] / total
    good_ratio = assessment_counts['good'] / total
    excellent_ratio = assessment_counts['excellent'] / total
    
    # Generate recommendation based on paper's criteria
    if poor_ratio > 0.25:
        return "More simulator data strongly recommended"
    elif poor_ratio > 0 or fair_ratio > 0.5:
        return "More simulator data recommended"
    elif excellent_ratio > 0.5:
        return "Simulator data sufficient - high confidence"
    else:
        return "Simulator data likely sufficient"

def create_bootstrap_diagnostic_plots(data_array, agent_names, stat_name, output_dir, stat_data=None):
    """
    Create comprehensive diagnostic plots to assess bootstrap convergence 
    and simulator data needs, as described in the bootstrap paper.
    
    Args:
        data_array: Array of bootstrap samples, shape (n_samples, n_agents)
        agent_names: List of agent names
        stat_name: Name of the statistic
        output_dir: Directory to save plots
        stat_data: Optional pre-computed statistics
    """
    n_samples = data_array.shape[0]
    n_agents = min(data_array.shape[1], len(agent_names))
    
    if stat_data is None:
        # Calculate basic statistics
        means = np.mean(data_array, axis=0)
        stds = np.std(data_array, axis=0, ddof=1)
        monte_carlo_errors = stds / np.sqrt(n_samples)
        relative_errors = np.zeros_like(monte_carlo_errors)
        
        # Calculate relative errors with protection against division by zero
        epsilon = 1e-10
        for i in range(len(monte_carlo_errors)):
            if abs(means[i]) > epsilon:
                relative_errors[i] = abs(monte_carlo_errors[i] / means[i])
            else:
                relative_errors[i] = monte_carlo_errors[i] / max(stds[i], epsilon)
    else:
        # Use pre-computed statistics
        means = stat_data['means']
        stds = stat_data['stds']
        monte_carlo_errors = stat_data['monte_carlo_errors']
        relative_errors = stat_data['relative_errors']
    
    # 1. Create Monte Carlo error diagnostics plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Bootstrap Diagnostics for {stat_name}\n(From Bootstrap Paper Methodology)', 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: Monte Carlo errors vs means
    ax = axes[0, 0]
    ax.bar(range(n_agents), monte_carlo_errors)
    ax.set_title('Monte Carlo Errors')
    ax.set_ylabel('Standard Error')
    ax.set_xticks(range(n_agents))
    ax.set_xticklabels([agent_names[i] if i < len(agent_names) else f"Agent {i}" 
                        for i in range(n_agents)], rotation=45, ha='right')
    ax.grid(True, alpha=0.3)
    
    # Add reference lines for Monte Carlo errors
    for i, error in enumerate(monte_carlo_errors):
        if i < n_agents:
            ax.text(i, error, f"{error:.4f}", ha='center', va='bottom', fontsize=8)
    
    # Plot 2: Relative Monte Carlo errors
    ax = axes[0, 1]
    bars = ax.bar(range(n_agents), relative_errors * 100)  # Convert to percentage
    
    # Color bars based on relative error thresholds from the paper
    colors = []
    for rel_error in relative_errors:
        if rel_error < 0.01:
            colors.append('green')        # Excellent
        elif rel_error < 0.05:
            colors.append('lightgreen')   # Good
        elif rel_error < 0.10:
            colors.append('orange')       # Fair
        else:
            colors.append('red')          # Poor - More data needed
    
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    ax.set_title('Relative Monte Carlo Errors\n(Key Metric from Bootstrap Paper)')
    ax.set_ylabel('Relative Error (%)')
    ax.set_xticks(range(n_agents))
    ax.set_xticklabels([agent_names[i] if i < len(agent_names) else f"Agent {i}" 
                        for i in range(n_agents)], rotation=45, ha='right')
    
    # Add reference lines for relative error thresholds
    ax.axhline(y=1, color='green', linestyle='--', alpha=0.7, 
              label='1% - Excellent')
    ax.axhline(y=5, color='lightgreen', linestyle='--', alpha=0.7, 
              label='5% - Good')
    ax.axhline(y=10, color='orange', linestyle='--', alpha=0.7, 
              label='10% - Fair')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Add text annotations for relative errors
    for i, rel_error in enumerate(relative_errors):
        if i < n_agents:
            ax.text(i, rel_error * 100, f"{rel_error*100:.1f}%", 
                   ha='center', va='bottom', fontsize=8)
    
    # Plot 3: Bootstrap running mean
    ax = axes[1, 0]
    
    # Calculate running means
    running_means = np.zeros((n_samples, n_agents))
    for i in range(1, n_samples + 1):
        running_means[i-1] = np.mean(data_array[:i], axis=0)
    
    for i in range(n_agents):
        if i < len(agent_names):
            ax.plot(range(1, n_samples + 1), running_means[:, i], 
                   label=agent_names[i])
        else:
            ax.plot(range(1, n_samples + 1), running_means[:, i], 
                   label=f"Agent {i}")
    
    ax.set_title('Bootstrap Running Mean\n(Assesses Bootstrap Sample Convergence)')
    ax.set_xlabel('Number of Bootstrap Samples')
    ax.set_ylabel('Mean Value')
    ax.grid(True, alpha=0.3)
    
    # Add legend if we don't have too many agents
    if n_agents <= 10:
        ax.legend(fontsize=8)
    
    # Plot 4: Bootstrap confidence interval stability
    ax = axes[1, 1]
    
    # Calculate CIs at different sample sizes
    sample_fractions = [0.25, 0.5, 0.75, 1.0]
    ci_widths = []
    
    for fraction in sample_fractions:
        n_subset = max(10, int(n_samples * fraction))
        subset = data_array[:n_subset]
        
        lower_ci = np.percentile(subset, 2.5, axis=0)
        upper_ci = np.percentile(subset, 97.5, axis=0)
        ci_width = upper_ci - lower_ci
        ci_widths.append(ci_width)
    
    # Plot CI width vs sample size
    width_array = np.array(ci_widths)
    
    # For each agent, plot CI width vs sample fraction
    for i in range(n_agents):
        if i < width_array.shape[1]:
            agent_widths = width_array[:, i]
            if i < len(agent_names):
                ax.plot(sample_fractions, agent_widths, 'o-', 
                       label=agent_names[i])
            else:
                ax.plot(sample_fractions, agent_widths, 'o-', 
                       label=f"Agent {i}")
    
    ax.set_title('95% CI Width vs Bootstrap Sample Size\n(Assesses CI Stability)')
    ax.set_xlabel('Fraction of Bootstrap Samples')
    ax.set_ylabel('CI Width')
    ax.grid(True, alpha=0.3)
    
    # Add legend if we don't have too many agents
    if n_agents <= 10:
        ax.legend(fontsize=8)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for suptitle
    
    # Save the plot
    plt.savefig(os.path.join(output_dir, f'{stat_name.replace(" ", "_")}_diagnostics.png'), 
               dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # 2. Create detailed bootstrap distribution plots for each agent
    create_agent_distribution_plots(data_array, agent_names, stat_name, output_dir)
    
    # 3. Create bootstrap paper convergence assessment plot
    create_convergence_assessment_plot(data_array, agent_names, stat_name, 
                                      output_dir, relative_errors, monte_carlo_errors)

def create_agent_distribution_plots(data_array, agent_names, stat_name, output_dir):
    """
    Create detailed bootstrap distribution plots for each agent.
    This helps assess the shape and stability of the bootstrap distributions.
    
    Args:
        data_array: Array of bootstrap samples, shape (n_samples, n_agents)
        agent_names: List of agent names
        stat_name: Name of the statistic
        output_dir: Directory to save plots
    """
    n_agents = min(data_array.shape[1], len(agent_names))
    
    # Maximum number of agents to plot per figure
    max_agents_per_fig = 6
    
    # Calculate number of figures needed
    n_figures = (n_agents + max_agents_per_fig - 1) // max_agents_per_fig
    
    for fig_idx in range(n_figures):
        start_idx = fig_idx * max_agents_per_fig
        end_idx = min(start_idx + max_agents_per_fig, n_agents)
        
        n_plots = end_idx - start_idx
        n_cols = min(3, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
        fig.suptitle(f'Bootstrap Distributions for {stat_name}\nAgents {start_idx+1}-{end_idx}', 
                    fontsize=16, fontweight='bold')
        
        # Handle single subplot case
        if n_rows * n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1 or n_cols == 1:
            axes = np.array(axes).reshape(n_rows, n_cols)
        
        for i, agent_idx in enumerate(range(start_idx, end_idx)):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col]
            
            # Get agent data
            if agent_idx < data_array.shape[1]:
                agent_data = data_array[:, agent_idx]
                agent_name = agent_names[agent_idx] if agent_idx < len(agent_names) else f"Agent {agent_idx}"
                
                # Calculate statistics
                mean_val = np.mean(agent_data)
                std_val = np.std(agent_data, ddof=1)
                lower_ci = np.percentile(agent_data, 2.5)
                upper_ci = np.percentile(agent_data, 97.5)
                
                # Plot histogram
                sns.histplot(agent_data, ax=ax, kde=True)
                
                # Add lines for mean and CI
                ax.axvline(mean_val, color='red', linestyle='-', linewidth=1.5, 
                          label=f'Mean: {mean_val:.4f}')
                ax.axvline(lower_ci, color='green', linestyle='--', linewidth=1, 
                          label=f'2.5%: {lower_ci:.4f}')
                ax.axvline(upper_ci, color='green', linestyle='--', linewidth=1, 
                          label=f'97.5%: {upper_ci:.4f}')
                
                # Calculate Monte Carlo error and relative error
                n_samples = len(agent_data)
                mc_error = std_val / np.sqrt(n_samples)
                rel_error = abs(mc_error / mean_val) if abs(mean_val) > 1e-10 else mc_error / max(std_val, 1e-10)
                
                # Add annotation with statistics
                stat_text = (f"Mean: {mean_val:.4f}\n"
                            f"Std: {std_val:.4f}\n"
                            f"95% CI: [{lower_ci:.4f}, {upper_ci:.4f}]\n"
                            f"MCE: {mc_error:.4f}\n"
                            f"Rel Error: {rel_error*100:.1f}%")
                
                # Color code the relative error assessment
                if rel_error < 0.01:
                    rel_error_text = "Excellent"
                    rel_error_color = 'green'
                elif rel_error < 0.05:
                    rel_error_text = "Good"
                    rel_error_color = 'blue'
                elif rel_error < 0.10:
                    rel_error_text = "Fair"
                    rel_error_color = 'orange'
                else:
                    rel_error_text = "Poor - More data needed"
                    rel_error_color = 'red'
                
                # Add assessment text
                assessment_text = f"Assessment: {rel_error_text}"
                
                # Position the annotations
                props = dict(boxstyle='round', facecolor='white', alpha=0.7)
                ax.text(0.05, 0.95, stat_text, transform=ax.transAxes, fontsize=9,
                       verticalalignment='top', bbox=props)
                ax.text(0.05, 0.65, assessment_text, transform=ax.transAxes, fontsize=9,
                       verticalalignment='top', color=rel_error_color, bbox=props)
                
                ax.set_title(agent_name)
                ax.grid(True, alpha=0.3)
            else:
                ax.axis('off')  # Hide unused subplots
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for suptitle
        
        # Save the figure
        fig_name = f'{stat_name.replace(" ", "_")}_distributions_{fig_idx+1}.png'
        plt.savefig(os.path.join(output_dir, fig_name), dpi=300, bbox_inches='tight')
        plt.close(fig)

def create_convergence_assessment_plot(data_array, agent_names, stat_name, output_dir, 
                                      relative_errors, monte_carlo_errors):
    """
    Create a summary convergence assessment plot as described in the bootstrap paper.
    This visualizes whether more simulator data is needed for each agent.
    
    Args:
        data_array: Array of bootstrap samples
        agent_names: List of agent names
        stat_name: Name of the statistic
        output_dir: Directory to save plots
        relative_errors: Pre-computed relative Monte Carlo errors
        monte_carlo_errors: Pre-computed Monte Carlo errors
    """
    n_agents = min(data_array.shape[1], len(agent_names))
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(f'Bootstrap Convergence Assessment for {stat_name}\n(Based on Bootstrap Paper Methodology)', 
                fontsize=16, fontweight='bold')
    
    # Prepare data
    agent_labels = [agent_names[i] if i < len(agent_names) else f"Agent {i}" 
                   for i in range(n_agents)]
    
    # Sort by relative error for better visualization
    sort_indices = np.argsort(relative_errors)[::-1]  # Descending order
    sorted_labels = [agent_labels[i] for i in sort_indices]
    sorted_rel_errors = [relative_errors[i] for i in sort_indices]
    
    # Create color-coded bars
    colors = []
    for rel_error in sorted_rel_errors:
        if rel_error < 0.01:
            colors.append('green')        # Excellent
        elif rel_error < 0.05:
            colors.append('lightgreen')   # Good
        elif rel_error < 0.10:
            colors.append('orange')       # Fair
        else:
            colors.append('red')          # Poor - More data needed
    
    # Create horizontal bar chart
    bars = ax.barh(sorted_labels, [100*e for e in sorted_rel_errors], color=colors)
    
    # Add text labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        rel_error = sorted_rel_errors[i]
        
        # Assessment label
        if rel_error < 0.01:
            assessment = "Excellent"
        elif rel_error < 0.05:
            assessment = "Good"
        elif rel_error < 0.10:
            assessment = "Fair"
        else:
            assessment = "Poor - More data needed"
        
        # Add percentage and assessment
        ax.text(width + 1, bar.get_y() + bar.get_height()/2, 
               f"{rel_error*100:.1f}% - {assessment}", 
               va='center', fontsize=9)
    
    # Add reference lines
    ax.axvline(x=1, color='green', linestyle='--', alpha=0.7, 
              label='1% - Excellent')
    ax.axvline(x=5, color='lightgreen', linestyle='--', alpha=0.7, 
              label='5% - Good')
    ax.axvline(x=10, color='orange', linestyle='--', alpha=0.7, 
              label='10% - Fair')
    
    ax.set_xlabel('Relative Monte Carlo Error (%)')
    ax.set_title('Relative Monte Carlo Error by Agent\n(Key Metric from Bootstrap Paper)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Add explanation text
    explanation = (
        "Bootstrap Paper Interpretation:\n"
        "• Relative Error < 1%: Excellent convergence, no more data needed\n"
        "• Relative Error < 5%: Good convergence, sufficient data\n"
        "• Relative Error < 10%: Fair convergence, consider more data\n"
        "• Relative Error > 10%: Poor convergence, more simulator data needed"
    )
    
    # Add text box with explanation
    props = dict(boxstyle='round', facecolor='white', alpha=0.7)
    ax.text(0.5, -0.15, explanation, transform=ax.transAxes, fontsize=10,
           va='center', ha='center', bbox=props)
    
    plt.tight_layout(rect=[0, 0, 1, 0.85])  # Make room for title and explanation
    
    # Save the plot
    plt.savefig(os.path.join(output_dir, f'{stat_name.replace(" ", "_")}_convergence_assessment.png'), 
               dpi=300, bbox_inches='tight')
    plt.close(fig)

def run_bootstrap_paper_diagnostics(bootstrap_results, agent_names, output_dir=None):
    """
    Top-level function to run all bootstrap paper diagnostics and generate 
    a comprehensive report on simulator data needs.
    
    Args:
        bootstrap_results: Dictionary containing bootstrap results
        agent_names: List of agent names
        output_dir: Directory to save results and plots
        
    Returns:
        DataFrame with assessment of simulator data needs
    """
    print("\nRunning diagnostics from Bootstrap Statistics for Empirical Games paper...")
    
    # Create output directory if needed
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Run the assessment
        assessment_df = assess_simulator_data_needs(
            bootstrap_results, 
            agent_names, 
            output_dir=output_dir, 
            plot_diagnostics=True
        )
        
        # Save assessment to CSV
        if output_dir:
            csv_path = os.path.join(output_dir, 'simulator_data_needs_assessment.csv')
            assessment_df.to_csv(csv_path, index=False)
            print(f"Saved simulator data needs assessment to {csv_path}")
        
        # Print summary to console
        print("\nSimulator Data Needs Assessment Summary (from Bootstrap Paper):")
        print("-" * 80)
        
        # Group by recommendation
        recommendation_groups = assessment_df.groupby('Overall Recommendation')
        for recommendation, group in recommendation_groups:
            print(f"\n{recommendation}:")
            for _, row in group.iterrows():
                print(f"  • {row['Agent']}")
        
        # Print high level summary
        print("\nKey Bootstrap Paper Metrics:")
        print("-" * 80)
        
        # Get Nash regret relative error column if it exists
        ne_rel_err_col = next((col for col in assessment_df.columns 
                              if 'Nash Equilibrium Regret Relative Error' in col), None)
        
        if ne_rel_err_col:
            # Calculate descriptive statistics
            max_rel_err = assessment_df[ne_rel_err_col].max() * 100
            mean_rel_err = assessment_df[ne_rel_err_col].mean() * 100
            agents_over_10pct = assessment_df[assessment_df[ne_rel_err_col] > 0.10]['Agent'].tolist()
            
            print(f"• Mean Relative Error: {mean_rel_err:.2f}%")
            print(f"• Maximum Relative Error: {max_rel_err:.2f}%")
            
            if agents_over_10pct:
                print(f"• Agents needing more data (>10% relative error): {', '.join(agents_over_10pct)}")
            else:
                print("• No agents with critical data needs (all < 10% relative error)")
        
        # Add a comprehensive conclusion
        more_data_needed = any(assessment_df['Overall Recommendation'].str.contains('recommended'))
        
        print("\nConclusion from Bootstrap Paper Methodology:")
        print("-" * 80)
        
        if more_data_needed:
            print("• More simulator data is recommended to improve statistical reliability.")
            print("• See diagnostic plots in the output directory for details on specific agents.")
        else:
            print("• Current simulator data appears sufficient based on bootstrap analysis.")
            print("• Bootstrap statistics show good convergence properties.")
        
        print("\nDiagnostic plots generated in the output directory show:")
        print("• Relative Monte Carlo errors (key metric from the paper)")
        print("• Bootstrap distribution stability")
        print("• Confidence interval convergence")
        print("• Individual agent bootstrap distributions")
        
        return assessment_df
    
    except Exception as e:
        print(f"Error running bootstrap paper diagnostics: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame({'Error': [str(e)]}) 