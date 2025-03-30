import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import seaborn as sns
from nash_equilibrium.nash_solver import replicator_dynamics_nash, _simplex_projection, compute_regret

def visualize_strategy_trace(trace, agent_names, title="Replicator Dynamics Trace"):
    """
    Visualize the trace of strategies during replicator dynamics.
    
    Args:
        trace: List of trace entries containing strategies and regrets
        agent_names: Names of the agents
        title: Title for the plot
    """
    # Extract strategies and regrets
    strategies = np.array([entry['strategy'] for entry in trace])
    regrets = np.array([entry['regret'] for entry in trace])
    iterations = np.array([entry['iteration'] for entry in trace])
    
    # Plot regret over iterations
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, regrets, 'b-', linewidth=2)
    plt.axhline(y=0.05, color='r', linestyle='--', label="ε = 0.05")
    plt.xlabel('Iteration')
    plt.ylabel('Regret')
    plt.title(f"{title} - Regret Over Iterations")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('rd_regret.png', dpi=300)
    
    # Check if we have enough iterations to analyze convergence
    if len(trace) > 10:
        # Calculate rate of regret decrease
        regret_changes = np.diff(regrets)
        avg_regret_change = np.mean(regret_changes[-min(50, len(regret_changes)):])
        print(f"Average regret change in final iterations: {avg_regret_change:.8f}")
        
        if avg_regret_change > -0.0001:
            print("WARNING: Regret is no longer decreasing significantly - may be stuck in a cycle")
    
    # Perform PCA to visualize trajectory in 2D/3D
    n_components = min(3, len(agent_names) - 1)
    pca = PCA(n_components=n_components)
    
    # Add pure strategies to help with interpretation
    n_agents = len(agent_names)
    pure_strategies = np.eye(n_agents)
    combined_data = np.vstack([strategies, pure_strategies])
    
    # Fit PCA on combined data
    transformed = pca.fit_transform(combined_data)
    
    # Split back into trajectory and pure strategies
    trajectory = transformed[:len(strategies)]
    pure_points = transformed[len(strategies):]
    
    # Calculate variance explained
    explained_variance = pca.explained_variance_ratio_
    print(f"Variance explained by PCA components: {explained_variance}")
    
    # Create 2D or 3D visualization
    if n_components >= 3:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot trajectory
        points = ax.scatter(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 
                  c=regrets, cmap='viridis', s=30, alpha=0.8)
        fig.colorbar(points, label='Regret')
        
        # Draw lines connecting points to show trajectory
        ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 'b-', alpha=0.5)
        
        # Mark start and end
        ax.scatter(trajectory[0, 0], trajectory[0, 1], trajectory[0, 2], 
                  c='g', s=100, marker='o', label='Start')
        ax.scatter(trajectory[-1, 0], trajectory[-1, 1], trajectory[-1, 2], 
                  c='r', s=100, marker='o', label='End')
        
        # Plot pure strategies
        ax.scatter(pure_points[:, 0], pure_points[:, 1], pure_points[:, 2],
                  c='k', s=80, marker='s')
        
        # Add labels for pure strategies
        for i, agent in enumerate(agent_names):
            ax.text(pure_points[i, 0], pure_points[i, 1], pure_points[i, 2], agent, fontsize=8)
        
        ax.set_xlabel(f'PC1 ({explained_variance[0]:.2%})')
        ax.set_ylabel(f'PC2 ({explained_variance[1]:.2%})')
        ax.set_zlabel(f'PC3 ({explained_variance[2]:.2%})')
        ax.set_title(f"{title} - Strategy Trajectory")
        ax.legend()
        
    else:
        # 2D visualization
        plt.figure(figsize=(10, 8))
        
        # Plot trajectory
        points = plt.scatter(trajectory[:, 0], trajectory[:, 1], 
                  c=regrets, cmap='viridis', s=30, alpha=0.8)
        plt.colorbar(points, label='Regret')
        
        # Draw lines connecting points to show trajectory
        plt.plot(trajectory[:, 0], trajectory[:, 1], 'b-', alpha=0.5)
        
        # Mark start and end
        plt.scatter(trajectory[0, 0], trajectory[0, 1], 
                  c='g', s=100, marker='o', label='Start')
        plt.scatter(trajectory[-1, 0], trajectory[-1, 1], 
                  c='r', s=100, marker='o', label='End')
        
        # Plot pure strategies
        plt.scatter(pure_points[:, 0], pure_points[:, 1],
                  c='k', s=80, marker='s')
        
        # Add labels for pure strategies
        for i, agent in enumerate(agent_names):
            plt.text(pure_points[i, 0], pure_points[i, 1], agent, fontsize=8)
        
        plt.xlabel(f'PC1 ({explained_variance[0]:.2%})')
        plt.ylabel(f'PC2 ({explained_variance[1]:.2%})')
        plt.title(f"{title} - Strategy Trajectory")
        plt.grid(True, alpha=0.3)
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('rd_trajectory.png', dpi=300)
    
    # Check for cycling by analyzing last part of trajectory
    if len(trace) > 50:
        last_strategies = strategies[-50:]
        # Calculate distances between consecutive strategies
        distances = np.linalg.norm(np.diff(last_strategies, axis=0), axis=1)
        avg_distance = np.mean(distances)
        
        # Calculate regret changes in last iterations
        last_regrets = regrets[-50:]
        regret_changes = np.diff(last_regrets)
        
        print(f"Average distance between consecutive strategies: {avg_distance:.8f}")
        
        # Check if strategies are changing but regret isn't decreasing
        if avg_distance > 0.001 and np.mean(regret_changes) > -0.0001:
            print("WARNING: Strategies are changing but regret isn't decreasing - likely cycling")
            
            # Analyze the cycle
            plt.figure(figsize=(10, 6))
            plt.plot(range(len(distances)), distances, 'b-')
            plt.xlabel('Iteration')
            plt.ylabel('Strategy Change')
            plt.title('Strategy Changes in Final Iterations')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('rd_cycle_analysis.png', dpi=300)
    
    # Create heatmap of probabilities over time
    plt.figure(figsize=(12, 8))
    sns.heatmap(strategies, cmap='viridis', 
                xticklabels=agent_names, 
                yticklabels=np.arange(0, len(trace), max(1, len(trace)//20)))
    plt.xlabel('Agent')
    plt.ylabel('Iteration')
    plt.title(f"{title} - Strategy Probabilities Over Time")
    plt.tight_layout()
    plt.savefig('rd_heatmap.png', dpi=300)
    
    # Plot support size over time
    support_sizes = [np.sum(strategy > 0.001) for strategy in strategies]
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, support_sizes, 'g-', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Support Size')
    plt.title(f"{title} - Support Size Over Iterations")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('rd_support.png', dpi=300)
    
    # Return the final strategy and regret
    return strategies[-1], regrets[-1]

def analyze_best_responses(strategy, payoff_matrix, agent_names):
    """
    Analyze best responses to the given strategy.
    
    Args:
        strategy: Strategy to analyze
        payoff_matrix: Game payoff matrix
        agent_names: Names of the agents
    """
    # Expected payoffs for each pure strategy
    expected_payoffs = np.dot(payoff_matrix, strategy)
    
    # Nash value
    nash_value = np.dot(strategy, expected_payoffs)
    
    # Calculate regret for each pure strategy
    regrets = expected_payoffs - nash_value
    
    # Create a DataFrame with results
    results = pd.DataFrame({
        'Agent': agent_names,
        'Strategy Weight': strategy,
        'Expected Payoff': expected_payoffs,
        'Regret': regrets
    })
    
    # Sort by regret to identify best responses
    results = results.sort_values('Regret', ascending=False)
    
    # Identify best responses (agents with highest regret)
    best_responses = results[results['Regret'] > 0.001]
    
    print("\nBest Responses Analysis:")
    print(results)
    
    print("\nBest Responses:")
    if len(best_responses) > 0:
        print(best_responses)
    else:
        print("No clear best responses - strategy is close to Nash equilibrium")
    
    print(f"\nMax regret: {np.max(regrets):.6f}")
    print(f"Nash value: {nash_value:.6f}")
    
    # Plot the payoffs and regrets
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot expected payoffs
    ax1.bar(agent_names, expected_payoffs)
    ax1.axhline(y=nash_value, color='r', linestyle='--', label="Nash Value")
    ax1.set_xlabel('Agent')
    ax1.set_ylabel('Expected Payoff')
    ax1.set_title('Expected Payoffs for Pure Strategies')
    ax1.set_xticklabels(agent_names, rotation=45, ha='right')
    ax1.legend()
    
    # Plot regrets
    ax2.bar(agent_names, regrets)
    ax2.axhline(y=0.05, color='r', linestyle='--', label="ε = 0.05")
    ax2.set_xlabel('Agent')
    ax2.set_ylabel('Regret')
    ax2.set_title('Regrets for Pure Strategies')
    ax2.set_xticklabels(agent_names, rotation=45, ha='right')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('rd_best_responses.png', dpi=300)
    
    return results

def main():
    # Load the performance matrix
    print("Loading performance matrix...")
    performance_matrix = pd.read_csv('meta_game_analysis/game_matrix_2_100_bootstrap/csv/performance_matrix.csv', index_col=0)
    
    # Get agent names
    agent_names = performance_matrix.index.tolist()
    
    # Convert to numpy array for calculations
    payoff_matrix = performance_matrix.values
    
    # Find Nash equilibrium using replicator dynamics and get trace
    print("Computing Nash equilibrium using replicator dynamics...")
    strategy, trace = replicator_dynamics_nash(payoff_matrix, max_iter=5000, epsilon=0.05, step_size=0.2, return_trace=True)
    
    # Visualize the trace
    print("Visualizing trace...")
    final_strategy, final_regret = visualize_strategy_trace(trace, agent_names)
    
    # Analyze best responses
    print("Analyzing best responses...")
    best_responses = analyze_best_responses(strategy, payoff_matrix, agent_names)
    
    # Check if the final strategy is an epsilon-Nash equilibrium
    regret, nash_value, _ = compute_regret(strategy, payoff_matrix)
    max_regret = np.max(regret)
    
    print("\nFinal Results:")
    print(f"Max regret: {max_regret:.6f}")
    print(f"Nash value: {nash_value:.6f}")
    print(f"Is a 0.05-Nash equilibrium: {max_regret <= 0.05}")
    
    if max_regret > 0.05:
        print("\nWarning: Failed to find an 0.05-Nash equilibrium")
        print("Possible reasons:")
        if len(trace) >= 1000:
            # Calculate rate of regret decrease in last iterations
            regrets = np.array([entry['regret'] for entry in trace[-100:]])
            if np.mean(np.diff(regrets)) > -0.0001:
                print("- Algorithm is cycling or stuck in a local minimum")
            else:
                print("- Algorithm needs more iterations to converge")
        else:
            print("- Try increasing max_iter")
        
        print("- Try different step sizes")
        print("- Try more diverse initial strategies")
        print("- The game may not have a pure or small-support Nash equilibrium")

if __name__ == "__main__":
    main() 