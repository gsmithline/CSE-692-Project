import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from nash_equilibrium.nash_solver import _simplex_projection, compute_regret, replicator_dynamics_nash, EPSILON

def find_eps_nash(payoff_matrix, agent_names, epsilon=0.05, max_iter=10000, verbose=True):
    """
    Find an ε-Nash equilibrium using multiple specialized techniques.
    
    Args:
        payoff_matrix: Game payoff matrix
        agent_names: Names of the agents
        epsilon: Epsilon value for Nash equilibrium
        max_iter: Maximum iterations
        verbose: Whether to print progress
        
    Returns:
        strategy: Epsilon-Nash equilibrium strategy
        info: Dictionary with additional information
    """
    n_agents = len(agent_names)
    
    best_strategy = None
    best_regret = float('inf')
    all_strategies = []
    all_regrets = []
    
    
    if verbose:
        print("Trying uniform strategy...")
    uniform = np.ones(n_agents) / n_agents
    uniform_regret, uniform_value, _ = compute_regret(uniform, payoff_matrix)
    max_uniform_regret = np.max(uniform_regret)
    
    if max_uniform_regret <= epsilon:
        if verbose:
            print("Uniform strategy is already an ε-Nash equilibrium!")
        info = {
            "regret": max_uniform_regret, 
            "value": uniform_value,
            "strategy_details": pd.DataFrame({
                'Agent': agent_names,
                'Weight': uniform,
                'Expected Payoff': payoff_matrix @ uniform,
                'Regret': uniform_regret
            }).sort_values('Regret', ascending=False)
        }
        return uniform, info
    
    all_strategies.append(uniform)
    all_regrets.append(max_uniform_regret)
    
    if max_uniform_regret < best_regret:
        best_strategy = uniform
        best_regret = max_uniform_regret
    
    if verbose:
        print("Trying pure strategies...")
    
    for i in range(n_agents):
        pure = np.zeros(n_agents)
        pure[i] = 1.0
        
        pure_regret, pure_value, _ = compute_regret(pure, payoff_matrix)
        max_pure_regret = np.max(pure_regret)
        
        all_strategies.append(pure)
        all_regrets.append(max_pure_regret)
        
        if max_pure_regret < best_regret:
            best_strategy = pure
            best_regret = max_pure_regret
            
            if max_pure_regret <= epsilon:
                if verbose:
                    print(f"Found ε-Nash equilibrium (pure strategy {agent_names[i]})!")
                info = {
                    "regret": max_pure_regret, 
                    "value": pure_value,
                    "strategy_details": pd.DataFrame({
                        'Agent': agent_names,
                        'Weight': pure,
                        'Expected Payoff': payoff_matrix @ pure,
                        'Regret': pure_regret
                    }).sort_values('Regret', ascending=False)
                }
                return pure, info
    
    if verbose:
        print("Trying replicator dynamics...")
    
    # Use the nash_solver implementation
    rd_strategy, iterations = replicator_dynamics_nash(payoff_matrix, max_iter=max_iter, epsilon=epsilon)
    rd_regret, rd_value, rd_expected_utils = compute_regret(rd_strategy, payoff_matrix)
    max_rd_regret = np.max(rd_regret)
    
    all_strategies.append(rd_strategy)
    all_regrets.append(max_rd_regret)
    
    if max_rd_regret < best_regret:
        best_strategy = rd_strategy
        best_regret = max_rd_regret
        
        if best_regret <= epsilon:
            if verbose:
                print(f"Found ε-Nash equilibrium using replicator dynamics!")
            info = {
                "regret": max_rd_regret,
                "value": rd_value,
                "strategy_details": pd.DataFrame({
                    'Agent': agent_names,
                    'Weight': rd_strategy,
                    'Expected Payoff': rd_expected_utils,
                    'Regret': rd_regret
                }).sort_values('Regret', ascending=False)
            }
            return best_strategy, info
    
    # 4. Try direct regret minimization
    if verbose:
        print("Trying direct regret minimization...")
    
    # Start from best strategy found so far
    drm_strategy, drm_info = direct_regret_minimization(
        payoff_matrix,
        initial_strategy=best_strategy,
        epsilon=epsilon,
        max_iter=3000,
        verbose=verbose
    )
    
    all_strategies.append(drm_strategy)
    all_regrets.append(drm_info["regret"])
    
    if drm_info["regret"] < best_regret:
        best_strategy = drm_strategy
        best_regret = drm_info["regret"]
        
        if best_regret <= epsilon:
            if verbose:
                print(f"Found ε-Nash equilibrium using direct regret minimization!")
            return best_strategy, drm_info
    
    # 5. Try support enumeration with small supports
    if verbose:
        print("Trying support enumeration for small supports...")
    
    # Order agents by their performance in best strategy so far
    expected_payoffs = payoff_matrix @ best_strategy
    ordered_indices = np.argsort(expected_payoffs)[::-1]  # Descending order
    
    # Try supports of size 2, 3, and 4
    for support_size in [2, 3, 4]:
        if support_size >= n_agents:
            continue
            
        # Try different supports with specified size
        for i in range(min(5, n_agents - support_size + 1)):  # Limit to 5 attempts
            support_indices = ordered_indices[i:i+support_size]
            
            se_strategy, se_info = support_enumeration(
                payoff_matrix,
                support_indices=support_indices,
                epsilon=epsilon,
                verbose=False
            )
            
            all_strategies.append(se_strategy)
            all_regrets.append(se_info["regret"])
            
            if se_info["regret"] < best_regret:
                best_strategy = se_strategy
                best_regret = se_info["regret"]
                
                if best_regret <= epsilon:
                    if verbose:
                        print(f"Found ε-Nash equilibrium using support enumeration!")
                        print(f"Support: {[agent_names[idx] for idx in support_indices]}")
                    return best_strategy, se_info
    
    # 6. Try evolutionary strategy with crossover and mutation
    if verbose:
        print("Trying evolutionary strategy optimization...")
    
    # Use top 10 strategies found so far as initial population
    top_indices = np.argsort(all_regrets)[:10]
    initial_population = [all_strategies[i] for i in top_indices]
    
    es_strategy, es_info = evolutionary_strategy(
        payoff_matrix,
        initial_population=initial_population,
        epsilon=epsilon,
        generations=100,
        verbose=verbose
    )
    
    if es_info["regret"] < best_regret:
        best_strategy = es_strategy
        best_regret = es_info["regret"]
        
        if best_regret <= epsilon:
            if verbose:
                print(f"Found ε-Nash equilibrium using evolutionary strategy!")
            return best_strategy, es_info
    
    # If we reach here, we didn't find an exact ε-Nash equilibrium
    if verbose:
        print(f"Failed to find exact ε-Nash equilibrium. Best regret: {best_regret:.6f}")
    
    info = {
        "regret": best_regret,
        "strategy_details": pd.DataFrame({
            'Agent': agent_names,
            'Weight': best_strategy,
            'Expected Payoff': payoff_matrix @ best_strategy,
            'Regret': compute_regret(best_strategy, payoff_matrix)[0]
        }).sort_values('Regret', ascending=False)
    }
    
    return best_strategy, info

def direct_regret_minimization(payoff_matrix, initial_strategy=None, epsilon=0.05, max_iter=1000, verbose=True):
    """
    Find an ε-Nash equilibrium using direct regret minimization.
    
    Args:
        payoff_matrix: Game payoff matrix
        initial_strategy: Initial strategy
        epsilon: Epsilon value for Nash equilibrium
        max_iter: Maximum iterations
        verbose: Whether to print progress
        
    Returns:
        strategy: Best strategy found
        info: Dictionary with additional information
    """
    n_agents = payoff_matrix.shape[0]
    
    # Initialize strategy
    if initial_strategy is None:
        strategy = np.ones(n_agents) / n_agents
    else:
        strategy = initial_strategy.copy()
    
    best_strategy = strategy.copy()
    best_regret = float('inf')
    
    trace = []
    
    # Run direct regret minimization
    for iteration in range(max_iter):
        # Calculate regret
        regret, nash_value, expected_payoffs = compute_regret(strategy, payoff_matrix)
        max_regret = np.max(regret)
        
        # Track best strategy
        if max_regret < best_regret:
            best_strategy = strategy.copy()
            best_regret = max_regret
            
            if best_regret <= epsilon:
                # Found ε-Nash equilibrium
                if verbose and (iteration % 100 == 0 or iteration < 10):
                    print(f"Iteration {iteration}: Found ε-Nash equilibrium with regret {best_regret:.6f}")
                break
        
        # Add to trace
        trace.append({
            'iteration': iteration,
            'strategy': strategy.copy(),
            'regret': max_regret
        })
        
        if verbose and iteration % 100 == 0:
            print(f"Iteration {iteration}: Current regret {max_regret:.6f}, Best regret {best_regret:.6f}")
        
        # Find best response
        best_response_idx = np.argmax(regret)
        
        # Direct regret minimization step
        step_size = max(0.1, 0.5 / (iteration + 1))  # Decreasing step size
        
        # Try different update approaches
        if iteration % 3 == 0:
            # Move weight toward best response from worst response
            new_strategy = strategy.copy()
            worst_response_idx = np.argmin(expected_payoffs)
            
            if strategy[worst_response_idx] > step_size:
                transfer = step_size
                new_strategy[worst_response_idx] -= transfer
                new_strategy[best_response_idx] += transfer
            else:
                # Move small weight from all strategies proportionally
                for i in range(n_agents):
                    if i != best_response_idx:
                        transfer = min(step_size * strategy[i], strategy[i] * 0.2)
                        new_strategy[i] -= transfer
                        new_strategy[best_response_idx] += transfer
        
        elif iteration % 3 == 1:
            # Frank-Wolfe style update
            target = np.zeros(n_agents)
            target[best_response_idx] = 1.0
            
            # Convex combination
            new_strategy = (1 - step_size) * strategy + step_size * target
        
        else:
            # Multiplicative weights update
            weights = np.exp(step_size * regret)
            new_strategy = strategy * weights
            new_strategy = new_strategy / np.sum(new_strategy)
        
        # Ensure projection to simplex
        new_strategy = _simplex_projection(new_strategy)
        
        # Check for small changes - may be stuck
        strategy_change = np.max(np.abs(new_strategy - strategy))
        
        if strategy_change < 1e-6 and iteration % 50 == 0:
            # Perturb the strategy to escape local minima
            perturbation = np.random.random(n_agents) * 0.1
            new_strategy = new_strategy * 0.9 + perturbation / np.sum(perturbation) * 0.1
            new_strategy = _simplex_projection(new_strategy)
        
        strategy = new_strategy
    
    # Final check and projection
    best_strategy = _simplex_projection(best_strategy)
    final_regret, final_value, _ = compute_regret(best_strategy, payoff_matrix)
    max_final_regret = np.max(final_regret)
    
    info = {
        "regret": max_final_regret,
        "value": final_value,
        "iterations": iteration + 1,
        "trace": trace,
        "converged": max_final_regret <= epsilon
    }
    
    return best_strategy, info

def support_enumeration(payoff_matrix, support_indices, epsilon=0.05, verbose=False):
    """
    Find an ε-Nash equilibrium by restricting to a specific support.
    
    Args:
        payoff_matrix: Game payoff matrix
        support_indices: Indices of strategies to include in support
        epsilon: Epsilon value for Nash equilibrium
        verbose: Whether to print progress
        
    Returns:
        strategy: Best strategy found
        info: Dictionary with additional information
    """
    n_agents = payoff_matrix.shape[0]
    k = len(support_indices)
    
    # Initialize restricted game
    restricted_payoff = payoff_matrix[np.ix_(support_indices, support_indices)]
    
    # Try to find equalizing strategy on restricted game
    # We want to find a strategy such that all pure strategies in support have equal payoff
    
    # Start with uniform
    restricted_strategy = np.ones(k) / k
    
    # Run optimization on restricted game
    restricted_optimal, r_info = direct_regret_minimization(
        restricted_payoff,
        initial_strategy=restricted_strategy,
        epsilon=epsilon,
        max_iter=500,
        verbose=False
    )
    
    # Map back to full game
    full_strategy = np.zeros(n_agents)
    for i, idx in enumerate(support_indices):
        full_strategy[idx] = restricted_optimal[i]
    
    # Check regret in full game
    regret, value, _ = compute_regret(full_strategy, payoff_matrix)
    max_regret = np.max(regret)
    
    info = {
        "regret": max_regret,
        "value": value,
        "support": support_indices,
        "restricted_regret": r_info["regret"],
        "converged": max_regret <= epsilon
    }
    
    return full_strategy, info

def evolutionary_strategy(payoff_matrix, initial_population, epsilon=0.05, generations=100, population_size=20, verbose=False):
    """
    Find an ε-Nash equilibrium using an evolutionary strategy approach.
    
    Args:
        payoff_matrix: Game payoff matrix
        initial_population: Initial population of strategies
        epsilon: Epsilon value for Nash equilibrium
        generations: Number of generations
        population_size: Size of population
        verbose: Whether to print progress
        
    Returns:
        strategy: Best strategy found
        info: Dictionary with additional information
    """
    n_agents = payoff_matrix.shape[0]
    
    # Initialize population
    population = list(initial_population)
    
    # Generate additional random strategies if needed
    while len(population) < population_size:
        random_strategy = np.random.random(n_agents)
        random_strategy = random_strategy / np.sum(random_strategy)
        population.append(random_strategy)
    
    best_strategy = None
    best_regret = float('inf')
    
    # Run evolutionary optimization
    for generation in range(generations):
        # Evaluate fitness (negative regret)
        fitness = []
        for strategy in population:
            regret, _, _ = compute_regret(strategy, payoff_matrix)
            max_regret = np.max(regret)
            fitness.append(-max_regret)  # Negative because we maximize fitness
            
            # Track best strategy
            if max_regret < best_regret:
                best_strategy = strategy.copy()
                best_regret = max_regret
                
                if best_regret <= epsilon:
                    # Found ε-Nash equilibrium
                    if verbose:
                        print(f"Generation {generation}: Found ε-Nash equilibrium with regret {best_regret:.6f}")
                    
                    info = {
                        "regret": best_regret,
                        "generations": generation + 1,
                        "converged": True
                    }
                    return best_strategy, info
        
        if verbose and generation % 10 == 0:
            print(f"Generation {generation}: Best regret {best_regret:.6f}")
        
        # Create new population
        new_population = []
        
        # Elitism: Keep best strategies
        elite_count = max(1, population_size // 5)
        elite_indices = np.argsort(fitness)[-elite_count:]
        for idx in elite_indices:
            new_population.append(population[idx])
        
        # Crossover and mutation
        while len(new_population) < population_size:
            # Tournament selection
            tournament_size = 3
            tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
            tournament_fitness = [fitness[i] for i in tournament_indices]
            parent1_idx = tournament_indices[np.argmax(tournament_fitness)]
            
            tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
            tournament_fitness = [fitness[i] for i in tournament_indices]
            parent2_idx = tournament_indices[np.argmax(tournament_fitness)]
            
            parent1 = population[parent1_idx]
            parent2 = population[parent2_idx]
            
            # Crossover
            crossover_point = np.random.randint(1, n_agents)
            child = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
            
            # Mutation
            mutation_rate = 0.2
            if np.random.random() < mutation_rate:
                # Random mutation
                mutation_idx = np.random.randint(n_agents)
                mutation_amount = np.random.random() * 0.2 - 0.1  # -0.1 to +0.1
                child[mutation_idx] += mutation_amount
            
            # Project to simplex
            child = _simplex_projection(child)
            
            new_population.append(child)
        
        population = new_population
    
    # Final check
    if best_strategy is None:
        best_strategy = population[np.argmax(fitness)]
    
    regret, value, _ = compute_regret(best_strategy, payoff_matrix)
    max_regret = np.max(regret)
    
    info = {
        "regret": max_regret,
        "value": value,
        "generations": generations,
        "converged": max_regret <= epsilon
    }
    
    return best_strategy, info

def main():
    # Load the performance matrix
    print("Loading performance matrix...")
    performance_matrix = pd.read_csv('meta_game_analysis/game_matrix_2_100_bootstrap/csv/performance_matrix.csv', index_col=0)
    
    # Get agent names
    agent_names = performance_matrix.index.tolist()
    
    # Convert to numpy array for calculations
    payoff_matrix = performance_matrix.values
    
    # Find ε-Nash equilibrium
    print("Finding ε-Nash equilibrium...")
    strategy, info = find_eps_nash(payoff_matrix, agent_names, epsilon=0.05, max_iter=10000)
    
    # Print results
    print("\nBest strategy found:")
    print(pd.DataFrame({
        'Agent': agent_names,
        'Weight': strategy,
    }).sort_values('Weight', ascending=False))
    
    print("\nStrategy details:")
    print(info["strategy_details"])
    
    print(f"\nMax regret: {info['regret']:.6f}")
    print(f"Is a 0.05-Nash equilibrium: {info['regret'] <= 0.05}")
    
    # Visualize strategy
    plt.figure(figsize=(10, 6))
    plt.bar(agent_names, strategy)
    plt.ylabel('Probability')
    plt.title('ε-Nash Equilibrium Strategy')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('eps_nash_strategy.png', dpi=300)
    
    return strategy, info

if __name__ == "__main__":
    main() 