"""
Nash equilibrium solver for 2-player games.
"""

import numpy as np
from scipy.optimize import linprog
import warnings

def milp_max_sym_ent_2p(game_matrix, max_iter=10000):
    """
    Compute maximum entropy Nash equilibrium for 2-player games
    
    Args:
        game_matrix: numpy array of payoffs
        max_iter: maximum iterations
        
    Returns:
        nash_strategy: equilibrium strategy (probability distribution over actions)
    """
    game_matrix_np = np.array(game_matrix, dtype=np.float64)
    
    if np.isnan(game_matrix_np).any():
        for j in range(game_matrix_np.shape[1]):
            col = game_matrix_np[:, j]
            if np.isnan(col).any():
                col_mean = np.nanmean(col)
                if np.isnan(col_mean):
                    col_mean = 0
                game_matrix_np[np.isnan(col), j] = col_mean
    
    n = game_matrix_np.shape[0]
    
    # Multiple attempts with different methods to find best solution
    best_strategy = None
    best_regret = float('inf')
    
    # First attempt: Try linear programming
    try:
        c = np.zeros(n + 1)
        c[-1] = -1  
        
        A_ub = np.zeros((n, n + 1))
        for i in range(n):
            A_ub[i, :-1] = -game_matrix_np[i, :]
            A_ub[i, -1] = 1
        b_ub = np.zeros(n)
        
        A_eq = np.zeros((1, n + 1))
        A_eq[0, :-1] = 1
        A_eq[0, -1] = 0
        b_eq = np.ones(1)
        
        bounds = [(0, None) for _ in range(n)] + [(None, None)]
        
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs', 
                          options={'disp': False, 
                                  'primal_feasibility_tolerance': 1e-10,
                                  'dual_feasibility_tolerance': 1e-10})
        
        if result.success:
            strategy = result.x[:-1]
            strategy = strategy / np.sum(strategy)
            regret = calculate_max_regret(game_matrix_np, strategy)
            best_strategy = strategy
            best_regret = regret
    except Exception as e:
        warnings.warn(f"Linear programming solver failed: {e}", UserWarning)
    
    # Second attempt: Use iterative method with multiple learning rates
    learning_rates = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01]
    for lr_init in learning_rates:
        # Initial strategy - uniform distribution
        strategy = np.ones(n) / n 
        
        # Use a stricter convergence criterion
        convergence_threshold = 1e-12
        
        for iteration in range(max_iter):
            br_payoff = game_matrix_np @ strategy  
            max_payoff = np.max(br_payoff)  
            
            epsilon = max_payoff - np.dot(br_payoff, strategy)
            
            if epsilon < convergence_threshold:
                break
                
            tol = 1e-10
            br_indices = np.where(np.abs(br_payoff - max_payoff) < tol)[0]
            
            if len(br_indices) == 1:
                new_strategy = np.zeros(n)
                new_strategy[br_indices[0]] = 1.0
            else:
                new_strategy = np.zeros(n)
                new_strategy[br_indices] = 1.0 / len(br_indices)
            
            # Use a decreasing learning rate for better convergence
            learning_rate = lr_init * (1.0 - iteration / max_iter)
            strategy = (1 - learning_rate) * strategy + learning_rate * new_strategy
            
            # Normalize to ensure valid probability distribution
            strategy = strategy / np.sum(strategy)
        
        regret = calculate_max_regret(game_matrix_np, strategy)
        if regret < best_regret:
            best_strategy = strategy.copy()
            best_regret = regret
    
    if best_strategy is not None:
        strategy = best_strategy
    else:
        #TODO: this should never happen
        strategy = np.ones(n) / n
    
    strategy = minimize_max_regret(game_matrix_np, strategy, max_iterations=100)
    
    support_threshold = 1e-16 #go as small as possible
    support = np.where(strategy > support_threshold)[0]
    
    if len(support) > 1:
        expected_payoffs = np.dot(game_matrix_np, strategy)
        avg_payoff = np.dot(strategy, expected_payoffs)
        
        equal_payoff_actions = []
        for action in support:
            if abs(expected_payoffs[action] - avg_payoff) < 1e-6:
                equal_payoff_actions.append(action)
        
        if len(equal_payoff_actions) > 1:
            total_prob = sum(strategy[action] for action in equal_payoff_actions)
            
            avg_prob = total_prob / len(equal_payoff_actions)
            for action in equal_payoff_actions:
                strategy[action] = avg_prob
            
            strategy = strategy / np.sum(strategy)
            
            new_regret = calculate_max_regret(game_matrix_np, strategy)
            if new_regret > best_regret + 1e-6:  
                strategy = best_strategy

    
    return strategy

def replicator_dynamics_nash(game_matrix, max_iter=10000, convergence_threshold=1e-12):
    """
    Compute Nash equilibrium using replicator dynamics for 2-player games
    
    Args:
        game_matrix: numpy array of payoffs
        max_iter: maximum iterations
        convergence_threshold: convergence criterion
        
    Returns:
        nash_strategy: equilibrium strategy (probability distribution over actions)
    """
    game_matrix_np = np.array(game_matrix, dtype=np.float64)
    
    # Handle missing values
    if np.isnan(game_matrix_np).any():
        for j in range(game_matrix_np.shape[1]):
            col = game_matrix_np[:, j]
            if np.isnan(col).any():
                col_mean = np.nanmean(col)
                if np.isnan(col_mean):
                    col_mean = 0
                game_matrix_np[np.isnan(col), j] = col_mean
    
    n = game_matrix_np.shape[0]
    
    best_strategy = None
    best_regret = float('inf')
    
    initial_points = [
        np.ones(n) / n,  # Uniform
    ]
    
    # Add some random initial points
    num_random_points = 10  # 1000 is too many for most applications
    for _ in range(num_random_points):
        random_point = np.random.random(n)
        random_point = random_point / np.sum(random_point)
        initial_points.append(random_point)
    
    # Try each initial point
    for initial_strategy in initial_points:
        # Initialize with the given strategy
        strategy = initial_strategy.copy()
        
        # Implementation of replicator dynamics with reinforcement learning
        for iteration in range(max_iter):
            # Expected payoff vector
            expected_payoffs = np.dot(game_matrix_np, strategy)
            
            # Average payoff
            avg_payoff = np.dot(strategy, expected_payoffs)
            
            # Compute new strategy using replicator dynamics equation
            new_strategy = strategy * (expected_payoffs / max(avg_payoff, 1e-10))
            
            # Normalize to ensure valid probability distribution
            new_strategy = new_strategy / new_strategy.sum()
            
            # Check for convergence
            diff = np.linalg.norm(new_strategy - strategy)
            if diff < convergence_threshold:
                strategy = new_strategy
                break
                
            strategy = new_strategy
        
        # Calculate regret for this strategy
        regret = calculate_max_regret(game_matrix_np, strategy)
        if regret < best_regret:
            best_strategy = strategy.copy()
            best_regret = regret
    
    # If we found a strategy, use it
    if best_strategy is not None:
        strategy = best_strategy
    else:
        # Fallback to uniform if all methods failed
        strategy = np.ones(n) / n
    
    # Final refinement step: perform regret minimization
    strategy = minimize_max_regret(game_matrix_np, strategy, max_iterations=100)
    
    # Final verification
    regret = calculate_max_regret(game_matrix_np, strategy)
    if regret > convergence_threshold:
        # If regret is still too high, try a more aggressive regret minimization
        strategy = minimize_max_regret(game_matrix_np, strategy, max_iterations=500, learning_rate=0.2)
    
    return strategy

def calculate_max_regret(game_matrix, strategy):
    """
    Calculate the maximum regret for a given strategy
    
    Args:
        game_matrix: numpy array of payoffs
        strategy: probability distribution over actions
        
    Returns:
        max_regret: maximum regret across all actions
    """
    expected_payoffs = np.dot(game_matrix, strategy)
    avg_payoff = np.dot(strategy, expected_payoffs)
    regrets = expected_payoffs - avg_payoff
    
    # A true Nash equilibrium should have non-positive regrets
    # The maximum regret should be at most 0
    return np.max(regrets)

def minimize_max_regret(game_matrix, initial_strategy, max_iterations=100, learning_rate=0.1):
    """
    Refine a strategy to minimize maximum regret
    
    Args:
        game_matrix: numpy array of payoffs
        initial_strategy: initial probability distribution over actions
        max_iterations: maximum number of iterations
        learning_rate: learning rate for updates
        
    Returns:
        refined_strategy: refined probability distribution
    """
    n = game_matrix.shape[0]
    strategy = initial_strategy.copy()
    best_strategy = strategy.copy()
    best_regret = calculate_max_regret(game_matrix, strategy)
    
    if best_regret <= 1e-12:
        return strategy
    
    learning_rates = [learning_rate, learning_rate*0.5, learning_rate*0.2, learning_rate*0.1]
    
    for current_lr_base in learning_rates:
        local_strategy = strategy.copy()
        
        for iteration in range(max_iterations):
            expected_payoffs = np.dot(game_matrix, local_strategy)
            avg_payoff = np.dot(local_strategy, expected_payoffs)
            regrets = expected_payoffs - avg_payoff
            
            max_regret_idx = np.argmax(regrets)
            max_regret = regrets[max_regret_idx]
            
            # If maximum regret is already non-positive, we're done
            if max_regret <= 1e-12:
                # Check if this is better than our best so far
                current_regret = calculate_max_regret(game_matrix, local_strategy)
                if current_regret < best_regret:
                    best_strategy = local_strategy.copy()
                    best_regret = current_regret
                break
            
            positive_regrets = np.maximum(regrets, 0)
            total_positive_regret = np.sum(positive_regrets)
            
            if total_positive_regret > 0:
                update = positive_regrets / total_positive_regret
            else:
                update = np.zeros(n)
                update[max_regret_idx] = 1.0
            
            # Adjust learning rate based on current iteration
            current_lr = current_lr_base * (1.0 - iteration / max_iterations)
            
            # Apply update
            local_strategy = (1 - current_lr) * local_strategy + current_lr * update
            
            # Normalize to ensure valid probability distribution
            local_strategy = local_strategy / np.sum(local_strategy)
            
            # Check if this improved the maximum regret
            current_regret = calculate_max_regret(game_matrix, local_strategy)
            if current_regret < best_regret:
                best_strategy = local_strategy.copy()
                best_regret = current_regret
        
    # For very small remaining regrets, try a more targeted approach
    if best_regret > 1e-12:
        # Get regrets for the best strategy
        expected_payoffs = np.dot(game_matrix, best_strategy)
        avg_payoff = np.dot(best_strategy, expected_payoffs)
        regrets = expected_payoffs - avg_payoff
        
        # Identify the indexes with positive regret
        positive_idxs = np.where(regrets > 1e-4)[0]
        
        # Try giving each of those indexes slightly more weight
        for idx in positive_idxs:
            for weight in [0.001, 0.01, 0.05, 0.1, 0.2]:
                # Create a new strategy with increased weight for this index
                modified_strategy = best_strategy.copy()
                modified_strategy[idx] += weight
                modified_strategy = modified_strategy / np.sum(modified_strategy)
                
                # Calculate regret
                modified_regret = calculate_max_regret(game_matrix, modified_strategy)
                
                # If this improves the regret, update the best strategy
                if modified_regret < best_regret:
                    best_strategy = modified_strategy
                    best_regret = modified_regret
                    
                    # If regret is now non-positive or extremely small, we're done
                    if best_regret <= 1e-12:
                        break
            
            # If regret is now non-positive or extremely small, we're done
            if best_regret <= 1e-12:
                break
    
    return best_strategy