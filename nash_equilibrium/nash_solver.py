"""
Nash equilibrium solver for 2-player games.
"""

import numpy as np
from scipy.optimize import linprog, minimize
import warnings
import scipy.sparse as sp
from scipy.special import entr

def milp_max_sym_ent_2p(game_matrix, max_iter=10000):
    """
    Compute maximum entropy Nash equilibrium for 2-player games
    
    Args:
        game_matrix: numpy array of payoffs
        max_iter: maximum iterations
        
    Returns:
        nash_strategy: equilibrium strategy (probability distribution over actions)
    """
    # Constants for numerical stability and threshold consistency
    EPSILON = 1e-8  # Main threshold for regret/convergence across all functions
    SUPPORT_THRESHOLD = EPSILON  # Threshold for determining support
    
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
    
    #multiple attempts with different methods to find best solution
    best_strategy = None
    best_regret = float('inf')
    
    #try linear programming with entropy regularization
    # used Zun's code as an example
    try:
        # 
        c = np.zeros(n + 1)
        c[-1] = -1 #max nash val
        
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
                                  'primal_feasibility_tolerance': EPSILON,
                                  'dual_feasibility_tolerance': EPSILON})
        
        if result.success:
            strategy = result.x[:-1]
            strategy = strategy / np.sum(strategy)
            regret = calculate_max_regret(game_matrix_np, strategy)
            
            if regret <= EPSILON:
                best_strategy = strategy
                best_regret = regret
                
                # identify the support of ne 
                support = strategy > SUPPORT_THRESHOLD
                
                if np.sum(support) > 1:
                    # get expected utility of ne 
                    expected_payoffs = np.dot(game_matrix_np, strategy)
                    nash_value = np.dot(strategy, expected_payoffs)
                    
                    def support_constraints(x):
                        expected_utils = np.dot(game_matrix_np, x)
                        return np.array([expected_utils[i] - nash_value for i in range(n) if support[i]])
                    
                    def neg_entropy(x):
                        return -np.sum(entr(x + EPSILON))  # Add epsilon to avoid log(0)
                    
                    def neg_entropy_grad(x):
                        return -np.log(x + EPSILON) - 1
                    
                    def sum_to_one(x):
                        return np.sum(x) - 1
                    
                    x0 = strategy.copy()
                    
                    bnds = [(0, 1) for _ in range(n)]
                    
                    constraints = []
                    
                    constraints.append({'type': 'eq', 'fun': sum_to_one})
                    
                    support_indices = np.where(support)[0]
                    for i in support_indices:
                        def constraint_i(x, i=i, nash_value=nash_value):
                            return np.dot(game_matrix_np[i], x) - nash_value
                        constraints.append({'type': 'eq', 'fun': constraint_i})
                    
                    try:
                        entropy_result = minimize(
                            neg_entropy,
                            x0,
                            jac=neg_entropy_grad,
                            bounds=bnds,
                            constraints=constraints,
                            method='SLSQP',
                            options={'maxiter': 1000, 'ftol': EPSILON}
                        )
                        
                        if entropy_result.success:
                            entropy_strategy = entropy_result.x
                            entropy_strategy = entropy_strategy / np.sum(entropy_strategy)
                            entropy_regret = calculate_max_regret(game_matrix_np, entropy_strategy)
                            
                            if entropy_regret <= best_regret + EPSILON:
                                best_strategy = entropy_strategy
                                best_regret = entropy_regret
                    except Exception as e:
                        warnings.warn(f"Entropy maximization failed: {e}", UserWarning)
    except Exception as e:
        warnings.warn(f"Linear programming solver failed: {e}", UserWarning)
    
    if best_strategy is None:
        learning_rates = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01]
        for lr_init in learning_rates:
            strategy = np.ones(n) / n 
            
            convergence_threshold = EPSILON
            
            for iteration in range(max_iter):
                br_payoff = game_matrix_np @ strategy  
                max_payoff = np.max(br_payoff)  
                
                epsilon = max_payoff - np.dot(br_payoff, strategy)
                
                if epsilon < convergence_threshold:
                    break
                    
                br_indices = np.where(np.abs(br_payoff - max_payoff) < EPSILON)[0]
                
                if len(br_indices) == 1:
                    new_strategy = np.zeros(n)
                    new_strategy[br_indices[0]] = 1.0
                else:
                    new_strategy = np.zeros(n)
                    new_strategy[br_indices] = 1.0 / len(br_indices)
                
                learning_rate = lr_init * (1.0 - iteration / max_iter)
                strategy = (1 - learning_rate) * strategy + learning_rate * new_strategy
                
                strategy = strategy / np.sum(strategy)
            
            regret = calculate_max_regret(game_matrix_np, strategy)
            if regret < best_regret:
                best_strategy = strategy.copy()
                best_regret = regret
    
    if best_strategy is not None:
        strategy = best_strategy
    else:
        strategy = np.ones(n) / n
    
    if best_regret > EPSILON:
        strategy = minimize_max_regret(game_matrix_np, strategy, max_iterations=100, epsilon=EPSILON)
    
    return strategy

def replicator_dynamics_nash(game_matrix, max_iter=10000, convergence_threshold=None):
    """
    Compute Nash equilibrium using replicator dynamics for 2-player games
    
    Args:
        game_matrix: numpy array of payoffs
        max_iter: maximum iterations
        convergence_threshold: convergence criterion
        
    Returns:
        nash_strategy: equilibrium strategy (probability distribution over actions)
    """
    EPSILON = 1e-8
    if convergence_threshold is None:
        convergence_threshold = EPSILON
    
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
    num_random_points = 10
    for _ in range(num_random_points):
        random_point = np.random.random(n)
        random_point = random_point / np.sum(random_point)
        initial_points.append(random_point)
    
    # Try each initial point
    for initial_strategy in initial_points:
        # Initialize with the given strategy
        strategy = initial_strategy.copy()
        
        for iteration in range(max_iter):
            expected_payoffs = np.dot(game_matrix_np, strategy)
            
            avg_payoff = np.dot(strategy, expected_payoffs)
            
            new_strategy = strategy * (expected_payoffs / max(avg_payoff, EPSILON))
            
           
            new_strategy = new_strategy / new_strategy.sum()
            
            diff = np.linalg.norm(new_strategy - strategy)
            if diff < convergence_threshold:
                strategy = new_strategy
                break
                
            strategy = new_strategy
        
        regret = calculate_max_regret(game_matrix_np, strategy)
        if regret < best_regret:
            best_strategy = strategy.copy()
            best_regret = regret
    
    if best_strategy is not None:
        strategy = best_strategy
    else:
        strategy = np.ones(n) / n
    
    if best_regret > EPSILON:
        strategy = minimize_max_regret(game_matrix_np, strategy, max_iterations=100, epsilon=EPSILON)
    
    regret = calculate_max_regret(game_matrix_np, strategy)
    if regret > convergence_threshold:
        strategy = minimize_max_regret(game_matrix_np, strategy, max_iterations=500, learning_rate=0.2, epsilon=EPSILON)
    
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
    
    return np.max(regrets)

def minimize_max_regret(game_matrix, initial_strategy, max_iterations=100, learning_rate=0.1, epsilon=1e-8):
    """
    Refine a strategy to minimize maximum regret
    
    Args:
        game_matrix: numpy array of payoffs
        initial_strategy: initial probability distribution over actions
        max_iterations: maximum number of iterations
        learning_rate: learning rate for updates
        epsilon: numerical tolerance threshold
        
    Returns:
        refined_strategy: refined probability distribution
    """
    n = game_matrix.shape[0]
    strategy = initial_strategy.copy()
    best_strategy = strategy.copy()
    best_regret = calculate_max_regret(game_matrix, strategy)
    
    if best_regret <= epsilon:
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
            
            if max_regret <= epsilon:
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
            
            current_lr = current_lr_base * (1.0 - iteration / max_iterations)
            
            local_strategy = (1 - current_lr) * local_strategy + current_lr * update
            
            local_strategy = local_strategy / np.sum(local_strategy)
            
            current_regret = calculate_max_regret(game_matrix, local_strategy)
            if current_regret < best_regret:
                best_strategy = local_strategy.copy()
                best_regret = current_regret
        
    if best_regret > epsilon:
        expected_payoffs = np.dot(game_matrix, best_strategy)
        avg_payoff = np.dot(best_strategy, expected_payoffs)
        regrets = expected_payoffs - avg_payoff
        
        positive_idxs = np.where(regrets > epsilon)[0]
        
        for idx in positive_idxs:
            for weight in [0.001, 0.01, 0.05, 0.1, 0.2]:
                modified_strategy = best_strategy.copy()
                modified_strategy[idx] += weight
                modified_strategy = modified_strategy / np.sum(modified_strategy)
                
                modified_regret = calculate_max_regret(game_matrix, modified_strategy)
                
                if modified_regret < best_regret:
                    best_strategy = modified_strategy
                    best_regret = modified_regret
                    
                    if best_regret <= epsilon:
                        break
            
            if best_regret <= epsilon:
                break
    
    return best_strategy