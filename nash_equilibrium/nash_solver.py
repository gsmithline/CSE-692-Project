"""
Nash equilibrium solver for 2-player games.
"""

import numpy as np
from scipy.optimize import linprog, minimize
import warnings
import scipy.sparse as sp
from scipy.special import entr
import cvxpy as cp
EPSILON = .05
def _simplex_projection(x):
    """
    Project onto probability simplex.
    """
    x = np.asarray(x)
    x = x.reshape(-1)
    if (x >= 0).all() and abs(np.sum(x) - 1) < 1e-10:
        return x
    
    n = len(x)
    u = np.sort(x)[::-1]
    cssv = np.cumsum(u) - 1
    rho = np.nonzero(u * np.arange(1, n+1) > cssv)[0][-1]
    theta = cssv[rho] / (rho + 1)
    return np.maximum(x - theta, 0)

def milp_max_sym_ent_2p(game_matrix, discrete_factors=100):
    """
    Compute maximum entropy Nash equilibrium for 2-player games following the Zun's implementation.
    Uses CVXPY with ECOS_BB solver to handle boolean variables and mixed integer constraints.
    
    Args:
        game_matrix: numpy array of payoffs
        discrete_factors: number of discrete factors for entropy approximation
        
    Returns:
        nash_strategy: equilibrium strategy (probability distribution over actions)
        
    Raises:
        RuntimeError: If both ECOS_BB and GLPK_MI solvers fail to find an optimal solution
    """
    shape = game_matrix.shape
    assert len(shape) == 2
    assert shape[0] == shape[1]
    
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
    
    M = shape[0]
    U = np.max(game_matrix_np) - np.min(game_matrix_np)
    
    # Variables
    x = cp.Variable(M)
    u = cp.Variable(1)
    z = cp.Variable(M)
    b = cp.Variable(M, boolean=True)
    
    #objective: minimize sum of z (which approximates -entropy)
    obj = cp.Minimize(cp.sum(z))
    
    #nash equilibrium constraints
    a_mat = np.ones(M).reshape((1, M))
    u_m = game_matrix_np @ x
    
    constraints = [
        u_m <= u,
        a_mat @ x == 1,
        x >= 0,
        u - u_m <= U * b,
        x <= 1 - b
    ]
    
    #entropy approximation constraints
    for k in range(discrete_factors):
        if k == 0:
            constraints.append(np.log(1/discrete_factors) * x <= z)
        else:
            #linear approximation of entropy at k/discrete_factors
            slope = ((k+1)*np.log((k+1)/discrete_factors) - k*np.log(k/discrete_factors))
            intercept = k/discrete_factors * np.log(k/discrete_factors)
            constraints.append(intercept + slope * (x - k/discrete_factors) <= z)
    
    # Create and solve the problem
    prob = cp.Problem(obj, constraints)
    
    try:
        # Try ECOS_BB first
        prob.solve(solver='ECOS_BB')
        if prob.status != 'optimal':
            raise ValueError(f"ECOS_BB solver failed with status: {prob.status}")
    except Exception as e:
        warnings.warn(f"Failed to solve with ECOS_BB: {e}")
        try:
            # Try GLPK_MI as fallback
            prob.solve(solver='GLPK_MI')
            if prob.status != 'optimal':
                raise ValueError(f"GLPK_MI solver failed with status: {prob.status}")
        except Exception as e:
            raise RuntimeError(f"Both ECOS_BB and GLPK_MI solvers failed. ECOS_BB error: {e}")
    
    # Extract and project solution
    ne_strategy = _simplex_projection(x.value.reshape(-1))

    regret, nash_value, expected_utils = compute_regret(ne_strategy, game_matrix_np)
    max_regret = np.max(regret)
    if max_regret < EPSILON:
        return ne_strategy
    else:
        raise RuntimeError(f"Failed to find Nash equilibrium within {EPSILON} regret")

def calculate_max_regret(game_matrix, strategy):
    """
    Calculate the maximum regret of a strategy in a 2-player game.
    
    Args:
        game_matrix: numpy array of payoffs
        strategy: probability distribution over actions
        
    Returns:
        max_regret: maximum regret of the strategy
    """
    expected_payoffs = np.dot(game_matrix, strategy)
    max_payoff = np.max(expected_payoffs)
    return max_payoff - np.dot(strategy, expected_payoffs)

def minimize_max_regret(game_matrix, strategy, max_iterations=100, epsilon=EPSILON):
    """
    Minimize the maximum regret of a strategy using gradient descent.
    
    Args:
        game_matrix: numpy array of payoffs
        strategy: initial probability distribution over actions
        max_iterations: maximum number of iterations
        epsilon: convergence threshold
        
    Returns:
        strategy: improved probability distribution over actions
    """
    n = game_matrix.shape[0]
    current_strategy = strategy.copy()
    
    for _ in range(max_iterations):
        expected_payoffs = np.dot(game_matrix, current_strategy)
        max_payoff = np.max(expected_payoffs)
        regret = max_payoff - np.dot(current_strategy, expected_payoffs)
        
        if regret < epsilon:
            break
            
        # Calculate gradient of regret
        gradient = np.zeros(n)
        for i in range(n):
            gradient[i] = game_matrix[i, i] - np.dot(current_strategy, game_matrix[i, :])
        
        # Project gradient onto simplex
        gradient = gradient - np.mean(gradient)
        gradient = np.maximum(gradient, 0)
        
        # Update strategy
        step_size = 0.1
        new_strategy = current_strategy + step_size * gradient
        new_strategy = new_strategy / np.sum(new_strategy)
        
        current_strategy = new_strategy
    
    return current_strategy

def replicator_dynamics_nash(game_matrix, max_iter=2000, epsilon=0.05, step_size=0.5, return_trace=False):
    """
    Compute Nash equilibrium using improved replicator dynamics with adaptive step size.
    
    Args:
        game_matrix: numpy array of payoffs
        max_iter: maximum iterations (default 2000)
        epsilon: convergence threshold
        step_size: initial step size for updates
        return_trace: whether to return the trace of strategies and regrets
        
    Returns:
        nash_strategy: equilibrium strategy (probability distribution over actions)
        trace_data: dict with 'strategies' and 'regrets' arrays if return_trace=True
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
    
    # Normalize payoffs to prevent overflow
    payoff_range = np.max(game_matrix_np) - np.min(game_matrix_np)
    if payoff_range > 20:  # If range is large, normalize to prevent exp overflow
        game_matrix_np = (game_matrix_np - np.min(game_matrix_np)) / payoff_range * 10
    
    n = game_matrix_np.shape[0]
    
    best_strategy = None
    best_regret = float('inf')
    best_trace_strategies = []
    best_trace_regrets = []
    best_trace_step_sizes = []
    
    # Try multiple starting points
    initial_points = [
        np.ones(n) / n,  # Uniform distribution
    ]
    
    # Add pure strategies as starting points
    for i in range(n):
        pure_strategy = np.zeros(n)
        pure_strategy[i] = 1.0
        initial_points.append(pure_strategy)
    
    # Add more random starting points
    num_random_points = 10
    for _ in range(num_random_points):
        random_point = np.random.random(n)
        random_point = random_point / np.sum(random_point)
        initial_points.append(random_point)
    
    for initial_idx, initial_strategy in enumerate(initial_points):
        strategy = initial_strategy.copy()
        trace_strategies = [strategy.copy()]
        trace_regrets = []
        trace_step_sizes = [step_size]
        
        current_step_size = step_size
        min_step_size = 1e-4
        max_step_size = 1.0
        
        try:
            for iteration in range(max_iter):
                # Calculate expected payoffs
                expected_payoffs = np.dot(game_matrix_np, strategy)
                
                # Calculate average payoff
                avg_payoff = np.dot(strategy, expected_payoffs)
                
                # Calculate the regret
                regret = np.max(expected_payoffs) - avg_payoff
                trace_regrets.append(regret)
                
                # Check early convergence
                if regret < epsilon:
                    if regret < best_regret:
                        best_strategy = strategy.copy()
                        best_regret = regret
                        if return_trace:
                            best_trace_strategies = trace_strategies
                            best_trace_regrets = trace_regrets
                            best_trace_step_sizes = trace_step_sizes
                    break
                
                # Standard replicator dynamics update with step size
                new_strategy = strategy * (1 + current_step_size * (expected_payoffs - avg_payoff))
                
                # Handle potential negative values and normalize
                new_strategy = np.maximum(new_strategy, 0)
                sum_strategy = np.sum(new_strategy)
                
                # Check for numerical issues
                if sum_strategy <= 0 or np.isnan(sum_strategy):
                    # Fallback to a more stable update rule
                    payoff_diff = expected_payoffs - avg_payoff
                    # Limit extreme values to prevent overflow
                    payoff_diff = np.clip(payoff_diff, -10, 10)
                    # Linear update instead of exponential
                    new_strategy = strategy + current_step_size * strategy * payoff_diff
                    new_strategy = np.maximum(new_strategy, 0)
                    sum_strategy = np.sum(new_strategy)
                    
                # Ensure proper normalization
                if sum_strategy > 0:
                    new_strategy = new_strategy / sum_strategy
                else:
                    # If we still have numerical issues, reset to uniform
                    new_strategy = np.ones(n) / n
                
                # Calculate strategy change
                strategy_change = np.max(np.abs(new_strategy - strategy))
                
                # Check convergence based on strategy change or regret
                if strategy_change < epsilon or regret < epsilon:
                    strategy = new_strategy
                    
                    # Calculate final regret
                    final_expected_payoffs = np.dot(game_matrix_np, strategy)
                    final_avg_payoff = np.dot(strategy, final_expected_payoffs)
                    final_regret = np.max(final_expected_payoffs) - final_avg_payoff
                    trace_regrets.append(final_regret)
                    trace_strategies.append(strategy.copy())
                    trace_step_sizes.append(current_step_size)
                    
                    if final_regret < best_regret:
                        best_strategy = strategy.copy()
                        best_regret = final_regret
                        if return_trace:
                            best_trace_strategies = trace_strategies
                            best_trace_regrets = trace_regrets
                            best_trace_step_sizes = trace_step_sizes
                    break
                
                # Adaptive step size adjustment
                if strategy_change > 0.5:  # Large change - reduce step size
                    current_step_size = max(min_step_size, current_step_size * 0.8)
                elif strategy_change < 0.001:  # Small change - increase step size
                    current_step_size = min(max_step_size, current_step_size * 1.2)
                    
                strategy = new_strategy
                trace_strategies.append(strategy.copy())
                trace_step_sizes.append(current_step_size)
            
            # If we didn't break early, we reached max iterations
            if iteration == max_iter - 1:
                # Calculate final regret
                final_expected_payoffs = np.dot(game_matrix_np, strategy)
                final_avg_payoff = np.dot(strategy, final_expected_payoffs)
                final_regret = np.max(final_expected_payoffs) - final_avg_payoff
                
                if final_regret < best_regret:
                    best_strategy = strategy.copy()
                    best_regret = final_regret
                    if return_trace:
                        best_trace_strategies = trace_strategies
                        best_trace_regrets = trace_regrets
                        best_trace_step_sizes = trace_step_sizes
                
        except Exception as e:
            # Skip this starting point if we encounter errors
            print(f"Error with starting point {initial_idx}: {e}")
            continue
    
    if best_strategy is None:
        raise RuntimeError("Failed to find Nash equilibrium")
        
    best_strategy = _simplex_projection(best_strategy)
    regret, nash_value, expected_utils = compute_regret(best_strategy, game_matrix_np)
    if np.max(regret) <= epsilon:
        if return_trace:
            return best_strategy, {
                'strategies': best_trace_strategies,
                'regrets': best_trace_regrets,
                'step_sizes': best_trace_step_sizes,
                'final_regret': np.max(regret),
                'nash_value': nash_value
            }
        else:
            return best_strategy
    else:
        print(f"Best regret found: {np.max(regret):.6f}")
        if return_trace:
            return best_strategy, {
                'strategies': best_trace_strategies,
                'regrets': best_trace_regrets,
                'step_sizes': best_trace_step_sizes,
                'final_regret': np.max(regret),
                'nash_value': nash_value
            }
        else:
            return best_strategy

def compute_regret(strategy, payoff_matrix):
    """
    Compute the regret for each agent given a strategy.
    
    Args:
        strategy: Nash equilibrium strategy vector
        payoff_matrix: Game payoff matrix
        
    Returns:
        regret: Vector of regrets for each agent
        nash_value: Expected utility when Nash plays against itself
    """
    # Expected utilities when playing against the Nash strategy
    expected_utils = payoff_matrix @ strategy
    
    # Nash equilibrium value (expected utility when Nash plays against itself)
    nash_value = strategy.T @ payoff_matrix @ strategy
    
    # Calculate regret for each pure strategy
    regret = expected_utils - nash_value
    
    return regret, nash_value, expected_utils
