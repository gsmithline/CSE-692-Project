"""
Nash equilibrium solver for 2-player games.
"""

import numpy as np
from scipy.optimize import linprog, minimize
import warnings
import scipy.sparse as sp
from scipy.special import entr
import cvxpy as cp
EPSILON = 0.05
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

def milp_max_sym_ent_2p(game_matrix, discrete_factors=20):
    """
    Compute maximum entropy Nash equilibrium for 2-player games following the paper's implementation.
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
    
    # Objective: minimize sum of z (which approximates -entropy)
    obj = cp.Minimize(cp.sum(z))
    
    # Nash equilibrium constraints
    a_mat = np.ones(M).reshape((1, M))
    u_m = game_matrix_np @ x
    
    constraints = [
        u_m <= u,
        a_mat @ x == 1,
        x >= 0,
        u - u_m <= U * b,
        x <= 1 - b
    ]
    
    # Entropy approximation constraints
    for k in range(discrete_factors):
        if k == 0:
            constraints.append(np.log(1/discrete_factors) * x <= z)
        else:
            # Linear approximation of entropy at k/discrete_factors
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
    return _simplex_projection(x.value.reshape(-1))

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

def replicator_dynamics_nash(game_matrix, max_iter=10000, convergence_threshold=None):
    """
    Compute Nash equilibrium using replicator dynamics for 2-player games.
    
    Args:
        game_matrix: numpy array of payoffs
        max_iter: maximum iterations
        convergence_threshold: convergence criterion
        
    Returns:
        nash_strategy: equilibrium strategy (probability distribution over actions)
    """
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
        np.ones(n) / n,
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
            
            # Project back to simplex
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
    
    return best_strategy