"""
Nash equilibrium solver for 2-player games.
"""

import numpy as np
from scipy.optimize import linprog, minimize
import warnings
import scipy.sparse as sp
from scipy.special import entr
import cvxpy as cp
import nashpy as nash
try:
    import open_spiel.python.egt.alpharank
    import open_spiel.python.egt.alpharank_visualizer
except ImportError:
    print("Warning: OpenSpiel not found in path. Some functionality will be limited.")
    # Define dummy modules or alternative implementations if needed
EPSILON = 1e-6
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
    
    
    x = cp.Variable(M)
    u = cp.Variable(1)
    z = cp.Variable(M)
    b = cp.Variable(M, boolean=True)
    #r = cp.Variable(1, nonneg=True)
    
    #objective is minimize sum of z (which approximates -entropy)
    obj = cp.Minimize(cp.sum(z))
    
    #nash equilibrium constraints
    a_mat = np.ones(M).reshape((1, M))
    u_m = game_matrix_np @ x
    
    constraints = [
        u_m <= u + EPSILON, #this makes it epsilon ME NE
        a_mat @ x == 1,
        x >= 0,
        u - u_m <= U * b,
        x <= 1 - b,
        #r <= 1,
        #r >= u - u_m
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
    
    prob = cp.Problem(obj, constraints)
    
    try:
        prob.solve(solver='ECOS_BB')
        if prob.status != 'optimal':
            raise ValueError(f"ECOS_BB solver failed with status: {prob.status}")
    except Exception as e:
        warnings.warn(f"Failed to solve with ECOS_BB: {e}")
        try:
            prob.solve(solver='GLPK_MI')
            if prob.status != 'optimal':
                raise ValueError(f"GLPK_MI solver failed with status: {prob.status}")
        except Exception as e:
            raise RuntimeError(f"Both ECOS_BB and GLPK_MI solvers failed. ECOS_BB error: {e}")
    
    ne_strategy = _simplex_projection(x.value.reshape(-1))

    regret, nash_value, expected_utils = compute_regret(ne_strategy, game_matrix_np)
    max_regret = np.max(regret)
    if max_regret <= EPSILON:
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
            
        gradient = np.zeros(n)
        for i in range(n):
            gradient[i] = game_matrix[i, i] - np.dot(current_strategy, game_matrix[i, :])
        
        gradient = gradient - np.mean(gradient)
        gradient = np.maximum(gradient, 0)
        
        step_size = 0.1
        new_strategy = current_strategy + step_size * gradient
        new_strategy = new_strategy / np.sum(new_strategy)
        
        current_strategy = new_strategy
    
    return current_strategy

def replicator_dynamics_nash(game_matrix, max_iter=2000, epsilon=EPSILON, step_size=0.5, return_trace=False):
    """
    Compute Nash equilibrium using improved replicator dynamics with adaptive step size.
    Compares results from custom implementation and nashpy implementation.
    Estimates basins of attraction for different equilibria.
    
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
    
    n = game_matrix_np.shape[0]
    
    # Generate initial points (greatly increased for better basin estimation)
    initial_points = [np.ones(n) / n]  # Uniform strategy
    
    # Add pure strategies
    for i in range(n):
        pure_strategy = np.zeros(n)
        pure_strategy[i] = 1.0
        initial_points.append(pure_strategy)
    
    # Add random points for better basin estimation
    num_random_points = 50  # Increased from 10 to 50
    for _ in range(num_random_points):
        random_point = np.random.random(n)
        random_point = random_point / np.sum(random_point)
        initial_points.append(random_point)
    
    # Functions to help with basin estimation
    def strategy_distance(s1, s2):
        """Calculate L1 distance between strategies"""
        return np.sum(np.abs(s1 - s2))
    
    def is_same_equilibrium(s1, s2, threshold=0.3):
        """Check if two strategies represent the same equilibrium"""
        return strategy_distance(s1, s2) <= threshold
    
    # Run nashpy implementation
    game = nash.Game(game_matrix_np)
    nashpy_equilibria = []
    nashpy_regrets = []
    
    for initial_idx, initial_strategy in enumerate(initial_points):
        try:
            initial_state = _simplex_projection(initial_strategy)
            timestamps = np.linspace(0, 100, max_iter)
            
            states = list(game.replicator_dynamics(initial_state, timestamps))
            
            if states:
                strategy = states[-1]
                strategy = _simplex_projection(strategy)
                regret, nash_value, _ = compute_regret(strategy, game_matrix_np)
                max_regret = np.max(regret)
                
                # Store this equilibrium if it's good enough
                if max_regret <= epsilon:
                    # Check if this is a new equilibrium or one we've seen before
                    is_new = True
                    for idx, existing_eq in enumerate(nashpy_equilibria):
                        if is_same_equilibrium(strategy, existing_eq):
                            is_new = False
                            # If this one has better regret, replace the old one
                            if max_regret < nashpy_regrets[idx]:
                                nashpy_equilibria[idx] = strategy.copy()
                                nashpy_regrets[idx] = max_regret
                            break
                    
                    if is_new:
                        nashpy_equilibria.append(strategy.copy())
                        nashpy_regrets.append(max_regret)
        except Exception as e:
            continue
    
    # Run custom implementation
    custom_equilibria = []
    custom_regrets = []
    equilibrium_basins = []  # Count of initial points converging to each equilibrium
    
    for initial_idx, initial_strategy in enumerate(initial_points):
        strategy = initial_strategy.copy()
        
        current_step_size = step_size
        min_step_size = 1e-2
        max_step_size = 10
        
        try:
            for iteration in range(max_iter):
                expected_payoffs = np.dot(game_matrix_np, strategy)
                avg_payoff = np.dot(strategy, expected_payoffs)
                
                regret_vector, nash_value, _ = compute_regret(strategy, game_matrix_np)
                current_regret = np.max(regret_vector)
                
                if current_regret < epsilon:
                    # We found an equilibrium, check if it's new or one we've seen
                    is_new = True
                    found_idx = -1
                    
                    for idx, existing_eq in enumerate(custom_equilibria):
                        if is_same_equilibrium(strategy, existing_eq):
                            is_new = False
                            found_idx = idx
                            # If this one has better regret, replace the old one
                            if current_regret < custom_regrets[idx]:
                                custom_equilibria[idx] = strategy.copy()
                                custom_regrets[idx] = current_regret
                            break
                    
                    if is_new:
                        custom_equilibria.append(strategy.copy())
                        custom_regrets.append(current_regret)
                        equilibrium_basins.append(1)
                    else:
                        equilibrium_basins[found_idx] += 1
                    
                    break
                
                # Replicator dynamics update
                new_strategy = strategy * (1 + current_step_size * (expected_payoffs - avg_payoff))
                new_strategy = np.maximum(new_strategy, 0)
                new_strategy = _simplex_projection(new_strategy)
                
                strategy_change = np.max(np.abs(new_strategy - strategy))
                
                new_regret_vector, _, _ = compute_regret(new_strategy, game_matrix_np)
                new_regret = np.max(new_regret_vector)
                
                # If regret got worse, try adding some noise
                if new_regret > current_regret or new_regret > current_regret * 0.99:
                    noise_factor = np.random.uniform(1e-2, 1)
                    payoff_diff = expected_payoffs - avg_payoff
                    new_strategy = strategy * (1 + current_step_size + noise_factor * payoff_diff)
                    new_strategy = _simplex_projection(new_strategy)
                    
                    new_regret_vector, _, _ = compute_regret(new_strategy, game_matrix_np)
                    new_regret = np.max(new_regret_vector)
                
                # Adaptive step size adjustment
                if strategy_change > 0.5:  
                    current_step_size = max(min_step_size, current_step_size * 0.8)
                elif strategy_change < 0.001:  
                    current_step_size = min(max_step_size, current_step_size * 1.2)
                    
                strategy = new_strategy.copy()
            
            # If we reached max iterations without convergence
            if iteration == max_iter - 1:
                final_regret_vector, _, _ = compute_regret(strategy, game_matrix_np)
                final_regret = np.max(final_regret_vector)
                
                if final_regret < epsilon:
                    # Check if it's a new equilibrium
                    is_new = True
                    found_idx = -1
                    
                    for idx, existing_eq in enumerate(custom_equilibria):
                        if is_same_equilibrium(strategy, existing_eq):
                            is_new = False
                            found_idx = idx
                            if final_regret < custom_regrets[idx]:
                                custom_equilibria[idx] = strategy.copy()
                                custom_regrets[idx] = final_regret
                            break
                    
                    if is_new:
                        custom_equilibria.append(strategy.copy())
                        custom_regrets.append(final_regret)
                        equilibrium_basins.append(1)
                    else:
                        equilibrium_basins[found_idx] += 1
                
        except Exception as e:
            continue
    
    # Combine nashpy and custom equilibria
    combined_equilibria = []
    combined_regrets = []
    combined_basins = []
    
    # First add all custom equilibria
    for i, eq in enumerate(custom_equilibria):
        combined_equilibria.append(eq)
        combined_regrets.append(custom_regrets[i])
        combined_basins.append(equilibrium_basins[i])
    
    # Then add nashpy equilibria if they're distinct
    for i, eq in enumerate(nashpy_equilibria):
        is_new = True
        for existing_eq in combined_equilibria:
            if is_same_equilibrium(eq, existing_eq):
                is_new = False
                break
        
        if is_new:
            combined_equilibria.append(eq)
            combined_regrets.append(nashpy_regrets[i])
            combined_basins.append(1)   
    
    if len(combined_equilibria) == 0: 
        best_strategy = milp_nash_2p(game_matrix_np, epsilon)
    else:
        basin_sizes = np.array(combined_basins)
        max_basin_size = np.max(basin_sizes)
        
        largest_basin_indices = np.where(basin_sizes == max_basin_size)[0]
        
        if len(largest_basin_indices) == 1:
            best_idx = largest_basin_indices[0]
        else:
            min_regret = float('inf')
            best_idx = -1
            
            for idx in largest_basin_indices:
                if combined_regrets[idx] < min_regret:
                    min_regret = combined_regrets[idx]
                    best_idx = idx
        
        best_strategy = combined_equilibria[best_idx]
        print(f"Selected equilibrium with basin size {combined_basins[best_idx]} and regret {combined_regrets[best_idx]:.6f}")
    
    best_strategy = _simplex_projection(best_strategy)
    regret, nash_value, expected_utils = compute_regret(best_strategy, game_matrix_np)
    
    if np.max(regret) <= epsilon:
        pass
    else:
        print("Falling back to plain MILP")
        best_strategy = milp_nash_2p(game_matrix_np, epsilon)
        regret, nash_value, expected_utils = compute_regret(best_strategy, game_matrix_np)
        
    if return_trace: #TODO: not really using here
        return best_strategy, {
            'strategies': [],  
            'regrets': [],
            'step_sizes': [],
            'final_regret': np.max(regret),
            'nash_value': nash_value,
            'basins': combined_basins if len(combined_equilibria) > 0 else []
        }
    else:
        return np.array(best_strategy), len(initial_points)

def compute_regret(strategy, payoff_matrix):
    
    expected_utils = payoff_matrix @ strategy
    
    nash_value = strategy.T @ payoff_matrix @ strategy
    
    regret = expected_utils - nash_value
    
    return regret, nash_value, expected_utils

def milp_nash_2p(game_matrix, regret_threshold=EPSILON):
    
    shape = game_matrix.shape
    assert len(shape) == 2
    assert shape[0] == shape[1], "Game matrix must be square for a symmetric game"
    
    game_matrix_np = np.array(game_matrix, dtype=np.float64)
    
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
    
    x = cp.Variable(M)  
    u = cp.Variable(1)  
    b = cp.Variable(M, boolean=True)  
    
    obj = cp.Minimize(cp.sum(b))
    
    a_mat = np.ones(M).reshape((1, M))
    u_m = game_matrix_np @ x
    
    constraints = [
        u_m <= u,               # Expected payoff for each pure strategy <= threshold
        a_mat @ x == 1,         # Strategy must be a probability distribution
        x >= 0,                 # Probabilities must be non-negative
        u - u_m <= U * b,       # If b[i]=0, strategy i must achieve the max payoff
        x <= 1 - b              # If b[i]=0, x[i] can be positive
    ]
    
    prob = cp.Problem(obj, constraints)
    
    try:
        prob.solve(solver='ECOS_BB')
        if prob.status != 'optimal':
            raise ValueError(f"ECOS_BB solver failed with status: {prob.status}")
    except Exception as e:
        warnings.warn(f"Failed to solve with ECOS_BB: {e}")
        try:
            prob.solve(solver='GLPK_MI')
            if prob.status != 'optimal':
                raise ValueError(f"GLPK_MI solver failed with status: {prob.status}")
        except Exception as e:
            raise RuntimeError(f"Both ECOS_BB and GLPK_MI solvers failed. ECOS_BB error: {e}")
    
    ne_strategy = _simplex_projection(x.value.reshape(-1))

    regret, nash_value, expected_utils = compute_regret(ne_strategy, game_matrix_np)
    max_regret = np.max(regret)
    
    if max_regret < regret_threshold:
        return ne_strategy
    else:
        rd_strategy, _ = replicator_dynamics_nash(game_matrix_np, epsilon=regret_threshold)
        return rd_strategy