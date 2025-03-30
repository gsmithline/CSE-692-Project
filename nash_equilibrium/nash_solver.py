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
    Compares results from custom implementation and nashpy implementation.
    
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
    
    # Create multiple starting points for both methods
    initial_points = [
        np.ones(n) / n,  # Uniform distribution
    ]
    
    # Add pure strategies as starting points
    for i in range(n):
        pure_strategy = np.zeros(n)
        pure_strategy[i] = 1.0
        initial_points.append(pure_strategy)
    
    # Add more random starting points
    game_matrix_np_2 = game_matrix_np.copy()
    num_random_points = 10
    for _ in range(num_random_points):
        random_point = np.random.random(n)
        random_point = random_point / np.sum(random_point)
        initial_points.append(random_point)
    
    # PART 1: Run nashpy implementation
    print("Running nashpy replicator dynamics implementation...")
    payoff_range = np.max(game_matrix_np) - np.min(game_matrix_np)
   
    #if payoff_range > 20: 
        #game_matrix_np = (game_matrix_np - np.min(game_matrix_np)) / payoff_range * 10
    game = nash.Game(game_matrix_np)
    nashpy_best_strategy = None
    nashpy_best_regret = float('inf')
    
    for initial_idx, initial_strategy in enumerate(initial_points):
        try:
            # Create initial state (x, y) for the game
            # For symmetric games, both players use the same strategy
            initial_state = initial_strategy 
            initial_state = _simplex_projection(initial_state)
            
            # Get the dynamics with timestamps and states
            timestamps = np.linspace(0, 100, max_iter)  # Create time points
            
            # Call replicator dynamics correctly
            states = list(game.replicator_dynamics(initial_state, timestamps))

            for state in states:
                strategy = state
                # Check for convergence by calculating regret
                strategy = _simplex_projection(strategy)
                regret, nash_value, expected_utils = compute_regret(strategy, game_matrix_np)
                max_regret = np.max(regret)
                
                if max_regret < nashpy_best_regret:
                    nashpy_best_strategy = strategy.copy()
                    nashpy_best_regret = max_regret
                
                    if max_regret <= epsilon:
                        print(f"  Nashpy found epsilon-Nash with regret {max_regret:.6f} at start point {initial_idx}")
                        break
            
            # Extract the final state
            '''
            if len(states) > 0:
                strategy = nashpy_best_strategy
                
                # Check for convergence by calculating regret
                regret, nash_value, expected_utils = compute_regret(strategy, game_matrix_np)
                max_regret = np.max(regret)
                
                print(f"  Nashpy finished with regret {max_regret:.6f} at start point {initial_idx}")
                if max_regret < nashpy_best_regret:
                    nashpy_best_strategy = strategy.copy()
                    nashpy_best_regret = max_regret
            '''
    
        except Exception as e:
            print(f"Error with nashpy at starting point {initial_idx}: {e}")
            continue
    
    # PART 2: Run custom implementation
    print("Running custom replicator dynamics implementation...")
    custom_best_strategy = None
    custom_best_regret = float('inf')
    best_trace_strategies = []
    best_trace_regrets = []
    best_trace_step_sizes = []
    
    for initial_idx, initial_strategy in enumerate(initial_points):
        strategy = initial_strategy.copy()
        trace_strategies = [strategy.copy()]
        trace_regrets = []
        trace_step_sizes = [step_size]
        
        current_step_size = step_size
        min_step_size = 1e-2
        max_step_size = 10
        
        try:
            for iteration in range(max_iter):
                # Calculate expected payoffs
                expected_payoffs = np.dot(game_matrix_np, strategy)
                
                # Calculate average payoff
                avg_payoff = np.dot(strategy, expected_payoffs)
                
                # Calculate the regret
                regret_vector, nash_value, _ = compute_regret(strategy, game_matrix_np)
                current_regret = np.max(regret_vector)
                trace_regrets.append(current_regret)
                
                # Check early convergence
                if current_regret < epsilon:
                    if current_regret < custom_best_regret:
                        custom_best_strategy = strategy.copy()
                        custom_best_regret = current_regret
                        if return_trace:
                            best_trace_strategies = trace_strategies
                            best_trace_regrets = trace_regrets
                            best_trace_step_sizes = trace_step_sizes
                    print(f"  Custom found epsilon-Nash with regret {current_regret:.6f} at start point {initial_idx}")
                    break
                
                # Standard replicator dynamics update with step size
                new_strategy = strategy * (1 + current_step_size * (expected_payoffs - avg_payoff))
                
                # Handle potential negative values and normalize
                new_strategy = np.maximum(new_strategy, 0)
                new_strategy = _simplex_projection(new_strategy)
                
                # Calculate strategy change
                strategy_change = np.max(np.abs(new_strategy - strategy))
                
                # Check if the update improved regret
                new_regret_vector, _, _ = compute_regret(new_strategy, game_matrix_np)
                new_regret = np.max(new_regret_vector)
                
                # Add noise to escape local minima if regret didn't improve enough
                if new_regret > current_regret or new_regret > current_regret * 0.99:
                    # Try a different update with small random noise
                    noise_factor = np.random.uniform(1e-2, 1)
                    payoff_diff = expected_payoffs - avg_payoff
                    # Clip extreme values to prevent numerical issues
                    #payoff_diff = np.clip(payoff_diff, -10, 10)
                    new_strategy = strategy * (1 + current_step_size + noise_factor * payoff_diff)
                    new_strategy = _simplex_projection(new_strategy)
                    
                    # Recalculate regret with the noisy update
                    new_regret_vector, _, _ = compute_regret(new_strategy, game_matrix_np)
                    new_regret = np.max(new_regret_vector)
                
                # Track best strategy found so far
                if new_regret < custom_best_regret:
                    custom_best_strategy = new_strategy.copy()
                    custom_best_regret = new_regret
                    
                    # Stop early if we found an epsilon-Nash
                    if new_regret <= epsilon:
                        strategy = new_strategy.copy()
                        trace_regrets.append(new_regret)
                        trace_strategies.append(strategy.copy())
                        trace_step_sizes.append(current_step_size)
                        print(f"  Custom found epsilon-Nash with regret {new_regret:.6f} at start point {initial_idx}")
                        break
                
                # Adaptive step size adjustment
                if strategy_change > 0.5:  # Large change - reduce step size
                    current_step_size = max(min_step_size, current_step_size * 0.8)
                elif strategy_change < 0.001:  # Small change - increase step size
                    current_step_size = min(max_step_size, current_step_size * 1.2)
                    
                strategy = new_strategy.copy()
                trace_strategies.append(strategy.copy())
                trace_step_sizes.append(current_step_size)
            
            # End of loop - make sure we've updated our best strategy
            if iteration == max_iter - 1:
                final_regret_vector, _, _ = compute_regret(strategy, game_matrix_np)
                final_regret = np.max(final_regret_vector)
                
                if final_regret < custom_best_regret:
                    custom_best_strategy = strategy.copy()
                    custom_best_regret = final_regret
                    
                print(f"  Custom reached max iterations with regret {final_regret:.6f} at start point {initial_idx}")
                
                if return_trace:
                    best_trace_strategies = trace_strategies
                    best_trace_regrets = trace_regrets
                    best_trace_step_sizes = trace_step_sizes
                
        except Exception as e:
            # Skip this starting point if we encounter errors
            print(f"Error with custom implementation at starting point {initial_idx}: {e}")
            continue
    
    # PART 3: Compare and return the best result
    print(f"Best nashpy regret: {nashpy_best_regret:.6f}")
    print(f"Best custom regret: {custom_best_regret:.6f}")
    
    if nashpy_best_strategy is None and custom_best_strategy is None:
        raise RuntimeError("Failed to find Nash equilibrium with either method")
    
    # Choose the best strategy between the two methods
    if nashpy_best_strategy is None or (custom_best_strategy is not None and custom_best_regret < nashpy_best_regret):
        best_strategy = custom_best_strategy
        best_regret = custom_best_regret
        print("Using custom implementation result (better regret)")
    else:
        best_strategy = nashpy_best_strategy
        best_regret = nashpy_best_regret
        print("Using nashpy implementation result (better regret)")
        
    best_strategy = _simplex_projection(best_strategy)
    regret, nash_value, expected_utils = compute_regret(best_strategy, game_matrix_np)


    
    if np.max(regret) <= epsilon:
        print(f"Found epsilon-Nash equilibrium with regret {np.max(regret):.6f}")
    else:
        print(f"Best regret found: {np.max(regret):.6f} (above epsilon threshold)")
        
    if return_trace:
        return best_strategy, {
            'strategies': best_trace_strategies,
            'regrets': best_trace_regrets,
            'step_sizes': best_trace_step_sizes,
            'final_regret': np.max(regret),
            'nash_value': nash_value
        }
    else:
        # Return best strategy (make sure it's a proper numpy array)
        return np.array(best_strategy), iteration

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



def milp_sym_ent_2p(game_matrix, discrete_factors=100):
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

def milp_nash_2p(game_matrix, regret_threshold=EPSILON):
    """
    Compute a Nash equilibrium for 2-player symmetric games using MILP.
    Unlike milp_max_sym_ent_2p, this doesn't maximize entropy but just finds any Nash equilibrium.
    
    Args:
        game_matrix: numpy array of payoffs
        regret_threshold: threshold for considering a strategy to be a Nash equilibrium
        
    Returns:
        nash_strategy: equilibrium strategy (probability distribution over actions)
        
    Raises:
        RuntimeError: If solvers fail to find a Nash equilibrium
    """
    shape = game_matrix.shape
    assert len(shape) == 2
    assert shape[0] == shape[1], "Game matrix must be square for a symmetric game"
    
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
    x = cp.Variable(M)  # Strategy
    u = cp.Variable(1)  # Utility threshold
    b = cp.Variable(M, boolean=True)  # Boolean variables for Nash conditions
    
    # Objective: find any Nash equilibrium (minimize 0 means we just want a feasible solution)
    # Alternative could be to minimize the sum of boolean variables to get a more "pure" strategy
    obj = cp.Minimize(cp.sum(b))
    
    # Nash equilibrium constraints
    a_mat = np.ones(M).reshape((1, M))
    u_m = game_matrix_np @ x
    
    constraints = [
        u_m <= u,               # Expected payoff for each pure strategy <= threshold
        a_mat @ x == 1,         # Strategy must be a probability distribution
        x >= 0,                 # Probabilities must be non-negative
        u - u_m <= U * b,       # If b[i]=0, strategy i must achieve the max payoff
        x <= 1 - b              # If b[i]=0, x[i] can be positive
    ]
    
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

    # Verify it's a Nash equilibrium by computing regret
    regret, nash_value, expected_utils = compute_regret(ne_strategy, game_matrix_np)
    max_regret = np.max(regret)
    
    if max_regret < regret_threshold:
        return ne_strategy
    else:
        # If regret is too high, try the replicator dynamics approach as a fallback
        print(f"MILP solution has high regret ({max_regret:.6f}). Trying replicator dynamics instead.")
        rd_strategy, _ = replicator_dynamics_nash(game_matrix_np, epsilon=regret_threshold)
        return rd_strategy