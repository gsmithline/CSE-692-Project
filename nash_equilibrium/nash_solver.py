"""
Nash equilibrium solver for 2-player games.
"""

import numpy as np
from scipy.optimize import linprog

def milp_max_sym_ent_2p(game_matrix, max_iter=100):
    """
    Compute maximum entropy Nash equilibrium for 2-player games
    
    This is a simplified version for demonstration. For complete implementation, 
    import the full solver from your solutions module.
    
    Args:
        game_matrix: numpy array of payoffs
        max_iter: maximum iterations
        
    Returns:
        nash_strategy: equilibrium strategy (probability distribution over actions)
    """
    # Ensure the game matrix is a properly formatted numpy array of floats
    game_matrix_np = np.array(game_matrix, dtype=np.float64)
    
    # Check for and handle NaN values
    if np.isnan(game_matrix_np).any():
        # Replace NaN with column means or 0 if all NaN
        for j in range(game_matrix_np.shape[1]):
            col = game_matrix_np[:, j]
            if np.isnan(col).any():
                col_mean = np.nanmean(col)
                if np.isnan(col_mean):
                    col_mean = 0
                game_matrix_np[np.isnan(col), j] = col_mean
    
    n = game_matrix_np.shape[0]
    
    strategy = np.ones(n) / n
    
    for _ in range(max_iter):
        br_payoff = game_matrix_np @ strategy
        max_payoff = np.max(br_payoff)
        
        epsilon = max_payoff - np.dot(br_payoff, strategy)
        
        if epsilon < 1e-8:
            break
            
        # Find the distribution of best responses
        # Use a tolerance to find indices that are close to max_payoff
        tol = 1e-8
        br_indices = np.where(np.abs(br_payoff - max_payoff) < tol)[0]
        
        if len(br_indices) == 1:
            new_strategy = np.zeros(n)
            new_strategy[br_indices[0]] = 1.0
        else:
            new_strategy = np.zeros(n)
            new_strategy[br_indices] = 1.0 / len(br_indices)
            
        strategy = 0.5 * strategy + 0.5 * new_strategy
        
    return strategy