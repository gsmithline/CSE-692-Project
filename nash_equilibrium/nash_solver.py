"""
Nash equilibrium solver for 2-player games.
"""

import numpy as np
from scipy.optimize import linprog

def milp_max_sym_ent_2p(game_matrix, max_iter=100):
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
    
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        
        if result.success:
            strategy = result.x[:-1]
            strategy = strategy / np.sum(strategy)
            return strategy
    except:
        pass
    
    strategy = np.ones(n) / n 
    
    for _ in range(max_iter):
        br_payoff = game_matrix_np @ strategy  
        max_payoff = np.max(br_payoff)  
        
        epsilon = max_payoff - np.dot(br_payoff, strategy)
        
        if epsilon < 1e-8:
            break
            
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