from copy import deepcopy
import numpy as np
import cvxpy as cp
from open_spiel.python.algorithms.projected_replicator_dynamics import _simplex_projection
# N = 5

# num_strategies_list = [2, 3, 4, 2, 5]


# epsilon = 0.0

# assert N == len(num_strategies_list)

# game_matrix =[np.random.uniform(size=num_strategies_list) for _ in range(N)]


def solve_optimal_cce(game, epsilon=1e-8, objective='MAX_GINI', solver='GUROBI'):
  N = len(game)
  num_strategies_list = game[0].shape
  strategies = cp.Variable(np.prod(num_strategies_list))
  constraints = []
  constraints.append(cp.sum(strategies) == 1)
  constraints.append(strategies >= 0)

  for n in range(N):
    expected_payoff = cp.sum(game[n].reshape(-1) @ strategies)
    for a_n in range(num_strategies_list[n]):
      indices = tuple([a_n if n_ == n else slice(None) for n_ in range(N)])
      deviation_payoff = np.copy(game[n])
      for a_nn in range(num_strategies_list[n]):
        indices_ = tuple([a_nn if n_ == n else slice(None) for n_ in range(N)])
        deviation_payoff[indices_] = game[n][indices]
      constraints.append(cp.sum(deviation_payoff.reshape(-1)
                         @ strategies)-expected_payoff <= epsilon)

  obj = OBJECTIVE[objective](game, strategies)
  prob = cp.Problem(obj, constraints)
  prob.solve(solver=solver)

  return strategies.value


def solve_optimal_ce(game, epsilon=1e-8, objective='MAX_GINI', solver='GUROBI'):
  N = len(game)
  num_strategies_list = game[0].shape
  strategies = cp.Variable(np.prod(num_strategies_list))
  constraints = []
  constraints.append(cp.sum(strategies) == 1)
  constraints.append(strategies >= 0)

  for n in range(N):
    for a_n in range(num_strategies_list[n]):
      # expected_payoff = cp.sum(game_matrix[n].reshape(-1) @ strategies)
      indices = tuple([a_n if n_ == n else slice(None) for n_ in range(N)])
      tmp_payoffs = np.zeros(shape=game[n].shape)
      tmp_payoffs[indices] = game[n][indices]
      post_payoffs = cp.sum(tmp_payoffs.reshape(-1) @ strategies)

      for a_np in range(num_strategies_list[n]):
        if a_np == a_n:
          continue
        indices_ = tuple([a_np if n_ == n else slice(None) for n_ in range(N)])
        deviation_payoff = np.zeros(shape=game[n].shape)
        deviation_payoff[indices] = game[n][indices_]
        constraints.append(cp.sum(deviation_payoff.reshape(-1)
                           @ strategies)-post_payoffs <= epsilon)

  obj = OBJECTIVE[objective](game, strategies)
  prob = cp.Problem(obj, constraints)
  prob.solve(solver=solver)

  return strategies.value


def max_gini(game, strategies):
  return cp.Minimize(cp.sum(cp.square(strategies)))


def max_entropy(game, strategies):
  return cp.Maximize(cp.sum(cp.entr(strategies)))


def max_log_nash_product(game, strategies):
  N = len(game)
  res = 0
  for n in range(N):
    res += cp.log(game[n].reshape(-1) @ strategies)
  return cp.Maximize(res)


def max_social_welfare(game, strategies):
  N = len(game)
  res = 0
  for n in range(N):
    res += game[n].reshape(-1) @ strategies
  return cp.Maximize(res)


OBJECTIVE = {
    'MAX_SOCIAL_WELFARE': max_social_welfare,
    'MAX_GINI': max_gini,
    'MAX_ENTROPY': max_entropy,
    'MAX_LOG_NASH_PRODUCT': max_log_nash_product,
}


def mip_nash(game, objective, solver='GUROBI'):
  assert len(game) == 2
  assert game[0].shape == game[1].shape

  (M, N) = game[0].shape

  U0 = np.max(game[0]) - np.min(game[0])
  U1 = np.max(game[1]) - np.min(game[1])

  x = cp.Variable(M)
  y = cp.Variable(N)
  u0 = cp.Variable(1)
  u1 = cp.Variable(1)
  b0 = cp.Variable(M, boolean=True)
  b1 = cp.Variable(N, boolean=True)

  u_m = game[0] @ y
  u_n = x @ game[1]

  # probabilities constraints
  constraints = [x >= 0, y >= 0, cp.sum(x) == 1, cp.sum(y) == 1]
  # support constraints
  constraints.extend([u_m <= u0, u0-u_m <= U0 * b0, x <= 1-b0])
  constraints.extend([u_n <= u1, u1-u_n <= U1 * b1, y <= 1-b1])

  variables = {'x': x, 'y': y, 'u0': u0,
               'u1': u1, 'b0': b0, 'b1': b1, 'game': game}

  obj = TWO_PLAYER_OBJECTIVE[objective](variables)
  prob = cp.Problem(obj, constraints)
  prob.solve(solver=solver)

  return _simplex_projection(x.value.reshape(-1)), _simplex_projection(y.value.reshape(-1))


def mip_nash_max_entropy(game, discrete_factors=20, solver='GUROBI'):
  assert len(game) == 2
  assert game[0].shape == game[1].shape

  (M, N) = game[0].shape

  U0 = np.max(game[0]) - np.min(game[0])
  U1 = np.max(game[1]) - np.min(game[1])

  x = cp.Variable(M)
  y = cp.Variable(N)
  u0 = cp.Variable(1)
  u1 = cp.Variable(1)
  b0 = cp.Variable(M, boolean=True)
  b1 = cp.Variable(N, boolean=True)

  z0 = cp.Variable(M)
  z1 = cp.Variable(N)

  u_m = game[0] @ y
  u_n = x @ game[1]

  # probabilities constraints
  constraints = [x >= 0, y >= 0, cp.sum(x) == 1, cp.sum(y) == 1]
  # support constraints
  constraints.extend([u_m <= u0, u0-u_m <= U0 * b0, x <= 1-b0])
  constraints.extend([u_n <= u1, u1-u_n <= U1 * b1, y <= 1-b1])

  for k in range(discrete_factors):
    if k == 0:
      constraints.append(np.log(1/discrete_factors) * x <= z0)
      constraints.append(np.log(1/discrete_factors) * y <= z1)
    else:
      constraints.append(k/discrete_factors * np.log(k/discrete_factors) + ((k+1)*np.log(
          (k+1)/discrete_factors)-k*np.log(k/discrete_factors))*(x - k/discrete_factors) <= z0)
      constraints.append(k/discrete_factors * np.log(k/discrete_factors) + ((k+1)*np.log(
          (k+1)/discrete_factors)-k*np.log(k/discrete_factors))*(y - k/discrete_factors) <= z1)

  obj = cp.Minimize(cp.sum(z0) + cp.sum(z1))
  prob = cp.Problem(obj, constraints)
  prob.solve(solver=solver)

  return _simplex_projection(x.value.reshape(-1)), _simplex_projection(y.value.reshape(-1))


def max_social_welfare_two_player(variables):
  return cp.Maximize(variables['u0'] + variables['u1'])


def min_social_welfare_two_player(variables):
  return cp.Minimize(variables['u0'] + variables['u1'])


def max_support_two_player(variables):
  return cp.Minimize(cp.sum(variables['b0']) + cp.sum(variables['b1']))


def min_support_two_player(variables):
  return cp.Maximize(cp.sum(variables['b0']) + cp.sum(variables['b1']))


def max_gini_two_player(variables):
  return cp.Minimize(cp.sum(cp.square(variables['x'])) + cp.sum(cp.square(variables['y'])))


TWO_PLAYER_OBJECTIVE = {
    'MAX_SOCIAL_WELFARE': max_social_welfare_two_player,
    'MIN_SOCIAL_WELFARE': min_social_welfare_two_player,
    'MAX_SUPPORT': max_support_two_player,
    'MIN_SUPPORT': min_support_two_player,
    'MAX_GINI': max_gini_two_player,
}


def milp_max_sym_ent_2p(game, discrete_factors=20):
  shape = game.shape
  U = np.max(game) - np.min(game)
  assert len(shape) == 2
  assert shape[0] == shape[1]
  M = shape[0]
  x = cp.Variable(M)
  u = cp.Variable(1)
  z = cp.Variable(M)

  obj = cp.Minimize(cp.sum(z))
  # obj = cp.Minimize(cp.sum(cp.square(x)))
  a_mat = np.ones(M).reshape((1, M))
  u_m = game @ x

  b = cp.Variable(M, boolean=True)
  constraints = [u_m <= u, a_mat @ x == 1,
                 x >= 0, u - u_m <= U * b, x <= 1 - b]

  for k in range(discrete_factors):
    if k == 0:
      constraints.append(np.log(1/discrete_factors) * x <= z)
    else:
      constraints.append(k/discrete_factors * np.log(k/discrete_factors) + ((k+1)*np.log(
          (k+1)/discrete_factors)-k*np.log(k/discrete_factors))*(x - k/discrete_factors) <= z)

  prob = cp.Problem(obj, constraints)
  prob.solve(solver='GUROBI')
  return _simplex_projection(x.value.reshape(-1))


def milp_max_payoff_sym_2p(game, objective, solver='GUROBI'):
  shape = game.shape
  U = np.max(game) - np.min(game)
  assert len(shape) == 2
  assert shape[0] == shape[1]
  M = shape[0]
  x = cp.Variable(M)
  u = cp.Variable(1)
  # z = cp.Variable(M)

  # obj = cp.Maximize(u)
  # obj = cp.Minimize(cp.sum(cp.square(x)))
  # a_mat = np.ones(M).reshape((1, M))
  u_m = game @ x

  b = cp.Variable(M, boolean=True)

  variables = {'x': x,
               'u': u, 'b': b, 'game': game}

  constraints = [u_m <= u, cp.sum(x) == 1,
                 x >= 0, u - u_m <= U * b, x <= 1 - b]

  obj = TWO_PLAYER_SYM_OBJECTIVE[objective](variables)
  prob = cp.Problem(obj, constraints)
  prob.solve(solver=solver)
  return _simplex_projection(x.value.reshape(-1))


def max_social_welfare_two_player_sym(variables):
  return cp.Maximize(variables['u'])


def min_social_welfare_two_player_sym(variables):
  return cp.Minimize(variables['u'])


def max_support_two_player_sym(variables):
  return cp.Minimize(cp.sum(variables['b']))


def min_support_two_player_sym(variables):
  return cp.Maximize(cp.sum(variables['b']))


def max_gini_two_player_sym(variables):
  return cp.Minimize(cp.sum(cp.square(variables['x'])))


TWO_PLAYER_SYM_OBJECTIVE = {
    'MAX_SOCIAL_WELFARE': max_social_welfare_two_player_sym,
    'MIN_SOCIAL_WELFARE': min_social_welfare_two_player_sym,
    'MAX_SUPPORT': max_support_two_player_sym,
    'MIN_SUPPORT': min_support_two_player_sym,
    'MAX_GINI': max_gini_two_player_sym,
}


# G = np.random.uniform(size=(5, 5))
# print(milp_max_payoff_sym_2p(G))


# print(cp.installed_solvers())

# print(solve_optimal_cce(game, 0, 'MAX_LOG_NASH_PRODUCT', solver='ECOS_BB'))
