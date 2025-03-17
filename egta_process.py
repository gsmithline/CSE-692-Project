import numpy as np
from jax.lib import xla_bridge
from absl import app
from absl import flags
from open_spiel.python.algorithms.rnad import rnad
import os
import pyspiel
import pickle
import haiku as hk
# from multiprocessing import Pool, TimeoutError
import time
import os
import copy
from functools import partial
from multiprocessing import cpu_count
import ray
from evaluator_and_generator import ValueData, ImpInfoData, PolicyData
import asyncio
from ray.util.actor_pool import ActorPool
import utils
import dod_config
from distributed_ray_trainer import SelfPlayController
from distributed_ray_trainer import ABRController
from search import ProxySearchPolicyFromSearchController
from search import ProxyPolicyNetFromSearchController
from baseline import DQNBRBaseline
import ppo_distributed_trainer
import ppo
from baseline import NFSPBaseline
import fcp
import time
from psro_v2 import psro_v2
from br_oracle import PPOOracle
from open_spiel.python.policy import UniformRandomPolicy
from solutions import milp_max_sym_ent_2p
import matplotlib.pyplot as plt
import random

FLAGS = flags.FLAGS
flags.DEFINE_integer("max_turns", 10, "max_turns")
flags.DEFINE_float("game_discount", 1.0, "game_discount")
flags.DEFINE_float("prob_end", 0.0, "prob_end")
flags.DEFINE_integer("num_eg", 100, "number of empirical games")
flags.DEFINE_integer("num_seed", 10, "number of seeds")
flags.DEFINE_integer("with_replacement", 1, "if sample with replacement")


def sample_empirical_game(scores, agent_types, seeds_sets):
  save_file_name = "policy_type{}_seed{}"
  # seeds = [np.random.choice(seeds_set) for seeds_set in seeds_sets]
  game_matrix = np.zeros((len(agent_types), len(agent_types)))
  acception_ratio_matrix = np.zeros((len(agent_types), len(agent_types)))
  social_welfare_matrix = np.zeros((len(agent_types), len(agent_types)))
  nash_welfare_matrix = np.zeros((len(agent_types), len(agent_types)))

  for i in range(len(agent_types)):
    for j in range(i, len(agent_types)):
      count = 0
      for i_seed in seeds_sets[i]:
        principal_agent_name = save_file_name.format(agent_types[i], i_seed)
        for j_seed in seeds_sets[j]:
          count += 1
          secondary_agent_name = save_file_name.format(agent_types[j], j_seed)
          if i == j:
            payoffs = scores[(principal_agent_name,
                              secondary_agent_name)][2][0]
            acception_ratio = scores[(
                principal_agent_name, secondary_agent_name)][2][2]
            # print(payoffs)
            game_matrix[i, j] += (payoffs[0] + payoffs[1]) / 2
            acception_ratio_matrix[i, j] += (
                acception_ratio[0] + acception_ratio[1]) / 2
            # social_welfare_matrix[i, j] += payoffs[0] + payoffs[1]
            # nash_welfare_matrix[i, j] += payoffs[0] * payoffs[1]
          else:
            payoffs0 = scores[(principal_agent_name,
                               secondary_agent_name)][2][0]
            payoffs1 = scores[(secondary_agent_name,
                               principal_agent_name)][2][0]
            acception_ratio0 = scores[(
                principal_agent_name, secondary_agent_name)][2][2]
            acception_ratio1 = scores[(
                secondary_agent_name, principal_agent_name)][2][2]
            game_matrix[i, j] += (payoffs0[0] + payoffs1[1]) / 2
            game_matrix[j, i] += (payoffs0[1] + payoffs1[0]) / 2
            acception_ratio_matrix[i, j] += (
                acception_ratio0[0] + acception_ratio1[1]) / 2
            acception_ratio_matrix[j, i] += (
                acception_ratio0[1] + acception_ratio1[0]) / 2

            # social_welfare_matrix[i, j] += (
            #     payoffs0[0] + payoffs0[1] + payoffs1[0] + payoffs1[1]) / 2

            # social_welfare_matrix[j, i] += (
            #     payoffs0[0] + payoffs0[1] + payoffs1[0] + payoffs1[1]) / 2
            # nash_welfare_matrix[i, j] += game_matrix[i, j] * game_matrix[j, i]
            # nash_welfare_matrix[j, i] += game_matrix[i, j] * game_matrix[j, i]

      game_matrix[i, j] /= count
      acception_ratio_matrix[i, j] /= count
      # social_welfare_matrix[i, j] = (game_matrix[i, j] + game_matrix[j, i])
      # nash_welfare_matrix[i, j] = (game_matrix[i, j] * game_matrix[j, i])
      if i != j:
        game_matrix[j, i] /= count
        acception_ratio_matrix[j, i] /= count
        # social_welfare_matrix[j, i] = (game_matrix[i, j] + game_matrix[j, i])
        # nash_welfare_matrix[j, i] = (game_matrix[i, j] * game_matrix[j, i])
      social_welfare_matrix[i, j] = (game_matrix[i, j] + game_matrix[j, i])
      nash_welfare_matrix[i, j] = (game_matrix[i, j] * game_matrix[j, i])
      if i != j:
        social_welfare_matrix[j, i] = (game_matrix[i, j] + game_matrix[j, i])
        nash_welfare_matrix[j, i] = (game_matrix[i, j] * game_matrix[j, i])

  return game_matrix, acception_ratio_matrix, social_welfare_matrix, nash_welfare_matrix


def compute_ne_regret(game_matrix, acception_ratio_matrix, social_welfare_matrix, nash_welfare_matrix):
  nash_strategy = milp_max_sym_ent_2p(game_matrix, 100)
  dev_payoff = game_matrix @ nash_strategy.reshape((-1, 1))

  mixed_payoff = nash_strategy.reshape((1, -1)) @ dev_payoff
  ne_regret = (dev_payoff - mixed_payoff).reshape(-1)
  acception_ratio = acception_ratio_matrix @ nash_strategy.reshape((-1, 1))
  social_welfares = social_welfare_matrix @ nash_strategy.reshape((-1, 1))
  nash_welfares = nash_welfare_matrix @ nash_strategy.reshape((-1, 1))

  sigma_dev_payoff = (nash_strategy.reshape((1, -1)) @ game_matrix).reshape(-1)
  real_nash_welfares = dev_payoff.reshape(-1) * sigma_dev_payoff
  return ne_regret, acception_ratio.reshape(-1), social_welfares.reshape(-1), nash_welfares.reshape(-1), real_nash_welfares


def compute_uniform_ranking(game_matrix, acception_ratio_matrix, social_welfare_matrix, nash_welfare_matrix):
  dev_payoff = np.mean(game_matrix, axis=1)
  acception_ratio = np.mean(acception_ratio_matrix, axis=1)
  social_welfares = np.mean(social_welfare_matrix, axis=1)
  nash_welfares = np.mean(nash_welfare_matrix, axis=1)

  sigma_dev_payoff = np.mean(game_matrix, axis=0)

  return dev_payoff, acception_ratio.reshape(-1), social_welfares.reshape(-1), nash_welfares.reshape(-1), dev_payoff * sigma_dev_payoff


@ray.remote
def sample_game_and_compute(scores, agent_types, seeds_sets, num_eg, num_seed):
  ne_regret_list = []
  ne_acception_ratio_list = []
  ne_social_welfare_list = []
  ne_nash_welfare_list = []
  real_ne_nash_welfare_list = []
  uniform_ranking_list = []
  uniform_acception_ratio_list = []
  uniform_social_welfare_list = []
  uniform_nash_welfare_list = []
  real_uniform_nash_welfare_list = []

  for i in range(num_eg):
    bootstrap_seeds = [random.choices(seed_set, k=min(num_seed, len(seed_set)))
                       for seed_set in seeds_sets]
    game_matrix, acception_ratio_matrix, social_welfare_matrix, nash_welfare_matrix = sample_empirical_game(
        scores, agent_types, bootstrap_seeds)
    ne_regret, ne_acception_ratio, ne_social_welfare, ne_nash_welfare, real_ne_nash_welfare = compute_ne_regret(
        game_matrix, acception_ratio_matrix, social_welfare_matrix, nash_welfare_matrix)
    ne_regret_list.append(ne_regret)
    ne_acception_ratio_list.append(ne_acception_ratio)
    ne_social_welfare_list.append(ne_social_welfare)
    ne_nash_welfare_list.append(ne_nash_welfare)
    real_ne_nash_welfare_list.append(real_ne_nash_welfare)
    uniform_ranking, uniform_acception_ratio, uniform_social_welfare, uniform_nash_welfare, real_uniform_nash_welfare = compute_uniform_ranking(
        game_matrix, acception_ratio_matrix, social_welfare_matrix, nash_welfare_matrix)
    uniform_ranking_list.append(uniform_ranking)
    uniform_acception_ratio_list.append(uniform_acception_ratio)
    uniform_social_welfare_list.append(uniform_social_welfare)
    uniform_nash_welfare_list.append(uniform_nash_welfare)
    real_uniform_nash_welfare_list.append(real_uniform_nash_welfare)

  return ne_regret_list, uniform_ranking_list, ne_acception_ratio_list, uniform_acception_ratio_list, ne_social_welfare_list, uniform_social_welfare_list, ne_nash_welfare_list, uniform_nash_welfare_list, real_ne_nash_welfare_list, real_uniform_nash_welfare_list


def main(_):
  # ray.init(address="local")
  save_path = "./new_bargaining_{}_{:.3f}_{:.3f}/".format(
      FLAGS.max_turns, FLAGS.game_discount, FLAGS.prob_end)
  with open(save_path+"all_scores.pkl", "rb") as f:
    scores = pickle.load(f)

  # agent_types = [
  #     "search", "search_policy_net",  "ppo", "idppo", "deepnash", "nfsp", "psro", "psro_last", "fcp_new", "deepnash_search_sp", "ppo_search_sp", "idppo_search_sp", "uniform", "tough", "soft", "az_search", "az_search_policy_net"]

  # agent_real_names = [
  #     "g_search", "g_search_pn",  "mappo", "idppo", "r-nad", "nfsp", "psro", "psro_last", "fcp", "g_search_r_nad", "g_search_mappo", "g_search_idppo", "uniform", "tough", "soft", "va_search", "va_search_pn"]

  agent_types = ["fcp_new",  "search", "idppo_search_sp", "ppo_search_sp", "search_policy_net", "deepnash_search_sp",
                 "idppo", "ppo", "nfsp", "psro", "psro_last", "deepnash",  "soft", "tough", "uniform", "az_search", "az_search_policy_net"]

  agent_real_names = ["fcp", "g_search", "g_search_idppo", "g_search_mappo", "g_search_pn", "g_search_r_nad",
                      "idppo", "mappo", "nfsp", "psro", "psro_last", "r_nad", "soft", "tough", "uniform", "va_search", "va_search_pn"]

  trained_seeds = [17, 27, 37, 47, 57, 67, 77, 87, 97, 107]

  seeds_sets = [[17] if t in ["uniform", "tough", "soft"]
                else trained_seeds for t in agent_types]

  # scores_list = list(scores.values())
  # for i in range(10):
  #   print(scores_list[i])

  ne_regret_list = []
  ne_acception_ratio_list = []
  ne_social_welfare_list = []
  ne_nash_welfare_list = []
  real_ne_nash_welfare_list = []
  uniform_ranking_list = []
  uniform_acception_ratio_list = []
  uniform_social_welfare_list = []
  uniform_nash_welfare_list = []
  real_uniform_nash_welfare_list = []

  for i in range(FLAGS.num_eg):
    if FLAGS.with_replacement:
      bootstrap_seeds = [random.choices(seed_set, k=min(FLAGS.num_seed, len(seed_set)))
                         for seed_set in seeds_sets]
    else:
      bootstrap_seeds = [random.sample(seed_set, k=min(FLAGS.num_seed, len(seed_set)))
                         for seed_set in seeds_sets]
    game_matrix, acception_ratio_matrix, social_welfare_matrix, nash_welfare_matrix = sample_empirical_game(
        scores, agent_types, bootstrap_seeds)
    # print(game_matrix)
    ne_regret, ne_acception_ratio, ne_social_welfare, ne_nash_welfare, real_ne_nash_welfare = compute_ne_regret(
        game_matrix, acception_ratio_matrix, social_welfare_matrix, nash_welfare_matrix)
    ne_regret_list.append(ne_regret)
    ne_acception_ratio_list.append(ne_acception_ratio)
    ne_social_welfare_list.append(ne_social_welfare)
    ne_nash_welfare_list.append(ne_nash_welfare)
    real_ne_nash_welfare_list.append(real_ne_nash_welfare)
    uniform_ranking, uniform_acception_ratio, uniform_social_welfare, uniform_nash_welfare, real_uniform_nash_welfare = compute_uniform_ranking(
        game_matrix, acception_ratio_matrix, social_welfare_matrix, nash_welfare_matrix)
    uniform_ranking_list.append(uniform_ranking)
    uniform_acception_ratio_list.append(uniform_acception_ratio)
    uniform_social_welfare_list.append(uniform_social_welfare)
    uniform_nash_welfare_list.append(uniform_nash_welfare)
    real_uniform_nash_welfare_list.append(real_uniform_nash_welfare)

  # multiple_results = [sample_game_and_compute.remote(scores, agent_types, seeds_sets, int(
  #     FLAGS.num_eg/num_process), FLAGS.num_seed) for _ in range(num_process)]
  # results = ray.get(multiple_results)

  results = ne_regret_list, uniform_ranking_list, ne_acception_ratio_list, uniform_acception_ratio_list, ne_social_welfare_list, uniform_social_welfare_list, ne_nash_welfare_list, uniform_nash_welfare_list, real_ne_nash_welfare_list, real_uniform_nash_welfare_list

  # ne_regret_list = np.concatenate([result[0] for result in results], axis=0)
  ne_regret_mean = np.mean(ne_regret_list, axis=0)
  ne_regret_std = np.std(ne_regret_list, axis=0)

  # ne_regret_acception_ratio_list = np.concatenate(
  #     [result[2] for result in results], axis=0)
  ne_regret_acception_ratio_mean = np.mean(
      ne_acception_ratio_list, axis=0)
  ne_regret_acception_ratio_std = np.std(
      ne_acception_ratio_list, axis=0)

  # ne_social_welfare_list = np.concatenate(
  # [result[4] for result in results], axis=0)
  ne_social_welfare_mean = np.mean(ne_social_welfare_list, axis=0)
  ne_social_welfare_std = np.std(ne_social_welfare_list, axis=0)

  # ne_nash_welfare_list = np.concatenate(
  #     [result[6] for result in results], axis=0)
  ne_nash_welfare_mean = np.mean(ne_nash_welfare_list, axis=0)
  ne_nash_welfare_std = np.std(ne_nash_welfare_list, axis=0)

  # real_ne_nash_welfare_list = np.concatenate(
  #     [result[8] for result in results], axis=0)
  real_ne_nash_welfare_mean = np.mean(real_ne_nash_welfare_list, axis=0)
  real_ne_nash_welfare_std = np.std(real_ne_nash_welfare_list, axis=0)

  # uniform_ranking_list = np.concatenate(
  #     [result[1] for result in results], axis=0)
  uniform_ranking_mean = np.mean(uniform_ranking_list, axis=0)
  uniform_ranking_std = np.std(uniform_ranking_list, axis=0)

  # uniform_ranking_acception_ratio_list = np.concatenate(
  #     [result[3] for result in results], axis=0)
  uniform_ranking_acception_ratio_mean = np.mean(
      uniform_acception_ratio_list, axis=0)
  uniform_ranking_acception_ratio_std = np.std(
      uniform_acception_ratio_list, axis=0)

  # uniform_social_welfare_list = np.concatenate(
  #     [result[5] for result in results], axis=0)
  uniform_social_welfare_mean = np.mean(
      uniform_social_welfare_list, axis=0)
  uniform_social_welfare_std = np.std(
      uniform_social_welfare_list, axis=0)

  # uniform_nash_welfare_list = np.concatenate(
  #     [result[7] for result in results], axis=0)
  uniform_nash_welfare_mean = np.mean(
      uniform_nash_welfare_list, axis=0)
  uniform_nash_welfare_std = np.std(
      uniform_nash_welfare_list, axis=0)

  # real_uniform_nash_welfare_list = np.concatenate(
  #     [result[9] for result in results], axis=0)
  real_uniform_nash_welfare_mean = np.mean(
      real_uniform_nash_welfare_list, axis=0)
  real_uniform_nash_welfare_std = np.std(
      real_uniform_nash_welfare_list, axis=0)

  # for i in range(FLAGS.num_eg):
  #   game_matrix = sample_empirical_game(scores, agent_types, seeds_sets)
  #   ne_regret = compute_ne_regret(game_matrix)
  #   ne_regret_list.append(ne_regret)
  #   uniform_ranking = compute_uniform_ranking(game_matrix)
  #   uniform_ranking_list.append(uniform_ranking)

  # end_time = time.time()
  # print("Time used:", end_time - start_time)
  # ne_regret_list = np.array(ne_regret_list)
  # ne_regret_mean = np.mean(ne_regret_list, axis=0)
  # ne_regret_std = np.std(ne_regret_list, axis=0)
  print(ne_regret_mean)
  ne_ranking = np.argsort(ne_regret_mean)[::-1]
  # print(ne_ranking, ne_regret_mean)

  print("NE Ranking:")

  for i in ne_ranking:

    print("method label {}".format(i),
          agent_real_names[i], ne_regret_mean[i], "+/-", ne_regret_std[i])

  print("NE ratio Ranking:")
  ne_acception_ratio_ranking = np.argsort(
      ne_regret_acception_ratio_mean)[:: -1]

  for i in ne_acception_ratio_ranking:
    # print(i)

    print("method label {}".format(i),
          agent_real_names[i], ne_regret_acception_ratio_mean[i], "+/-", ne_regret_acception_ratio_std[i])

  print("NE social welfare Ranking:")
  ne_social_welfare_ranking = np.argsort(
      ne_social_welfare_mean)[:: -1]

  for i in ne_social_welfare_ranking:
    # print(i)

    print("method label {}".format(i),
          agent_real_names[i], ne_social_welfare_mean[i], "+/-", ne_social_welfare_std[i])

  print("NE nash welfare Ranking:")
  ne_nash_welfare_ranking = np.argsort(
      ne_nash_welfare_mean)[:: -1]

  for i in ne_nash_welfare_ranking:
    # print(i)

    print("method label {}".format(i),
          agent_real_names[i], ne_nash_welfare_mean[i], "+/-", ne_nash_welfare_std[i])

  print("real NE nash welfare Ranking:")
  real_ne_nash_welfare_ranking = np.argsort(
      real_ne_nash_welfare_mean)[:: -1]

  for i in real_ne_nash_welfare_ranking:
    # print(i)

    print("method label {}".format(i),
          agent_real_names[i], real_ne_nash_welfare_mean[i], "+/-", real_ne_nash_welfare_std[i])

  # uniform_ranking_list = np.array(uniform_ranking_list)
  # uniform_ranking_mean = np.mean(uniform_ranking_list, axis=0)
  # uniform_ranking_std = np.std(uniform_ranking_list, axis=0)
  uniform_ranking = np.argsort(uniform_ranking_mean)[::-1]

  print("Uniform Ranking:")
  for i in uniform_ranking:

    print("method label {}".format(i), agent_real_names[i], uniform_ranking_mean[i],
          "+/-", uniform_ranking_std[i])

  uniform_ranking_acception_ratio = np.argsort(
      uniform_ranking_acception_ratio_mean)[:: -1]

  print("Uniform acception ratio Ranking:")
  for i in uniform_ranking_acception_ratio:

    print("method label {}".format(i), agent_real_names[i], uniform_ranking_acception_ratio_mean[i],
          "+/-", uniform_ranking_acception_ratio_std[i])

  print("uniform social welfare ranking:")

  uniform_social_welfare_ranking = np.argsort(
      uniform_social_welfare_mean)[:: -1]

  for i in uniform_social_welfare_ranking:
    # print(i)

    print("method label {}".format(i),
          agent_real_names[i], uniform_social_welfare_mean[i], "+/-", uniform_social_welfare_std[i])

  print("uniform nash welfare ranking:")

  uniform_nash_welfare_ranking = np.argsort(
      uniform_nash_welfare_mean)[:: -1]

  for i in uniform_nash_welfare_ranking:
    # print(i)

    print("method label {}".format(i),
          agent_real_names[i], uniform_nash_welfare_mean[i], "+/-", uniform_nash_welfare_std[i])

  print("real uniform nash welfare ranking:")

  real_uniform_nash_welfare_ranking = np.argsort(
      real_uniform_nash_welfare_mean)[:: -1]

  for i in real_uniform_nash_welfare_ranking:
    # print(i)

    print("method label {}".format(i),
          agent_real_names[i], real_uniform_nash_welfare_mean[i], "+/-", real_uniform_nash_welfare_std[i])

  with open(save_path+"ranking_lists_{}.pkl".format(FLAGS.num_eg), "wb") as f:
    pickle.dump((ne_regret_mean, ne_regret_std,
                uniform_ranking_mean, uniform_ranking_std, ne_regret_acception_ratio_mean, ne_regret_acception_ratio_std, uniform_ranking_acception_ratio_mean, uniform_ranking_acception_ratio_std, ne_social_welfare_mean, ne_social_welfare_std, uniform_social_welfare_mean, uniform_social_welfare_std, ne_nash_welfare_mean, ne_nash_welfare_std, uniform_nash_welfare_mean, uniform_nash_welfare_std, real_ne_nash_welfare_mean, real_ne_nash_welfare_std, real_uniform_nash_welfare_mean, real_uniform_nash_welfare_std), f)

  n_bins = 20
  for i in range(len(agent_types)):
     fig, axs = plt.subplots(1, 1, sharey=True, tight_layout=True)
     axs.hist(ne_regret_list[:, i], bins=n_bins)
     axs.set_title(agent_real_names[i])
     axs.set_xlabel("NE-regret")
     axs.set_ylabel("Frequency")
     locs = axs.get_yticks()
     axs.set_yticklabels(['%.2f' % (i/FLAGS.num_eg) for i in locs])

     fig.savefig(save_path+"ne_regret_{}.pdf".format(agent_real_names[i]))


if __name__ == "__main__":
  app.run(main)
