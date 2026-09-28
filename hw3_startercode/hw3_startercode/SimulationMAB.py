"""
Task 1: Multi-Armed Bandits Simulation
Run with: python SimulationMAB.py

This simulates a K-armed bandit by using basis_vector action set.
It runs: EpsilonGreedy, Explore-then-Commit (multiple m values), UCB, Thompson Sampling
"""
import copy
import numpy as np
from random import sample, shuffle
import datetime
import os.path
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for saving plots
import matplotlib.pyplot as plt
import argparse

from conf import sim_files_folder, save_address
from util_functions import featureUniform, gaussianFeature
from Articles import ArticleManager
from Users import UserManager
from Simulation import simulateOnlineData

from lib.EpsilonGreedyMultiArmedBandit import EpsilonGreedyMultiArmedBandit
from lib.ExploreThenCommit import ExploreThenCommitMAB
from lib.UCB import UCBMultiArmedBandit
from lib.ThompsonSamplingMAB import ThompsonSamplingMAB

if __name__ == '__main__':
    ## Environment Settings ##
    context_dimension = 10          # K = number of arms
    n_articles = context_dimension  # must equal context_dimension for basis_vector
    actionset = "basis_vector"
    testing_iterations = 5000
    NoiseScale = 0.1
    n_users = 10
    poolArticleSize = None          # use all articles each round

    ## Set Up Simulation ##
    UM = UserManager(context_dimension, n_users, thetaFunc=gaussianFeature, argv={'l2_limit': 1})
    users = UM.simulateThetafromUsers()
    AM = ArticleManager(context_dimension, n_articles=n_articles, argv={'l2_limit': 1})
    articles = AM.simulateArticlePool(actionset)

    simExperiment = simulateOnlineData(
        context_dimension=context_dimension,
        testing_iterations=testing_iterations,
        plot=False,  # we'll do our own plotting
        articles=articles,
        users=users,
        noise=lambda: np.random.normal(scale=NoiseScale),
        signature=AM.signature,
        NoiseScale=NoiseScale,
        poolArticleSize=poolArticleSize
    )

    # ============================================================
    # Experiment 1: Compare UCB, Thompson Sampling, EpsilonGreedy
    # ============================================================
    print("=" * 60)
    print("Experiment 1: UCB vs Thompson Sampling vs EpsilonGreedy")
    print("=" * 60)

    algorithms = {}
    algorithms['EpsilonGreedy'] = EpsilonGreedyMultiArmedBandit(num_arm=n_articles, epsilon=None)
    algorithms['UCB(alpha=1.0)'] = UCBMultiArmedBandit(num_arm=n_articles, alpha=1.0)
    algorithms['ThompsonSampling'] = ThompsonSamplingMAB(num_arm=n_articles, noise_scale=NoiseScale)

    finalRegret = simExperiment.runAlgorithms(algorithms)

    # ============================================================
    # Experiment 2: Explore-then-Commit with different m values
    # ============================================================
    print("\n" + "=" * 60)
    print("Experiment 2: Explore-then-Commit with different m values")
    print("=" * 60)

    # Re-create simulation (fresh state)
    simExperiment2 = simulateOnlineData(
        context_dimension=context_dimension,
        testing_iterations=testing_iterations,
        plot=False,
        articles=articles,
        users=users,
        noise=lambda: np.random.normal(scale=NoiseScale),
        signature=AM.signature,
        NoiseScale=NoiseScale,
        poolArticleSize=poolArticleSize
    )

    m_values = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    algorithms2 = {}
    for m in m_values:
        algorithms2[f'ETC(m={m})'] = ExploreThenCommitMAB(num_arm=n_articles, m=m)

    finalRegret2 = simExperiment2.runAlgorithms(algorithms2)

    # Print final regrets
    print("\n--- Final Accumulated Regret ---")
    for name, regret_list in finalRegret2.items():
        if len(regret_list) > 0:
            print(f"{name}: {regret_list[-1]:.2f}")

    print("\nDone! Check SimulationResults/ for CSV output files.")
