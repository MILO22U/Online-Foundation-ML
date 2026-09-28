"""
Task 2: Linear Bandits Simulation
Run with: python SimulationLinear.py

This simulates linear bandits with random feature vectors.
It runs: EpsilonGreedyLinear, LinUCB, LinTS
"""
import copy
import numpy as np
from random import sample, shuffle
import datetime
import os.path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse

from conf import sim_files_folder, save_address
from util_functions import featureUniform, gaussianFeature
from Articles import ArticleManager
from Users import UserManager
from Simulation import simulateOnlineData

from lib.EpsilonGreedyLinearBandit import EpsilonGreedyLinearBandit
from lib.LinUCB import LinUCBBandit
from lib.LinTS import LinTSBandit

if __name__ == '__main__':
    ## Environment Settings ##
    context_dimension = 25
    actionset = "random"
    testing_iterations = 5000
    NoiseScale = 0.1
    n_articles = 25
    n_users = 10
    poolArticleSize = 10  # randomly sample 10 articles per round (common for linear bandits)

    ## Set Up Simulation ##
    UM = UserManager(context_dimension, n_users, thetaFunc=gaussianFeature, argv={'l2_limit': 1})
    users = UM.simulateThetafromUsers()
    AM = ArticleManager(context_dimension, n_articles=n_articles, argv={'l2_limit': 1})
    articles = AM.simulateArticlePool(actionset)

    simExperiment = simulateOnlineData(
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

    ## Initiate Bandit Algorithms ##
    algorithms = {}
    algorithms['EpsilonGreedyLinear'] = EpsilonGreedyLinearBandit(dimension=context_dimension, lambda_=0.1, epsilon=None)
    algorithms['LinUCB(alpha=0.5)'] = LinUCBBandit(dimension=context_dimension, lambda_=1.0, alpha=0.5)
    algorithms['LinTS(v=0.1)'] = LinTSBandit(dimension=context_dimension, lambda_=1.0, noise_scale=NoiseScale, v_scale=0.1)

    ## Run Simulation ##
    print("=" * 60)
    print("Linear Bandits: LinUCB vs LinTS vs EpsilonGreedy")
    print(f"Settings: dim={context_dimension}, n_articles={n_articles}, "
          f"poolSize={poolArticleSize}, noise={NoiseScale}")
    print("=" * 60)

    finalRegret = simExperiment.runAlgorithms(algorithms)

    print("\n--- Final Accumulated Regret ---")
    for name, regret_list in finalRegret.items():
        if len(regret_list) > 0:
            print(f"{name}: {regret_list[-1]:.2f}")

    print("\nDone! Check SimulationResults/ for CSV output files.")
