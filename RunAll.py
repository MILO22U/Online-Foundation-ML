"""
=============================================================================
HW3: Foundation of Online Machine Learning — Complete Simulation Runner
=============================================================================
Run with:   python RunAll.py

This script produces ALL results required for the homework submission:
  Task 1: Multi-Armed Bandits (ETC, UCB, Thompson Sampling)
  Task 2: Linear Bandits (LinUCB, LinTS)

Output:
  - Plots saved as PNG files in SimulationResults/
  - Console output with final regret numbers
  - CSV files with detailed regret/parameter data
=============================================================================
"""

import copy
import numpy as np
from random import sample, shuffle
import datetime
import os
import os.path
import matplotlib
matplotlib.use('Agg')  # non-interactive backend so plots save to file
import matplotlib.pyplot as plt

from conf import sim_files_folder, save_address
from util_functions import featureUniform, gaussianFeature
from Articles import ArticleManager
from Users import UserManager
from Simulation import simulateOnlineData

# Import all algorithms
from lib.EpsilonGreedyMultiArmedBandit import EpsilonGreedyMultiArmedBandit
from lib.EpsilonGreedyLinearBandit import EpsilonGreedyLinearBandit
from lib.ExploreThenCommit import ExploreThenCommitMAB
from lib.UCB import UCBMultiArmedBandit
from lib.ThompsonSamplingMAB import ThompsonSamplingMAB
from lib.LinUCB import LinUCBBandit
from lib.LinTS import LinTSBandit

# Make sure output directories exist
os.makedirs(save_address, exist_ok=True)

# ============================================================================
# Helper function to run simulation and collect regret over time
# ============================================================================
def run_and_collect(simExperiment, algorithms):
    """
    Run the simulation and return BatchCumlateRegret for plotting.
    We manually replicate what runAlgorithms does, but capture the data.
    """
    startTime = datetime.datetime.now()

    testing_iterations = simExperiment.testing_iterations
    batchSize = simExperiment.batchSize
    userSize = len(simExperiment.users)

    tim_ = []
    BatchCumlateRegret = {}
    AlgRegret = {}
    ThetaDiffList = {}
    ThetaDiff = {}

    for alg_name, alg in algorithms.items():
        AlgRegret[alg_name] = []
        BatchCumlateRegret[alg_name] = []
        if alg.CanEstimateUserPreference:
            ThetaDiffList[alg_name] = []

    for iter_ in range(testing_iterations):
        for alg_name, alg in algorithms.items():
            if alg.CanEstimateUserPreference:
                ThetaDiff[alg_name] = 0

        for u in simExperiment.users:
            simExperiment.regulateArticlePool()
            noise = simExperiment.noise()
            OptimalReward, OptimalArticle = simExperiment.GetOptimalReward(u, simExperiment.articlePool)
            OptimalReward += noise

            for alg_name, alg in algorithms.items():
                pickedArticle = alg.decide(simExperiment.articlePool, u.id)
                reward = simExperiment.getReward(u, pickedArticle) + noise
                alg.updateParameters(pickedArticle, reward, u.id)
                regret = OptimalReward - reward
                AlgRegret[alg_name].append(regret)

                if alg.CanEstimateUserPreference:
                    ThetaDiff[alg_name] += simExperiment.getL2Diff(u.theta, alg.getTheta(u.id))

        for alg_name, alg in algorithms.items():
            if alg.CanEstimateUserPreference:
                ThetaDiffList[alg_name] += [ThetaDiff[alg_name] / userSize]

        if iter_ % batchSize == 0:
            if iter_ % 1000 == 0:
                print(f"  Iteration {iter_}/{testing_iterations}  "
                      f"Elapsed: {datetime.datetime.now() - startTime}")
            tim_.append(iter_)
            for alg_name in algorithms.keys():
                BatchCumlateRegret[alg_name].append(sum(AlgRegret[alg_name]) / userSize)

    return tim_, BatchCumlateRegret, ThetaDiffList


# ############################################################################
#                      TASK 1: MULTI-ARMED BANDITS
# ############################################################################

print("=" * 70)
print("  TASK 1: MULTI-ARMED BANDITS")
print("=" * 70)

# --- Environment settings for MAB ---
K = 10                          # number of arms
context_dimension = K
n_articles = K                  # must equal K for basis_vector
actionset = "basis_vector"
testing_iterations = 10000      # number of time steps
NoiseScale = 0.1                # std dev of Gaussian noise
n_users = 10
poolArticleSize = None          # all arms available each round

# Set up environment (shared across MAB experiments)
np.random.seed(42)  # for reproducibility
UM = UserManager(context_dimension, n_users, thetaFunc=gaussianFeature, argv={'l2_limit': 1})
users = UM.simulateThetafromUsers()
AM = ArticleManager(context_dimension, n_articles=n_articles, argv={'l2_limit': 1})
articles = AM.simulateArticlePool(actionset)

# ============================================================================
# Task 1a: Compare EpsilonGreedy vs UCB vs Thompson Sampling
# ============================================================================
print("\n" + "-" * 70)
print("  Task 1a: EpsilonGreedy vs UCB vs Thompson Sampling")
print(f"  K={K}, NoiseScale={NoiseScale}, iterations={testing_iterations}")
print("-" * 70)

sim1a = simulateOnlineData(
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

algorithms_1a = {}
algorithms_1a['EpsilonGreedy'] = EpsilonGreedyMultiArmedBandit(num_arm=n_articles, epsilon=None)
algorithms_1a['UCB (alpha=1.0)'] = UCBMultiArmedBandit(num_arm=n_articles, alpha=1.0)
algorithms_1a['Thompson Sampling'] = ThompsonSamplingMAB(num_arm=n_articles, noise_scale=NoiseScale)

tim_1a, regret_1a, _ = run_and_collect(sim1a, algorithms_1a)

# Plot
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
for alg_name in algorithms_1a.keys():
    ax.plot(tim_1a, regret_1a[alg_name], label=alg_name, linewidth=2)
ax.legend(loc='upper left', fontsize=12)
ax.set_xlabel("Iteration", fontsize=12)
ax.set_ylabel("Cumulative Regret", fontsize=12)
ax.set_title(f"Task 1: MAB — EpsilonGreedy vs UCB vs Thompson Sampling\n"
             f"(K={K}, noise={NoiseScale}, T={testing_iterations})", fontsize=13)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(save_address, 'Task1a_MAB_comparison.png'), dpi=150)
plt.close()
print("\n  Final Cumulative Regret:")
for alg_name in algorithms_1a.keys():
    print(f"    {alg_name}: {regret_1a[alg_name][-1]:.2f}")
print(f"  Plot saved: {save_address}/Task1a_MAB_comparison.png")


# ============================================================================
# Task 1b: Explore Then Commit with different m values
# ============================================================================
print("\n" + "-" * 70)
print("  Task 1b: Explore Then Commit — varying m")
print(f"  K={K}, NoiseScale={NoiseScale}, iterations={testing_iterations}")
print("-" * 70)

m_values = [1, 2, 3, 5, 8, 10, 15, 20, 50, 100, 200, 500]

sim1b = simulateOnlineData(
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

algorithms_1b = {}
for m in m_values:
    algorithms_1b[f'ETC (m={m})'] = ExploreThenCommitMAB(num_arm=n_articles, m=m)

tim_1b, regret_1b, _ = run_and_collect(sim1b, algorithms_1b)

# Plot 1b-i: All ETC curves on one plot
fig, ax = plt.subplots(1, 1, figsize=(12, 7))
for alg_name in algorithms_1b.keys():
    ax.plot(tim_1b, regret_1b[alg_name], label=alg_name, linewidth=1.5)
ax.legend(loc='upper left', fontsize=9, ncol=2)
ax.set_xlabel("Iteration", fontsize=12)
ax.set_ylabel("Cumulative Regret", fontsize=12)
ax.set_title(f"Task 1b: Explore Then Commit — Different m values\n"
             f"(K={K}, noise={NoiseScale}, T={testing_iterations})", fontsize=13)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(save_address, 'Task1b_ETC_all_m.png'), dpi=150)
plt.close()

# Plot 1b-ii: Final regret vs m (bar chart / line plot)
final_regrets_etc = []
for m in m_values:
    name = f'ETC (m={m})'
    final_regrets_etc.append(regret_1b[name][-1])

fig, ax = plt.subplots(1, 1, figsize=(10, 6))
ax.bar(range(len(m_values)), final_regrets_etc, tick_label=[str(m) for m in m_values],
       color='steelblue', edgecolor='black')
ax.set_xlabel("Exploration Phase m (pulls per arm)", fontsize=12)
ax.set_ylabel("Final Cumulative Regret", fontsize=12)
ax.set_title(f"Task 1b: ETC Final Regret vs m\n"
             f"(K={K}, noise={NoiseScale}, T={testing_iterations})", fontsize=13)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(save_address, 'Task1b_ETC_regret_vs_m.png'), dpi=150)
plt.close()

print("\n  Final Cumulative Regret for each m:")
best_m = m_values[np.argmin(final_regrets_etc)]
best_regret = min(final_regrets_etc)
for i, m in enumerate(m_values):
    marker = " <-- BEST" if m == best_m else ""
    print(f"    m={m:>4d}: {final_regrets_etc[i]:>10.2f}{marker}")
print(f"\n  Best m = {best_m} with regret = {best_regret:.2f}")
print(f"  Plots saved: {save_address}/Task1b_ETC_all_m.png")
print(f"               {save_address}/Task1b_ETC_regret_vs_m.png")


# ============================================================================
# Task 1c: UCB — report equation and performance
# ============================================================================
print("\n" + "-" * 70)
print("  Task 1c: UCB — Testing different alpha values")
print("-" * 70)

sim1c = simulateOnlineData(
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

algorithms_1c = {}
alpha_values = [0.1, 0.5, 1.0, 2.0]
for alpha in alpha_values:
    algorithms_1c[f'UCB (alpha={alpha})'] = UCBMultiArmedBandit(num_arm=n_articles, alpha=alpha)

tim_1c, regret_1c, _ = run_and_collect(sim1c, algorithms_1c)

fig, ax = plt.subplots(1, 1, figsize=(10, 6))
for alg_name in algorithms_1c.keys():
    ax.plot(tim_1c, regret_1c[alg_name], label=alg_name, linewidth=2)
ax.legend(loc='upper left', fontsize=12)
ax.set_xlabel("Iteration", fontsize=12)
ax.set_ylabel("Cumulative Regret", fontsize=12)
ax.set_title(f"Task 1c: UCB — Different alpha values\n"
             f"(K={K}, noise={NoiseScale}, T={testing_iterations})", fontsize=13)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(save_address, 'Task1c_UCB.png'), dpi=150)
plt.close()

print("\n  UCB Equation: UCB(i) = x_bar_i + alpha * sqrt(log(t) / N_i)")
print("  where x_bar_i = empirical mean of arm i, t = total pulls, N_i = pulls of arm i")
print("\n  Final Cumulative Regret:")
for alg_name in algorithms_1c.keys():
    print(f"    {alg_name}: {regret_1c[alg_name][-1]:.2f}")
print(f"  Plot saved: {save_address}/Task1c_UCB.png")


# ============================================================================
# Task 1d: Thompson Sampling — report posterior and performance
# ============================================================================
print("\n" + "-" * 70)
print("  Task 1d: Thompson Sampling")
print("-" * 70)

print("\n  Prior:     mu_i ~ N(0, 1) for each arm i")
print("  Likelihood: r_t | mu_i ~ N(mu_i, sigma^2)")
print("  Posterior:  mu_i | data ~ N(m_i, tau_i^2)")
print("    where tau_i^2 = 1 / (1/prior_var + n_i / sigma^2)")
print("          m_i     = tau_i^2 * (prior_mean/prior_var + sum_rewards_i / sigma^2)")
print(f"  sigma = {NoiseScale}")
print(f"\n  Thompson Sampling regret (from Task 1a): {regret_1a['Thompson Sampling'][-1]:.2f}")


# ############################################################################
#                      TASK 2: LINEAR BANDITS
# ############################################################################

print("\n\n" + "=" * 70)
print("  TASK 2: LINEAR BANDITS")
print("=" * 70)

# --- Environment settings for Linear Bandits ---
context_dimension_lin = 25
actionset_lin = "random"
testing_iterations_lin = 10000
NoiseScale_lin = 0.1
n_articles_lin = 25
n_users_lin = 10
poolArticleSize_lin = 10        # randomly sample 10 articles per round

np.random.seed(123)
UM_lin = UserManager(context_dimension_lin, n_users_lin, thetaFunc=gaussianFeature, argv={'l2_limit': 1})
users_lin = UM_lin.simulateThetafromUsers()
AM_lin = ArticleManager(context_dimension_lin, n_articles=n_articles_lin, argv={'l2_limit': 1})
articles_lin = AM_lin.simulateArticlePool(actionset_lin)

# ============================================================================
# Task 2a: LinUCB
# ============================================================================
print("\n" + "-" * 70)
print("  Task 2a: LinUCB — Testing different alpha values")
print(f"  dim={context_dimension_lin}, n_articles={n_articles_lin}, "
      f"poolSize={poolArticleSize_lin}, noise={NoiseScale_lin}")
print("-" * 70)

sim2a = simulateOnlineData(
    context_dimension=context_dimension_lin,
    testing_iterations=testing_iterations_lin,
    plot=False,
    articles=articles_lin,
    users=users_lin,
    noise=lambda: np.random.normal(scale=NoiseScale_lin),
    signature=AM_lin.signature,
    NoiseScale=NoiseScale_lin,
    poolArticleSize=poolArticleSize_lin
)

algorithms_2a = {}
algorithms_2a['EpsilonGreedy Linear'] = EpsilonGreedyLinearBandit(
    dimension=context_dimension_lin, lambda_=0.1, epsilon=None)
algorithms_2a['LinUCB (alpha=0.1, lambda=1.0)'] = LinUCBBandit(
    dimension=context_dimension_lin, lambda_=1.0, alpha=0.1)
algorithms_2a['LinUCB (alpha=0.5, lambda=1.0)'] = LinUCBBandit(
    dimension=context_dimension_lin, lambda_=1.0, alpha=0.5)
algorithms_2a['LinUCB (alpha=1.0, lambda=1.0)'] = LinUCBBandit(
    dimension=context_dimension_lin, lambda_=1.0, alpha=1.0)

tim_2a, regret_2a, theta_diff_2a = run_and_collect(sim2a, algorithms_2a)

# Regret plot
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
for alg_name in algorithms_2a.keys():
    ax.plot(tim_2a, regret_2a[alg_name], label=alg_name, linewidth=2)
ax.legend(loc='upper left', fontsize=10)
ax.set_xlabel("Iteration", fontsize=12)
ax.set_ylabel("Cumulative Regret", fontsize=12)
ax.set_title(f"Task 2a: LinUCB — Cumulative Regret\n"
             f"(dim={context_dimension_lin}, poolSize={poolArticleSize_lin}, "
             f"noise={NoiseScale_lin})", fontsize=13)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(save_address, 'Task2a_LinUCB_regret.png'), dpi=150)
plt.close()

# Parameter estimation error plot
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
time_axis = range(testing_iterations_lin)
for alg_name in theta_diff_2a.keys():
    ax.plot(time_axis, theta_diff_2a[alg_name], label=alg_name, linewidth=1.5)
ax.legend(loc='upper right', fontsize=9)
ax.set_xlabel("Iteration", fontsize=12)
ax.set_ylabel("L2 Estimation Error (log scale)", fontsize=12)
ax.set_yscale('log')
ax.set_title(f"Task 2a: LinUCB — Parameter Estimation Error\n"
             f"(dim={context_dimension_lin})", fontsize=13)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(save_address, 'Task2a_LinUCB_theta_error.png'), dpi=150)
plt.close()

print("\n  LinUCB Equations:")
print("    Parameter estimation (ridge regression):")
print("      A = lambda * I + sum(x_t * x_t^T)")
print("      b = sum(x_t * r_t)")
print("      theta_hat = A^{-1} * b")
print("    UCB for arm selection:")
print("      UCB(x) = theta_hat^T * x + alpha * sqrt(x^T * A^{-1} * x)")
print(f"    Hyperparameters: lambda=1.0, alpha tested=[0.1, 0.5, 1.0]")
print("\n  Final Cumulative Regret:")
for alg_name in algorithms_2a.keys():
    print(f"    {alg_name}: {regret_2a[alg_name][-1]:.2f}")
print(f"  Plots saved: {save_address}/Task2a_LinUCB_regret.png")
print(f"               {save_address}/Task2a_LinUCB_theta_error.png")


# ============================================================================
# Task 2b: LinTS
# ============================================================================
print("\n" + "-" * 70)
print("  Task 2b: LinTS — Testing different v_scale values")
print(f"  dim={context_dimension_lin}, n_articles={n_articles_lin}, "
      f"poolSize={poolArticleSize_lin}, noise={NoiseScale_lin}")
print("-" * 70)

sim2b = simulateOnlineData(
    context_dimension=context_dimension_lin,
    testing_iterations=testing_iterations_lin,
    plot=False,
    articles=articles_lin,
    users=users_lin,
    noise=lambda: np.random.normal(scale=NoiseScale_lin),
    signature=AM_lin.signature,
    NoiseScale=NoiseScale_lin,
    poolArticleSize=poolArticleSize_lin
)

algorithms_2b = {}
algorithms_2b['EpsilonGreedy Linear'] = EpsilonGreedyLinearBandit(
    dimension=context_dimension_lin, lambda_=0.1, epsilon=None)
algorithms_2b['LinTS (v=0.01, lambda=1.0)'] = LinTSBandit(
    dimension=context_dimension_lin, lambda_=1.0, noise_scale=NoiseScale_lin, v_scale=0.01)
algorithms_2b['LinTS (v=0.1, lambda=1.0)'] = LinTSBandit(
    dimension=context_dimension_lin, lambda_=1.0, noise_scale=NoiseScale_lin, v_scale=0.1)
algorithms_2b['LinTS (v=0.5, lambda=1.0)'] = LinTSBandit(
    dimension=context_dimension_lin, lambda_=1.0, noise_scale=NoiseScale_lin, v_scale=0.5)

tim_2b, regret_2b, theta_diff_2b = run_and_collect(sim2b, algorithms_2b)

# Regret plot
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
for alg_name in algorithms_2b.keys():
    ax.plot(tim_2b, regret_2b[alg_name], label=alg_name, linewidth=2)
ax.legend(loc='upper left', fontsize=10)
ax.set_xlabel("Iteration", fontsize=12)
ax.set_ylabel("Cumulative Regret", fontsize=12)
ax.set_title(f"Task 2b: LinTS — Cumulative Regret\n"
             f"(dim={context_dimension_lin}, poolSize={poolArticleSize_lin}, "
             f"noise={NoiseScale_lin})", fontsize=13)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(save_address, 'Task2b_LinTS_regret.png'), dpi=150)
plt.close()

# Parameter estimation error plot
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
time_axis = range(testing_iterations_lin)
for alg_name in theta_diff_2b.keys():
    ax.plot(time_axis, theta_diff_2b[alg_name], label=alg_name, linewidth=1.5)
ax.legend(loc='upper right', fontsize=9)
ax.set_xlabel("Iteration", fontsize=12)
ax.set_ylabel("L2 Estimation Error (log scale)", fontsize=12)
ax.set_yscale('log')
ax.set_title(f"Task 2b: LinTS — Parameter Estimation Error\n"
             f"(dim={context_dimension_lin})", fontsize=13)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(save_address, 'Task2b_LinTS_theta_error.png'), dpi=150)
plt.close()

print("\n  LinTS Equations:")
print("    Prior:     theta ~ N(0, (1/lambda) * I)")
print("    Posterior: theta | data ~ N(mu, V)")
print("      where V  = (lambda * I + sum(x_t * x_t^T))^{-1}")
print("            mu = V * sum(x_t * r_t)")
print("    Decision:")
print("      Sample theta_tilde ~ N(mu, v^2 * V)")
print("      Pick arm x = argmax_x theta_tilde^T * x")
print(f"    Hyperparameters: lambda=1.0, v tested=[0.01, 0.1, 0.5]")
print("\n  Final Cumulative Regret:")
for alg_name in algorithms_2b.keys():
    print(f"    {alg_name}: {regret_2b[alg_name][-1]:.2f}")
print(f"  Plots saved: {save_address}/Task2b_LinTS_regret.png")
print(f"               {save_address}/Task2b_LinTS_theta_error.png")


# ============================================================================
# Task 2c: Combined comparison — LinUCB vs LinTS vs EpsilonGreedy
# ============================================================================
print("\n" + "-" * 70)
print("  Task 2c: Combined — LinUCB vs LinTS vs EpsilonGreedy")
print("-" * 70)

sim2c = simulateOnlineData(
    context_dimension=context_dimension_lin,
    testing_iterations=testing_iterations_lin,
    plot=False,
    articles=articles_lin,
    users=users_lin,
    noise=lambda: np.random.normal(scale=NoiseScale_lin),
    signature=AM_lin.signature,
    NoiseScale=NoiseScale_lin,
    poolArticleSize=poolArticleSize_lin
)

algorithms_2c = {}
algorithms_2c['EpsilonGreedy Linear'] = EpsilonGreedyLinearBandit(
    dimension=context_dimension_lin, lambda_=0.1, epsilon=None)
algorithms_2c['LinUCB (alpha=0.5)'] = LinUCBBandit(
    dimension=context_dimension_lin, lambda_=1.0, alpha=0.5)
algorithms_2c['LinTS (v=0.1)'] = LinTSBandit(
    dimension=context_dimension_lin, lambda_=1.0, noise_scale=NoiseScale_lin, v_scale=0.1)

tim_2c, regret_2c, theta_diff_2c = run_and_collect(sim2c, algorithms_2c)

fig, ax = plt.subplots(1, 1, figsize=(10, 6))
for alg_name in algorithms_2c.keys():
    ax.plot(tim_2c, regret_2c[alg_name], label=alg_name, linewidth=2)
ax.legend(loc='upper left', fontsize=12)
ax.set_xlabel("Iteration", fontsize=12)
ax.set_ylabel("Cumulative Regret", fontsize=12)
ax.set_title(f"Task 2: Linear Bandits — LinUCB vs LinTS vs EpsilonGreedy\n"
             f"(dim={context_dimension_lin}, poolSize={poolArticleSize_lin}, "
             f"noise={NoiseScale_lin})", fontsize=13)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(save_address, 'Task2c_Linear_comparison.png'), dpi=150)
plt.close()

fig, ax = plt.subplots(1, 1, figsize=(10, 6))
time_axis = range(testing_iterations_lin)
for alg_name in theta_diff_2c.keys():
    ax.plot(time_axis, theta_diff_2c[alg_name], label=alg_name, linewidth=1.5)
ax.legend(loc='upper right', fontsize=10)
ax.set_xlabel("Iteration", fontsize=12)
ax.set_ylabel("L2 Estimation Error (log scale)", fontsize=12)
ax.set_yscale('log')
ax.set_title(f"Task 2: Linear Bandits — Parameter Estimation Error", fontsize=13)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(save_address, 'Task2c_Linear_theta_error.png'), dpi=150)
plt.close()

print("\n  Final Cumulative Regret:")
for alg_name in algorithms_2c.keys():
    print(f"    {alg_name}: {regret_2c[alg_name][-1]:.2f}")
print(f"  Plots saved: {save_address}/Task2c_Linear_comparison.png")
print(f"               {save_address}/Task2c_Linear_theta_error.png")


# ############################################################################
#                      SUMMARY
# ############################################################################
print("\n\n" + "=" * 70)
print("  COMPLETE SUMMARY")
print("=" * 70)

print("\n  TASK 1: Multi-Armed Bandits (K=10, noise=0.1)")
print("  -------------------------------------------------")
print(f"    EpsilonGreedy:      {regret_1a['EpsilonGreedy'][-1]:>10.2f}")
print(f"    UCB (alpha=1.0):    {regret_1a['UCB (alpha=1.0)'][-1]:>10.2f}")
print(f"    Thompson Sampling:  {regret_1a['Thompson Sampling'][-1]:>10.2f}")
print(f"    Best ETC (m={best_m}):  {best_regret:>10.2f}")

print(f"\n  TASK 2: Linear Bandits (dim=25, noise=0.1)")
print("  -------------------------------------------------")
for alg_name in algorithms_2c.keys():
    print(f"    {alg_name:30s}: {regret_2c[alg_name][-1]:>10.2f}")

print(f"\n  All plots saved in: {save_address}/")
print("  Files generated:")
print("    Task1a_MAB_comparison.png")
print("    Task1b_ETC_all_m.png")
print("    Task1b_ETC_regret_vs_m.png")
print("    Task1c_UCB.png")
print("    Task2a_LinUCB_regret.png")
print("    Task2a_LinUCB_theta_error.png")
print("    Task2b_LinTS_regret.png")
print("    Task2b_LinTS_theta_error.png")
print("    Task2c_Linear_comparison.png")
print("    Task2c_Linear_theta_error.png")

print("\n" + "=" * 70)
print("  COMMANDS USED TO GENERATE RESULTS:")
print("    python RunAll.py")
print("=" * 70)
print("\nDone!")
