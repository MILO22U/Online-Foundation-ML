import numpy as np

class ThompsonSamplingStruct:
    def __init__(self, num_arm, noise_scale):
        """
        Thompson Sampling with Gaussian likelihood and Gaussian prior.
        
        Prior:      mu_i ~ N(0, 1)  for each arm i
        Likelihood: reward | mu_i ~ N(mu_i, sigma^2)  where sigma = noise_scale
        
        Posterior after n_i observations with sample mean x_bar_i:
            mu_i | data ~ N(posterior_mean, posterior_variance)
            posterior_variance = 1 / (1/prior_var + n_i / sigma^2)
            posterior_mean = posterior_variance * (prior_mean/prior_var + n_i * x_bar_i / sigma^2)
        """
        self.d = num_arm
        self.sigma2 = noise_scale ** 2  # known noise variance

        # Prior: N(0, 1) for each arm
        self.prior_mean = np.zeros(self.d)
        self.prior_var = np.ones(self.d)

        # Sufficient statistics
        self.sum_rewards = np.zeros(self.d)   # sum of rewards for each arm
        self.UserArmTrials = np.zeros(self.d) # number of pulls per arm

        self.time = 0

    def updateParameters(self, articlePicked_id, click):
        self.sum_rewards[articlePicked_id] += click
        self.UserArmTrials[articlePicked_id] += 1
        self.time += 1

    def getTheta(self):
        # Return posterior means as the estimate
        means = np.zeros(self.d)
        for i in range(self.d):
            post_var = 1.0 / (1.0 / self.prior_var[i] + self.UserArmTrials[i] / self.sigma2)
            post_mean = post_var * (self.prior_mean[i] / self.prior_var[i] + self.sum_rewards[i] / self.sigma2)
            means[i] = post_mean
        return means

    def decide(self, pool_articles):
        best_article = None
        best_sample = float('-inf')

        for article in pool_articles:
            arm_id = article.id
            # Compute posterior for this arm
            post_var = 1.0 / (1.0 / self.prior_var[arm_id] + self.UserArmTrials[arm_id] / self.sigma2)
            post_mean = post_var * (self.prior_mean[arm_id] / self.prior_var[arm_id] + self.sum_rewards[arm_id] / self.sigma2)

            # Sample from posterior
            sample = np.random.normal(post_mean, np.sqrt(post_var))

            if sample > best_sample:
                best_sample = sample
                best_article = article

        return best_article


class ThompsonSamplingMAB:
    def __init__(self, num_arm, noise_scale=0.1):
        self.users = {}
        self.num_arm = num_arm
        self.noise_scale = noise_scale
        self.CanEstimateUserPreference = False  # MAB

    def decide(self, pool_articles, userID):
        if userID not in self.users:
            self.users[userID] = ThompsonSamplingStruct(self.num_arm, self.noise_scale)
        return self.users[userID].decide(pool_articles)

    def updateParameters(self, articlePicked, click, userID):
        self.users[userID].updateParameters(articlePicked.id, click)

    def getTheta(self, userID):
        return self.users[userID].getTheta()
