import numpy as np

class UCBStruct:
    def __init__(self, num_arm, alpha):
        """
        UCB1 algorithm.
        For each arm, UCB = empirical_mean + alpha * sqrt(log(t) / N_i)
        where t = total pulls, N_i = pulls of arm i
        
        alpha: exploration parameter (default typically sqrt(2) or tuned)
        """
        self.d = num_arm
        self.alpha = alpha

        self.UserArmMean = np.zeros(self.d)    # empirical mean reward per arm
        self.UserArmTrials = np.zeros(self.d)  # number of pulls per arm
        self.time = 0

    def updateParameters(self, articlePicked_id, click):
        n = self.UserArmTrials[articlePicked_id]
        self.UserArmMean[articlePicked_id] = (self.UserArmMean[articlePicked_id] * n + click) / (n + 1)
        self.UserArmTrials[articlePicked_id] += 1
        self.time += 1

    def getTheta(self):
        return self.UserArmMean

    def decide(self, pool_articles):
        # First, play each arm at least once
        for article in pool_articles:
            if self.UserArmTrials[article.id] == 0:
                return article

        # After all arms played once, use UCB formula
        best_article = None
        best_ucb = float('-inf')

        for article in pool_articles:
            arm_id = article.id
            # UCB = mean + alpha * sqrt( log(t) / N_i )
            ucb_value = self.UserArmMean[arm_id] + self.alpha * np.sqrt(
                np.log(self.time) / self.UserArmTrials[arm_id]
            )
            if ucb_value > best_ucb:
                best_ucb = ucb_value
                best_article = article

        return best_article


class UCBMultiArmedBandit:
    def __init__(self, num_arm, alpha=1.0):
        self.users = {}
        self.num_arm = num_arm
        self.alpha = alpha
        self.CanEstimateUserPreference = False  # MAB: no theta estimation

    def decide(self, pool_articles, userID):
        if userID not in self.users:
            self.users[userID] = UCBStruct(self.num_arm, self.alpha)
        return self.users[userID].decide(pool_articles)

    def updateParameters(self, articlePicked, click, userID):
        self.users[userID].updateParameters(articlePicked.id, click)

    def getTheta(self, userID):
        return self.users[userID].UserArmMean
