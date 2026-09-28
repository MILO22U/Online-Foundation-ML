import numpy as np

class ExploreThenCommitStruct:
    def __init__(self, num_arm, m):
        """
        num_arm: number of arms (K)
        m: number of times to explore EACH arm before committing
        So total exploration = m * K rounds
        """
        self.d = num_arm
        self.m = m  # explore each arm m times

        self.UserArmMean = np.zeros(self.d)    # running mean reward for each arm
        self.UserArmTrials = np.zeros(self.d)  # how many times each arm has been pulled
        self.time = 0
        self.committed_arm = None  # will be set after exploration phase

    def updateParameters(self, articlePicked_id, click):
        # Update the running average for the picked arm
        n = self.UserArmTrials[articlePicked_id]
        self.UserArmMean[articlePicked_id] = (self.UserArmMean[articlePicked_id] * n + click) / (n + 1)
        self.UserArmTrials[articlePicked_id] += 1
        self.time += 1

        # Check if exploration is done (each arm pulled m times)
        # If so, commit to the best arm
        if self.committed_arm is None and np.all(self.UserArmTrials >= self.m):
            self.committed_arm = np.argmax(self.UserArmMean)

    def getTheta(self):
        return self.UserArmMean

    def decide(self, pool_articles):
        # EXPLORATION PHASE: find an arm that hasn't been pulled m times yet
        if self.committed_arm is None:
            # Find arms in the pool that still need exploration
            for article in pool_articles:
                if self.UserArmTrials[article.id] < self.m:
                    return article
            # If all pool arms explored m times, commit now
            self.committed_arm = np.argmax(self.UserArmMean)

        # COMMIT PHASE: always pick the best arm
        for article in pool_articles:
            if article.id == self.committed_arm:
                return article

        # Fallback: if committed arm not in pool, pick best available
        best_article = None
        best_mean = float('-inf')
        for article in pool_articles:
            if self.UserArmMean[article.id] > best_mean:
                best_mean = self.UserArmMean[article.id]
                best_article = article
        return best_article


class ExploreThenCommitMAB:
    def __init__(self, num_arm, m):
        self.users = {}
        self.num_arm = num_arm
        self.m = m
        self.CanEstimateUserPreference = False  # MAB: no theta estimation

    def decide(self, pool_articles, userID):
        if userID not in self.users:
            self.users[userID] = ExploreThenCommitStruct(self.num_arm, self.m)
        return self.users[userID].decide(pool_articles)

    def updateParameters(self, articlePicked, click, userID):
        self.users[userID].updateParameters(articlePicked.id, click)

    def getTheta(self, userID):
        return self.users[userID].UserArmMean
