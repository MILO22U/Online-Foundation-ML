import numpy as np

class LinUCBStruct:
    def __init__(self, featureDimension, lambda_, alpha):
        """
        LinUCB algorithm.
        
        Model: reward = theta^T * x + noise
        
        Maintains:
            A = lambda * I + sum(x_t * x_t^T)   (Gram matrix)
            b = sum(x_t * reward_t)
            theta_hat = A^{-1} * b               (ridge regression estimate)
        
        Decision: pick arm x that maximizes:
            UCB(x) = theta_hat^T * x + alpha * sqrt(x^T * A^{-1} * x)
        
        alpha: exploration parameter, controls width of confidence interval
        lambda_: regularization parameter
        """
        self.d = featureDimension
        self.lambda_ = lambda_
        self.alpha = alpha

        self.A = lambda_ * np.identity(self.d)       # d x d matrix
        self.b = np.zeros(self.d)                     # d-dim vector
        self.AInv = np.linalg.inv(self.A)             # A^{-1}
        self.UserTheta = np.zeros(self.d)             # theta_hat = A^{-1} b
        self.time = 0

    def updateParameters(self, articlePicked_FeatureVector, click):
        # A = A + x * x^T
        self.A += np.outer(articlePicked_FeatureVector, articlePicked_FeatureVector)
        # b = b + x * reward
        self.b += articlePicked_FeatureVector * click
        # Update inverse and estimate
        self.AInv = np.linalg.inv(self.A)
        self.UserTheta = self.AInv.dot(self.b)
        self.time += 1

    def getTheta(self):
        return self.UserTheta

    def getA(self):
        return self.A

    def decide(self, pool_articles):
        best_article = None
        best_ucb = float('-inf')

        for article in pool_articles:
            x = article.featureVector
            # Predicted reward: theta_hat^T * x
            predicted = np.dot(self.UserTheta, x)
            # Confidence bonus: alpha * sqrt(x^T * A^{-1} * x)
            confidence = self.alpha * np.sqrt(np.dot(x, self.AInv.dot(x)))
            # UCB = predicted + confidence
            ucb_value = predicted + confidence

            if ucb_value > best_ucb:
                best_ucb = ucb_value
                best_article = article

        return best_article


class LinUCBBandit:
    def __init__(self, dimension, lambda_=1.0, alpha=0.5):
        self.users = {}
        self.dimension = dimension
        self.lambda_ = lambda_
        self.alpha = alpha
        self.CanEstimateUserPreference = True  # Linear: can estimate theta

    def decide(self, pool_articles, userID):
        if userID not in self.users:
            self.users[userID] = LinUCBStruct(self.dimension, self.lambda_, self.alpha)
        return self.users[userID].decide(pool_articles)

    def updateParameters(self, articlePicked, click, userID):
        self.users[userID].updateParameters(articlePicked.featureVector[:self.dimension], click)

    def getTheta(self, userID):
        return self.users[userID].UserTheta
