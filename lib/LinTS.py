import numpy as np

class LinTSStruct:
    def __init__(self, featureDimension, lambda_, noise_scale, v_scale):
        """
        Linear Thompson Sampling.
        
        Model: reward = theta^T * x + noise,  noise ~ N(0, sigma^2)
        
        Prior:     theta ~ N(0, (1/lambda) * I)
        Posterior: theta | data ~ N(mu, V)
            where:
            V = (A)^{-1} = (lambda * I + sum(x_t * x_t^T))^{-1}
            mu = V * b = V * sum(x_t * reward_t)
        
        Decision: 
            Sample theta_tilde ~ N(mu, v^2 * V)
            Pick arm x that maximizes theta_tilde^T * x
        
        v_scale: scaling factor for sampling variance (controls exploration)
                 Theoretically v^2 = sigma^2, but can be tuned
        """
        self.d = featureDimension
        self.lambda_ = lambda_
        self.sigma = noise_scale
        self.v = v_scale  # exploration scaling

        self.A = lambda_ * np.identity(self.d)       # d x d
        self.b = np.zeros(self.d)                     # d-dim
        self.AInv = np.linalg.inv(self.A)             # A^{-1} = posterior covariance (up to scaling)
        self.UserTheta = np.zeros(self.d)             # posterior mean
        self.time = 0

    def updateParameters(self, articlePicked_FeatureVector, click):
        self.A += np.outer(articlePicked_FeatureVector, articlePicked_FeatureVector)
        self.b += articlePicked_FeatureVector * click
        self.AInv = np.linalg.inv(self.A)
        self.UserTheta = self.AInv.dot(self.b)
        self.time += 1

    def getTheta(self):
        return self.UserTheta

    def getA(self):
        return self.A

    def decide(self, pool_articles):
        # Sample theta_tilde from posterior N(mu, v^2 * A^{-1})
        try:
            theta_sample = np.random.multivariate_normal(self.UserTheta, self.v**2 * self.AInv)
        except np.linalg.LinAlgError:
            # Fallback if covariance not positive definite
            theta_sample = self.UserTheta

        best_article = None
        best_pta = float('-inf')

        for article in pool_articles:
            x = article.featureVector
            # Use sampled theta to compute predicted reward
            pta = np.dot(theta_sample, x)
            if pta > best_pta:
                best_pta = pta
                best_article = article

        return best_article


class LinTSBandit:
    def __init__(self, dimension, lambda_=1.0, noise_scale=0.1, v_scale=0.1):
        self.users = {}
        self.dimension = dimension
        self.lambda_ = lambda_
        self.noise_scale = noise_scale
        self.v_scale = v_scale
        self.CanEstimateUserPreference = True  # Linear: can estimate theta

    def decide(self, pool_articles, userID):
        if userID not in self.users:
            self.users[userID] = LinTSStruct(self.dimension, self.lambda_, self.noise_scale, self.v_scale)
        return self.users[userID].decide(pool_articles)

    def updateParameters(self, articlePicked, click, userID):
        self.users[userID].updateParameters(articlePicked.featureVector[:self.dimension], click)

    def getTheta(self, userID):
        return self.users[userID].UserTheta
