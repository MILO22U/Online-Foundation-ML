# Detailed Code Explanation — HW3 Bandit Algorithms

## THE BIG PICTURE: How the Simulation Works

Before understanding any algorithm, you need to understand how the simulation calls your code.

### The Loop (inside Simulation.py, line 100-127):

```
For each time step t = 0, 1, 2, ..., T:
    For each user u:
        1. Generate an article pool (the "arms" you can choose from)
        2. Find the OPTIMAL article (best possible reward) — used to calculate regret
        3. For each algorithm:
            a. Call  alg.decide(pool_articles, user_id)    → algorithm picks an article
            b. Calculate reward = theta^T * featureVector + noise
            c. Call  alg.updateParameters(article, reward, user_id)  → algorithm learns
            d. Record regret = optimal_reward - actual_reward
```

### What Your Algorithm Must Do:

Every algorithm you write must have a **two-level structure**:

**Outer class** (e.g., `UCBMultiArmedBandit`):
- Manages a dictionary of users: `self.users = {}`
- When a new user is seen, creates a new "Struct" for them
- Delegates all work to the per-user struct
- Has `self.CanEstimateUserPreference` flag

**Inner class** (e.g., `UCBStruct`):
- Stores the actual algorithm state for ONE user
- Has the real `decide()` and `updateParameters()` logic

This pattern is the SAME for every algorithm. The only thing that changes is the
math inside `decide()` and `updateParameters()`.

---

## UNDERSTANDING THE ENVIRONMENT

### Articles (Arms):
- Each article has an `id` (integer: 0, 1, 2, ...) and a `featureVector`
- In MAB mode (`basis_vector`): featureVector = [0,0,...,1,...,0] (one-hot)
  - article 0 → [1,0,0,...,0]
  - article 1 → [0,1,0,...,0]
  - This means each arm is independent — knowing about arm 0 tells you nothing about arm 1
- In Linear mode (`random`): featureVector = random unit vector
  - Now arms share information through their features

### Users:
- Each user has a hidden `theta` vector (you never see this!)
- Reward = theta^T * featureVector + noise
- Your job: figure out which article gives the highest theta^T * featureVector

### Key Insight:
- In MAB: since features are basis vectors, reward of arm i = theta[i] + noise
  - So each arm has its own independent mean reward
  - You just need to estimate the mean of each arm
- In Linear: reward depends on theta^T * x
  - All arms are connected through the shared theta
  - Observing one arm helps you learn about ALL arms

---

## FILE 1: ExploreThenCommit.py (Explore Then Commit)

### The Algorithm Idea:
Phase 1 (EXPLORE): Pull each arm exactly m times. Record the average reward.
Phase 2 (COMMIT):  Forever after, always pull the arm with the highest average.

### Code Walkthrough:

```python
class ExploreThenCommitStruct:
    def __init__(self, num_arm, m):
        self.d = num_arm                          # K = number of arms
        self.m = m                                 # explore each arm m times
        self.UserArmMean = np.zeros(self.d)        # average reward per arm
        self.UserArmTrials = np.zeros(self.d)      # count of pulls per arm
        self.time = 0
        self.committed_arm = None                  # None = still exploring
```

**What each variable stores:**
- `UserArmMean[i]` = average reward seen from arm i so far
- `UserArmTrials[i]` = how many times arm i has been pulled
- `committed_arm` = None during exploration, set to best arm ID after

### decide() — How it picks an arm:

```python
    def decide(self, pool_articles):
        # EXPLORATION PHASE
        if self.committed_arm is None:
            for article in pool_articles:
                if self.UserArmTrials[article.id] < self.m:
                    return article           # ← pick first under-explored arm
            # If we get here, all arms explored m times
            self.committed_arm = np.argmax(self.UserArmMean)  # ← commit!

        # COMMIT PHASE
        for article in pool_articles:
            if article.id == self.committed_arm:
                return article               # ← always pick committed arm
```

**Step by step:**
1. If we haven't committed yet (`committed_arm is None`):
   - Loop through available articles
   - Find one that's been pulled fewer than m times
   - Pick it (to continue exploring)
2. If all arms have been pulled m times:
   - Look at `UserArmMean` → find which arm has the highest average
   - Set that as `committed_arm` — we'll never change our mind again
3. After committing:
   - Always return the committed arm

### updateParameters() — How it learns:

```python
    def updateParameters(self, articlePicked_id, click):
        n = self.UserArmTrials[articlePicked_id]
        # Running average formula: new_avg = (old_avg * count + new_value) / (count + 1)
        self.UserArmMean[articlePicked_id] = (self.UserArmMean[articlePicked_id] * n + click) / (n + 1)
        self.UserArmTrials[articlePicked_id] += 1
        self.time += 1

        # Check: is exploration done?
        if self.committed_arm is None and np.all(self.UserArmTrials >= self.m):
            self.committed_arm = np.argmax(self.UserArmMean)
```

**The running average formula:**
- Old average was based on `n` observations
- New observation is `click` (the reward)
- New average = (old_avg × n + new_value) / (n + 1)
- Example: mean was 0.5 from 4 pulls, new reward is 0.8
  → new mean = (0.5 × 4 + 0.8) / 5 = 2.8 / 5 = 0.56

### Why does m matter?
- m too SMALL (e.g., 1): You try each arm only once. With noise, you might think a bad arm
  is good just by luck. You commit to a bad arm forever → HIGH regret.
- m too LARGE (e.g., 1000): You spend 1000 × K rounds exploring. Even though you'll find
  the best arm accurately, you wasted tons of time on bad arms → HIGH regret.
- m JUST RIGHT: Good balance. You get a reasonably accurate estimate without wasting
  too many rounds.

---

## FILE 2: UCB.py (Upper Confidence Bound)

### The Algorithm Idea:
Be "optimistic in the face of uncertainty." For each arm, compute:
   UCB(i) = mean(i) + α × √(log(t) / N_i)

- `mean(i)`: average reward of arm i (what we think it gives)
- `α × √(log(t) / N_i)`: "bonus" for uncertainty (arms pulled less get a bigger bonus)
- Pick the arm with the HIGHEST UCB

### The Math Intuition:
- An arm you've pulled 1000 times: you know its mean well → small bonus
- An arm you've pulled 2 times: you're very uncertain → big bonus
- This naturally balances exploration vs exploitation!
- The log(t) grows slowly, so the bonus shrinks over time

### Code Walkthrough:

```python
class UCBStruct:
    def __init__(self, num_arm, alpha):
        self.d = num_arm
        self.alpha = alpha                         # exploration parameter
        self.UserArmMean = np.zeros(self.d)        # mean reward per arm
        self.UserArmTrials = np.zeros(self.d)      # pull count per arm
        self.time = 0
```

### decide():

```python
    def decide(self, pool_articles):
        # Step 1: Play each arm at least once (UCB needs at least 1 observation)
        for article in pool_articles:
            if self.UserArmTrials[article.id] == 0:
                return article

        # Step 2: Compute UCB for each arm, pick the highest
        best_article = None
        best_ucb = float('-inf')
        for article in pool_articles:
            arm_id = article.id
            ucb_value = self.UserArmMean[arm_id] + self.alpha * np.sqrt(
                np.log(self.time) / self.UserArmTrials[arm_id]
            )
            if ucb_value > best_ucb:
                best_ucb = ucb_value
                best_article = article
        return best_article
```

**Step by step:**
1. First, make sure every arm has been tried at least once (otherwise we'd divide by 0)
2. For each available arm, compute:
   - `UserArmMean[arm_id]` = the exploitation term (what we know)
   - `alpha * sqrt(log(time) / trials)` = the exploration bonus
3. Pick the arm with the highest total UCB value

**Example with 3 arms after 100 total pulls:**
- Arm 0: mean=0.7, pulled 80 times → UCB = 0.7 + 1.0 × √(log(100)/80) = 0.7 + 0.24 = 0.94
- Arm 1: mean=0.5, pulled 15 times → UCB = 0.5 + 1.0 × √(log(100)/15) = 0.5 + 0.55 = 1.05
- Arm 2: mean=0.3, pulled 5 times  → UCB = 0.3 + 1.0 × √(log(100)/5)  = 0.3 + 0.96 = 1.26
→ UCB picks arm 2! Even though its mean is lowest, the uncertainty is highest.
  Maybe arm 2 is actually great and we just got unlucky in 5 tries.

### updateParameters():
Same running average formula as ETC — nothing special here.

---

## FILE 3: ThompsonSamplingMAB.py (Thompson Sampling)

### The Algorithm Idea:
Bayesian approach. For each arm, maintain a probability distribution (posterior) over
what we BELIEVE the true mean reward is. Then:
1. SAMPLE a random value from each arm's posterior
2. Pick the arm whose sample is highest

Arms we're uncertain about → wide posterior → samples can be very high or low
Arms we know well → narrow posterior → samples are close to the true mean

### The Math (Gaussian case):

**Setup:**
- True reward of arm i: reward ~ N(μ_i, σ²)   where σ is the noise level
- We want to learn μ_i for each arm

**Prior (what we believe before seeing data):**
- μ_i ~ N(0, 1)   (we start by assuming each arm's mean is around 0)

**After seeing n_i rewards from arm i with sum S_i:**

Posterior variance:  τ² = 1 / (1/prior_var + n_i / σ²)
Posterior mean:      m  = τ² × (prior_mean/prior_var + S_i / σ²)

So: μ_i | data ~ N(m, τ²)

### Code Walkthrough:

```python
class ThompsonSamplingStruct:
    def __init__(self, num_arm, noise_scale):
        self.d = num_arm
        self.sigma2 = noise_scale ** 2       # σ² = known noise variance

        # Prior: N(0, 1) for each arm
        self.prior_mean = np.zeros(self.d)   # prior mean = 0
        self.prior_var = np.ones(self.d)     # prior variance = 1

        # Sufficient statistics (what we need to compute posterior)
        self.sum_rewards = np.zeros(self.d)  # S_i = sum of all rewards from arm i
        self.UserArmTrials = np.zeros(self.d) # n_i = number of pulls of arm i
```

### decide():

```python
    def decide(self, pool_articles):
        best_article = None
        best_sample = float('-inf')

        for article in pool_articles:
            arm_id = article.id

            # Compute posterior parameters for this arm
            post_var = 1.0 / (1.0/self.prior_var[arm_id] + self.UserArmTrials[arm_id]/self.sigma2)
            post_mean = post_var * (self.prior_mean[arm_id]/self.prior_var[arm_id]
                                    + self.sum_rewards[arm_id]/self.sigma2)

            # Draw a random sample from this posterior
            sample = np.random.normal(post_mean, np.sqrt(post_var))

            if sample > best_sample:
                best_sample = sample
                best_article = article

        return best_article
```

**Step by step with an example:**

Suppose arm 0 has been pulled 50 times, sum of rewards = 35, noise σ²=0.01:
- post_var = 1/(1/1 + 50/0.01) = 1/(1 + 5000) = 0.0002
- post_mean = 0.0002 × (0/1 + 35/0.01) = 0.0002 × 3500 = 0.7
- Sample from N(0.7, 0.0002) → very tight around 0.7

Suppose arm 1 has been pulled 2 times, sum of rewards = 1.0:
- post_var = 1/(1 + 2/0.01) = 1/(1 + 200) = 0.00498
- post_mean = 0.00498 × (0 + 1.0/0.01) = 0.00498 × 100 = 0.498
- Sample from N(0.498, 0.00498) → could range from ~0.36 to ~0.64

So arm 0's sample will almost always be ~0.7, but arm 1's sample might
occasionally be > 0.7, in which case TS explores arm 1.

### updateParameters():

```python
    def updateParameters(self, articlePicked_id, click):
        self.sum_rewards[articlePicked_id] += click   # add reward to running sum
        self.UserArmTrials[articlePicked_id] += 1     # increment count
        self.time += 1
```

Very simple! We just store the raw sum and count. The posterior is computed
on-the-fly in `decide()` whenever we need it.

---

## FILE 4: LinUCB.py (Linear UCB)

### Key Difference from MAB:
In MAB, each arm was independent. In Linear bandits:
- reward = θ^T × x + noise
- θ is SHARED across all arms (it's the user's preference vector)
- x is the feature vector of the arm
- When you observe reward from one arm, it helps you learn θ, which helps for ALL arms

### The Math:

**Ridge Regression (parameter estimation):**
We maintain:
- A = λI + Σ(x_t × x_t^T)     ← "covariance-like" matrix (d × d)
- b = Σ(x_t × reward_t)        ← "correlation" vector (d × 1)
- θ̂ = A⁻¹ × b                  ← ridge regression estimate of θ

**UCB (decision):**
For each arm with feature x:
   UCB(x) = θ̂^T × x  +  α × √(x^T × A⁻¹ × x)
             ↑                    ↑
         predicted reward     confidence bonus

The term x^T × A⁻¹ × x measures how "uncertain" we are about the reward
of arm x given our current data. Arms in unexplored "directions" have higher uncertainty.

### Code Walkthrough:

```python
class LinUCBStruct:
    def __init__(self, featureDimension, lambda_, alpha):
        self.d = featureDimension
        self.lambda_ = lambda_                         # regularization
        self.alpha = alpha                              # exploration parameter

        self.A = lambda_ * np.identity(self.d)         # A starts as λI (d×d matrix)
        self.b = np.zeros(self.d)                       # b starts as zeros (d-dim vector)
        self.AInv = np.linalg.inv(self.A)              # A⁻¹ (precomputed for speed)
        self.UserTheta = np.zeros(self.d)              # θ̂ = A⁻¹b (starts at zero)
```

### updateParameters():

```python
    def updateParameters(self, articlePicked_FeatureVector, click):
        # A = A + x × xᵀ   (rank-1 update of the Gram matrix)
        self.A += np.outer(articlePicked_FeatureVector, articlePicked_FeatureVector)

        # b = b + x × reward
        self.b += articlePicked_FeatureVector * click

        # Recompute inverse and estimate
        self.AInv = np.linalg.inv(self.A)
        self.UserTheta = self.AInv.dot(self.b)        # θ̂ = A⁻¹b
```

**What's np.outer(x, x)?**
If x = [1, 2, 3], then np.outer(x, x) = [[1,2,3], [2,4,6], [3,6,9]]
It's a d×d matrix. Adding this to A "records" that we observed direction x.

**What does A⁻¹ represent?**
A⁻¹ is like a covariance matrix — it measures our uncertainty about θ.
- Directions we've observed a lot: A is big → A⁻¹ is small → low uncertainty
- Directions we haven't observed: A stays at λI → A⁻¹ is large → high uncertainty

### decide():

```python
    def decide(self, pool_articles):
        best_article = None
        best_ucb = float('-inf')

        for article in pool_articles:
            x = article.featureVector

            # Exploitation: how much reward we EXPECT
            predicted = np.dot(self.UserTheta, x)    # θ̂ᵀx

            # Exploration: how UNCERTAIN we are about this prediction
            confidence = self.alpha * np.sqrt(np.dot(x, self.AInv.dot(x)))  # α√(xᵀA⁻¹x)

            ucb_value = predicted + confidence

            if ucb_value > best_ucb:
                best_ucb = ucb_value
                best_article = article

        return best_article
```

**Example:**
Suppose d=2, θ̂ = [0.5, 0.3], and we have two articles:
- Article A: x = [1, 0] → predicted = 0.5, confidence depends on how much we've seen [1,0]
- Article B: x = [0, 1] → predicted = 0.3, confidence depends on how much we've seen [0,1]

If we've pulled Article A many times but Article B few times, Article B will have
a higher confidence bonus, which might make its UCB higher despite lower predicted reward.

---

## FILE 5: LinTS.py (Linear Thompson Sampling)

### The Algorithm Idea:
Same as LinUCB for parameter estimation (ridge regression), but instead of using
a confidence bound, we SAMPLE from the posterior distribution over θ.

### The Math:

**Posterior over θ:**
- θ | data ~ N(μ, V)
- μ = A⁻¹b                     (same as LinUCB's θ̂)
- V = v² × A⁻¹                 (posterior covariance, scaled by v²)

**Decision:**
1. Sample θ̃ ~ N(μ, v² × A⁻¹)   ← random draw from posterior
2. For each arm x, compute θ̃ᵀx
3. Pick the arm with highest θ̃ᵀx

### Code Walkthrough:

The `__init__` and `updateParameters` are IDENTICAL to LinUCB (same ridge regression).
The only difference is in `decide()`:

```python
    def decide(self, pool_articles):
        # Sample θ̃ from the posterior distribution
        try:
            theta_sample = np.random.multivariate_normal(
                self.UserTheta,           # mean = A⁻¹b
                self.v**2 * self.AInv     # covariance = v² × A⁻¹
            )
        except np.linalg.LinAlgError:
            theta_sample = self.UserTheta  # fallback: use mean if sampling fails

        # Use θ̃ to pick the best arm
        best_article = None
        best_pta = float('-inf')
        for article in pool_articles:
            x = article.featureVector
            pta = np.dot(theta_sample, x)   # θ̃ᵀx
            if pta > best_pta:
                best_pta = pta
                best_article = article
        return best_article
```

**Why np.random.multivariate_normal?**
Unlike MAB Thompson Sampling where each arm had its own independent posterior,
here θ is a d-dimensional vector and all dimensions are correlated through A⁻¹.
So we need to sample from a MULTIVARIATE Gaussian, not d independent ones.

**What does v_scale control?**
- v_scale LARGE → samples are far from the mean → MORE exploration
- v_scale SMALL → samples are close to the mean → MORE exploitation (nearly greedy)
- Theoretically v² should equal σ² (the noise variance), but it can be tuned

---

## THE OUTER CLASS PATTERN (Same for All Algorithms)

Every algorithm has an outer class that looks almost identical:

```python
class SomeAlgorithm:
    def __init__(self, ...):
        self.users = {}                          # dictionary: userID → Struct
        self.CanEstimateUserPreference = False    # False for MAB, True for Linear

    def decide(self, pool_articles, userID):
        if userID not in self.users:
            self.users[userID] = SomeStruct(...)  # create new struct for new user
        return self.users[userID].decide(pool_articles)

    def updateParameters(self, articlePicked, click, userID):
        # For MAB:    pass articlePicked.id
        # For Linear: pass articlePicked.featureVector
        self.users[userID].updateParameters(...)

    def getTheta(self, userID):
        return self.users[userID].getTheta()   # or .UserTheta or .UserArmMean
```

**Why a dictionary of users?**
The simulation has multiple users (n_users=10). Each user has a DIFFERENT hidden θ.
So the algorithm keeps a separate "struct" (separate estimates, separate counts)
for each user. When user 3 arrives, we look up self.users[3] and use that user's
personalized data.

**Why CanEstimateUserPreference?**
- MAB algorithms only know arm means, not the user's θ vector → set to False
- Linear algorithms estimate θ̂ via ridge regression → set to True
- The simulation uses this flag to decide whether to track parameter estimation error

---

## SIMULATION SCRIPTS

### SimulationMAB.py — What it does:
1. Sets up K=10 arms with basis_vector features (MAB environment)
2. Runs Experiment 1: EpsilonGreedy vs UCB vs Thompson Sampling
3. Runs Experiment 2: Explore-then-Commit with m = 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000
4. Prints final regret for each

### SimulationLinear.py — What it does:
1. Sets up d=25 dimensional random features (Linear environment)
2. Runs EpsilonGreedyLinear vs LinUCB vs LinTS
3. Prints final regret for each

---

## SUMMARY TABLE

| Algorithm | Type   | Key Data Stored        | Decision Rule                                    |
|-----------|--------|------------------------|--------------------------------------------------|
| ETC       | MAB    | mean[], count[], m     | Explore m times each, then commit to best        |
| UCB       | MAB    | mean[], count[]        | Pick highest mean + α√(log(t)/N_i)               |
| TS (MAB)  | MAB    | sum[], count[]         | Sample from posterior, pick highest sample        |
| LinUCB    | Linear | A (d×d), b (d×1)      | Pick highest θ̂ᵀx + α√(xᵀA⁻¹x)                  |
| LinTS     | Linear | A (d×d), b (d×1)      | Sample θ̃ ~ N(A⁻¹b, v²A⁻¹), pick highest θ̃ᵀx    |
