# A beginner's roadmap to local regularization in multiclass learning

**The central question is deceptively simple: must a learning algorithm "peek" at unlabeled training data to learn every multiclass problem?** Asilis, Devic, Dughmi, Sharan, and Teng (COLT 2024) proved that *unsupervised* local regularization — where the regularizer sees the test point and the unlabeled training features — characterizes all multiclass learnability with optimal sample complexity. But they left open whether the unsupervised component is truly necessary: can a *local* regularizer (seeing only the test point, not the other training features) suffice? The authors conjectured no, and proposed a candidate counterexample class H△. As of March 2026, Jafar, Asilis, and Dughmi (COLT 2025) have proved the negative answer in the transductive model using cryptographic secret-sharing arguments, but **the PAC model case remains wide open** — making this one of the most compelling open problems in learning theory today.

---

## How binary and multiclass learning diverge

To understand why this problem exists, you must first grasp a dramatic structural gap between binary and multiclass classification. In binary classification (labels in {0,1}), four things are equivalent: finite VC dimension, uniform convergence, PAC learnability, and learnability by ERM. This is the Fundamental Theorem of PAC Learning (Blumer, Ehrenfeucht, Haussler, Warmuth 1989), and it means that a single, simple algorithmic principle — pick any hypothesis minimizing training error — universally works. **Structural risk minimization (SRM)**, which adds a complexity penalty ψ(h) to break ties among equally good hypotheses, is just a refinement that doesn't change what's learnable.

Multiclass classification with large or infinite label sets shatters this clean picture. Daniely and Shalev-Shwartz (2014) proved that **no proper learner** (one that must output a hypothesis from H) can learn all learnable multiclass classes — and since ERM and SRM are proper, they fail too. The correct characterization requires the **DS dimension** (named after Daniely and Shalev-Shwartz), which generalizes VC dimension using "pseudo-cubes" in one-inclusion hypergraphs. Brukhim, Carmon, Dinur, Moran, and Yehudayoff (FOCS 2022) completed the picture by proving that finite DS dimension is both necessary *and* sufficient for multiclass PAC learnability, resolving a conjecture from 2014. But their algorithm is complex and doesn't fit any simple template like "minimize empirical risk plus a penalty."

This is where local regularization enters. A **standard regularizer** ψ: H → ℝ≥0 penalizes hypotheses globally — the penalty on hypothesis h is the same regardless of where you evaluate it. A **local regularizer** ψ: H × X → ℝ≥0 lets the penalty depend on the test point x, so the learner outputs h(x) for whichever h minimizes the empirical loss L_S(h) + ψ(h,x). This is powerful because different hypotheses can "win" at different test points, making the overall predictor *improper* — it stitches together predictions from different hypotheses across the domain. An **unsupervised local regularizer** ψ: H × X^{<ω} × X → ℝ≥0 additionally sees the unlabeled training features S_X, enabling a kind of unsupervised pre-training step.

The hierarchy of what each framework can learn is strict:

- **ERM / Standard SRM** (proper, ψ(h)): Learns all binary classes, fails on some learnable multiclass classes
- **Local regularization** (improper, ψ(h,x)): Strictly more powerful — but does it learn everything?
- **Unsupervised local regularization** (improper, ψ(h, S_X, x)): **Characterizes all multiclass learnability** with optimal sample complexity

---

## The two open problems and what is already known

**Open Problem 1** asks whether local regularization (without seeing unlabeled data) suffices to learn all learnable multiclass hypothesis classes, ideally with optimal or near-optimal sample complexity. **Open Problem 2** asks whether the specific candidate class H△ — constructed from triples of finite subsets of ℕ with particular intersection properties, inspired by the "first Cantor class" of Daniely and Shalev-Shwartz — can be learned by any local regularizer.

Several key results frame the landscape. Asilis et al. (COLT 2024) proved in their companion paper "Regularization and Optimal Multiclass Learning" that removing *either* locality (test point access) *or* unsupervised access (unlabeled training data) from unsupervised local regularization causes failure on some learnable class. Specifically, **Proposition 16** exhibits a PAC-learnable class that no local regularizer can learn — but this class uses the "first Cantor class" with a countably infinite label set and an infinite domain, and the argument relies on a specific symmetry structure. The open question is whether a more carefully constructed class (like H△) can resist all local regularizers while remaining PAC-learnable, or whether perhaps every PAC-learnable class *can* be handled by some local regularizer.

The main positive result (**Theorem 20** of the companion paper) is that unsupervised local regularization characterizes learnability: for any learnable H, there exists an unsupervised local regularizer whose induced SRM learners all achieve optimal transductive error up to a constant factor. The construction uses **maximum-entropy orientations of one-inclusion hypergraphs**, which yield Bayesian learners where the prior is computed from unlabeled data. The **Hall complexity** π_H(n) — a graph-theoretic quantity derived from OIG orientations via Hall's marriage theorem — exactly characterizes optimal transductive error rates: ε_H(n) = π_H(n)/n.

---

## The core papers you must understand

### The DS dimension and why proper learning fails

**Daniely and Shalev-Shwartz (2014), "Optimal Learners for Multiclass Problems"** is the paper that launched this entire research program. It introduced the DS dimension and proved that any generic optimal multiclass learner must be improper. The DS dimension measures whether H can realize "pseudo-cubes" on finite subsets — structures analogous to the full binary hypercube {0,1}^n that witnesses VC shattering, but generalized to arbitrary label sets. A set S is DS-shattered if H restricted to S contains a pseudo-cube of dimension |S|: a collection of labelings where, for each coordinate position i, changing just coordinate i produces a distinct labeling, and these "neighbors" are uniquely paired. Finite DS dimension is necessary for learnability. The paper also gave improved OIG-based algorithms and showed their near-optimality.

**Brukhim et al. (FOCS 2022), "A Characterization of Multiclass Learnability"** resolved the sufficiency direction: finite DS dimension implies PAC learnability. This was a major breakthrough requiring fundamentally new techniques — the proof passes through **list PAC learning** (where the learner outputs a short list of candidate labels) as an intermediate relaxation, uses topological arguments about simplicial complexes, and constructs learning algorithms via sample compression. A key corollary: there exist hypothesis classes with **Natarajan dimension 1** (the classical multiclass dimension) that are not learnable, definitively showing the Natarajan dimension is insufficient for characterizing multiclass learnability with unbounded label sets.

### Compression, OIGs, and optimal learning

**David, Moran, and Yehudayoff (NeurIPS 2016)** proved that multiclass PAC learnability is equivalent to the existence of sample compression schemes of logarithmic size. A **sample compression scheme** consists of a compressor (selecting a small subsample from the full training set) and a reconstructor (recovering a correct hypothesis from just the subsample). This equivalence is powerful because compression schemes are a purely combinatorial object. However, **Pabbaraju (ALT 2024)** later showed that, unlike binary classification, multiclass learnable classes do *not* always admit compression schemes whose size depends only on the DS dimension — the compression size must grow with sample size. This is a fundamental binary/multiclass separation that complicates attempts to use compression as a bridge to local regularization.

**One-inclusion graphs (OIGs)**, introduced by Haussler, Littlestone, and Warmuth (1994), are the combinatorial backbone of modern optimal learning algorithms. For a class H restricted to sample S, the OIG has vertices corresponding to the distinct labelings H|_S can produce, with hyperedges connecting labelings that agree on all but one point. An **orientation** of the OIG — assigning each hyperedge a direction — defines a prediction strategy: for each training scenario, the orientation determines the predicted label. The optimal learner corresponds to the orientation minimizing maximum out-degree. **Aden-Ali, Cherapanamjeri, Shetty, and Zhivotovskiy (FOCS 2023)** showed that aggregating OIG-based transductive learners yields optimal PAC bounds, while their COLT 2023 paper demonstrated that individual OIG algorithms are not always optimal in high probability (refuting a conjecture by Warmuth).

### Other essential references

**Charikar and Pabbaraju (STOC 2023)** completely characterized k-list PAC learnability via the k-DS dimension, revealing a hierarchy of multiclass difficulty parameterized by list size. **Montasser, Hanneke, and Srebro (COLT 2019, NeurIPS 2022)** showed that adversarially robust PAC learning requires improperness — a structural parallel to multiclass — and that "local learners" in the robust setting are suboptimal, providing a cautionary analogy. **Dughmi, Kalayci, and York (ALT 2025)** showed PAC learning reduces to transductive learning for bounded losses, establishing that transductive understanding is nearly sufficient for PAC understanding. **Vaškevičius and Zhivotovskiy (Bernoulli 2023)** demonstrated that constrained least squares (a form of proper learning) is suboptimal for regression, paralleling the multiclass failure of proper learning.

---

## Techniques for a positive resolution

A positive resolution would prove that every learnable multiclass class can be learned by some local regularizer ψ(h,x), despite the regularizer not seeing the unlabeled training features.

**The compression route.** If a compression scheme could be converted into a local regularizer, this would yield a positive answer since every learnable class has a compression scheme (David, Moran, Yehudayoff 2016). The intuition is appealing: a compression scheme selects a small *labeled* subsample and reconstructs from it, without needing the unlabeled features of non-selected points. However, the reconstruction function in multiclass compression is not obviously expressible as an argmin over H of empirical loss plus a penalty, and the compression schemes from Brukhim et al. (2022) involve list learning and OIG machinery that may inherently use information about the unlabeled sample.

**Exploiting PAC distributional structure.** The most promising path to a positive answer leverages the fact that in PAC learning, the test point and training points are drawn i.i.d. from the *same* distribution. Unlike the transductive model (where an adversary picks all points), the PAC learner can potentially infer distributional properties from the labeled data alone — the labels carry information about which hypotheses are consistent, and the test point x itself is a sample from the training distribution. This is precisely where the PAC and transductive models diverge, and why Jafar et al.'s transductive impossibility result does not automatically extend to PAC.

**OIG orientation without unlabeled data.** The optimal unsupervised local regularizer from Asilis et al. computes a maximum-entropy orientation of the OIG, which requires knowing the full projection H|_S — and hence the unlabeled sample. The question is whether this orientation can be *approximated* using only the training labels and the test point. If the OIG structure can be statistically estimated from labeled data, local regularization might suffice.

---

## Techniques for a negative resolution

A negative resolution would prove that some specific learnable class (ideally H△) cannot be learned by any local regularizer.

**The symmetry/secret-sharing approach (strongest known technique).** Jafar, Asilis, and Dughmi (COLT 2025) proved the transductive impossibility using a class H_otp based on **one-time pad cryptography**. Each hypothesis divides a domain into two halves, with labels encoding a "shared secret" via XOR. The key property: seeing the label of one half reveals *nothing* about the hypothesis identity — perfect information-theoretic secrecy. They formalized this by showing that any local regularizer inducing vanishing transductive error must have **Condorcet-cyclic preferences** over hypotheses at typical test points, which is impossible for a deterministic regularizer. Extending this to PAC requires handling the i.i.d. structure, which partially "breaks" the secrecy by correlating the test and training distributions.

**Information-theoretic lower bounds.** Classical tools include Fano's inequality (bounding error probability via mutual information between the hypothesis and observations), Le Cam's method (two-point testing), and Assouad's lemma (reducing to multiple binary testing problems). For the local regularization problem, the key would be formalizing that the information channel available to a local regularizer — empirical losses on training data conditioned on the test point — has insufficient capacity to distinguish among competing hypotheses that require knowledge of the training features.

**Communication complexity reductions.** One can view the local regularizer as a communication protocol: the "message" consists of (empirical losses computed from training data, test point x), and the "receiver" must produce a prediction. Lower bounds on the communication complexity of this protocol could establish that local regularization is insufficient. **Yao's minimax principle** (1983) is directly applicable: to prove no randomized local regularizer works, it suffices to exhibit a distribution over learning instances that defeats all deterministic local regularizers.

**Formalizing "peeking is necessary."** The core intuition behind the conjectured negative answer is that a learner "must peek into the training set S to learn the geometry of the underlying distribution." To formalize this, one would need to construct a class where the optimal prediction at test point x depends critically on *which* domain points appear in the training set (their geometric arrangement), not just their labels. The H△ class is designed with this property: the intersection structure of the triples of subsets encodes relationships that can only be decoded by knowing which elements are present.

---

## Recent developments reshaping the landscape (2024–2026)

The most significant development is **Jafar, Asilis, and Dughmi's "Local Regularizers Are Not Transductive Learners" (COLT 2025)**, which provides the first formal separation: there exists a learnable multiclass problem that cannot be transductively learned by any local regularizer. Their construction uses one-time-pad secret sharing, generalizing the Cantor class. The PAC case remains open, and the authors explicitly highlight "the tantalizing possibility of a PAC/transductive separation with respect to local regularization."

Several companion results from the same research group at USC deepen the picture. **"Transductive Learning Is Compact" (Asilis et al., NeurIPS 2024)** proves that transductive sample complexity depends only on finite projections of the hypothesis class — meaning any transductive impossibility must manifest on finite subclasses. **"Proper Learnability and the Role of Unlabeled Data" (Asilis et al., ALT 2025)** introduces *distributional regularization*, showing that when the marginal distribution over X is known, optimal proper learning becomes possible. This underscores that the issue isn't properness per se but *access to distributional information*. The same paper proves that no finite aggregation of proper learners can learn all learnable multiclass classes in the standard PAC model, further separating binary from multiclass.

**Dughmi's survey "PAC Learning is just Bipartite Matching (Sort of)" (ACM SIGACT News, June 2025)** provides an accessible tutorial connecting PAC learning to bipartite matching via OIGs and transductive learning. This is an excellent entry point for understanding the framework underlying both open problems.

On the quantitative side, **Cohen, Erez, Hanneke, Koren, Mansour, Moran, and Zhang (November 2025)** proved that agnostic multiclass PAC sample complexity is governed by **two interacting dimensions**: roughly DS^{1.5}/ε + Nat/ε², where DS is the DS dimension and Nat is the Natarajan dimension. This shows the multiclass landscape is fundamentally richer than binary, where a single dimension (VC) controls everything. **Hanneke, Moran, and Zhang (NeurIPS 2024)** tightened realizable multiclass bounds to within a single log factor of optimal. **Attias, Hanneke, and Ramaswami (ALT 2025)** gave reductions from multiclass compression to binary compression, showing that resolving the binary compression conjecture would immediately yield multiclass results.

---

## A structured study path from beginner to research frontier

### Phase 1: Mathematical foundations (2–4 weeks)

You need solid grounding in probability (concentration inequalities especially), basic combinatorics (Sauer-Shelah lemma, graph theory), and proof-based mathematical reasoning. Start with:

- **"Understanding Machine Learning: From Theory to Algorithms"** by Shalev-Shwartz and Ben-David (2014), freely available online. Read Chapters 2–7 carefully (PAC model, uniform convergence, VC dimension, non-uniform learnability/SRM). Then read Chapters 13 (regularization), 17 (multiclass overview), and **Chapter 29** (multiclass learnability — co-authored with Daniely himself).
- **"Foundations of Machine Learning"** by Mohri, Rostamizadeh, and Talwalkar. Read Chapter 3 (Rademacher complexity, VC dimension) and Appendix D (concentration inequalities).
- For probability review: the appendices of either textbook cover the necessary concentration inequalities (Hoeffding, McDiarmid, Chernoff bounds, union bound).

### Phase 2: Classical papers (2–3 weeks)

Read in this order, with the textbook chapters as scaffolding:

1. **Valiant (1984)**, "A Theory of the Learnable" — short, readable, introduces PAC learning
2. **Blumer et al. (1989)**, "Learnability and the Vapnik-Chervonenkis Dimension" — the Fundamental Theorem
3. **Haussler, Littlestone, Warmuth (1994)** — introduces OIGs for binary classification (read the key construction, not necessarily every proof)

### Phase 3: The multiclass revolution (3–4 weeks)

4. **Daniely and Shalev-Shwartz (2014)**, "Optimal Learners for Multiclass Problems" — DS dimension, proper learning fails
5. **David, Moran, Yehudayoff (2016)**, compression-learnability equivalence
6. **Brukhim et al. (FOCS 2022)**, "A Characterization of Multiclass Learnability" — the breakthrough
7. **Aden-Ali et al. (FOCS 2023)**, "Optimal PAC Bounds Without Uniform Convergence" — OIG-to-PAC conversion

### Phase 4: The target papers (2–3 weeks)

8. **Asilis et al. (COLT 2024)**, "Regularization and Optimal Multiclass Learning" — the companion paper introducing the full regularization framework, local regularization, unsupervised local regularization, Hall complexity, and optimal learners via maximum-entropy OIG orientations
9. **Asilis et al. (COLT 2024)**, "Open Problem: Can Local Regularization Learn All Multiclass Problems?" — the open problem itself

### Phase 5: The frontier (1–2 weeks)

10. **Jafar, Asilis, Dughmi (COLT 2025)**, "Local Regularizers Are Not Transductive Learners" — partial resolution via secret sharing
11. **Dughmi (SIGACT News, 2025)**, "PAC Learning is just Bipartite Matching (Sort of)" — excellent survey contextualizing everything
12. **Asilis et al. (ALT 2025)**, "Proper Learnability and the Role of Unlabeled Data" — distributional regularization

### Key blog posts for intuition

- **"The Curious Landscape of Multiclass Learning"** by Brukhim and Pabbaraju (Learning Theory Alliance Blog, April 2024) — accessible overview of the entire DS dimension story
- **"One-Inclusion Graphs and Optimal Sample Complexity of PAC Learning"** (LeT-All Blog, Parts 1 & 2, 2024) — tutorial on OIGs

---

## Twenty concepts you must master

The following concepts form the conceptual vocabulary of this research area. Master them roughly in order:

1. **PAC learning** — distribution-free guarantee: m(ε,δ) samples suffice for error ≤ ε with probability ≥ 1−δ, for *all* distributions
2. **Realizable vs. agnostic settings** — whether a perfect hypothesis exists; for multiclass, these are equivalent (David et al. 2016)
3. **VC dimension and shattering** — maximum set size where all binary labelings are realized; characterizes binary learnability
4. **Uniform convergence** — empirical risk converges to true risk uniformly over H; equivalent to finite VC for binary, but *fails* for multiclass
5. **ERM and SRM** — minimize training error, optionally plus a complexity penalty; both are proper and fail for multiclass
6. **Proper vs. improper learning** — whether the output must be in H; improper is strictly more powerful for multiclass
7. **Natarajan dimension** — classical multiclass dimension; insufficient for unbounded label sets
8. **DS dimension** — the correct characterization: finite DS dim ↔ multiclass learnability
9. **Graph dimension** — intermediate dimension; characterizes learnability for finite label sets
10. **Pseudo-cubes** — the combinatorial structures witnessing DS-shattering
11. **One-inclusion graphs/hypergraphs** — vertices are labelings, edges connect labelings differing at one point; orientations define learners
12. **Sample compression schemes** — compress training set to small subsample; reconstruct correct hypothesis from subsample alone
13. **Local regularization** — ψ(h,x): penalty depends on hypothesis and test point, enabling improperness
14. **Unsupervised local regularization** — ψ(h, S_X, x): additionally sees unlabeled training features
15. **Hall complexity** — graph-theoretic quantity from OIGs exactly characterizing transductive error rates
16. **Transductive learning** — learner sees all unlabeled points; test point randomly selected from pool
17. **List PAC learning** — predict a short list of labels; correct if true label is included
18. **Minimax rates** — optimal worst-case sample complexity over all distributions and targets
19. **Rademacher complexity** — measures class richness by correlation with random noise; data-dependent bounds
20. **Online-to-batch conversion** — transform online/transductive guarantees into PAC guarantees via exchangeability

---

## Conclusion: where the frontier stands and where to push

The open problem sits at a precise fault line in learning theory: the boundary between what labeled data alone can tell a learner and what requires observing the structure of the input distribution. For binary classification, this boundary doesn't exist — ERM suffices, and no distributional information beyond labels is needed. For multiclass with unbounded labels, the boundary is real: unsupervised access to training features is provably necessary in the transductive model (Jafar et al. 2025). The PAC model, with its i.i.d. structure, may or may not heal this gap.

Three novel insights should guide your approach. First, the **PAC/transductive distinction is the crux**: in PAC, the test point comes from the same distribution as training points, potentially allowing the learner to infer distributional properties from labels alone. If you believe the PAC answer is also negative, you need information-theoretic arguments that survive the i.i.d. structure. If you believe it's positive, you need to show that i.i.d. sampling provides enough implicit distributional information to substitute for explicit unsupervised access. Second, the **recent two-dimensional structure of agnostic multiclass complexity** (DS and Natarajan dimensions both matter, per Cohen et al. 2025) suggests the problem is richer than previously thought — local regularization may suffice for some dimension regimes but not others. Third, **distributional regularization** (Asilis et al. ALT 2025) — which assumes knowledge of the marginal distribution — characterizes proper learning, suggesting that the key resource is distributional information, and the question is whether labeled PAC data implicitly provides enough of it.