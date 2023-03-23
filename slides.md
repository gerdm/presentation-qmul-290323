---
# try also 'default' to start simple
theme: default
background: /cover-lofi-dalle.png
---

# One-pass learning methods for neural networks
## PhD Fire Talks

Gerardo Duran-Martin

March 2023

---

# Sequential estimation

How can we efficiently train neural networks in one pass of the data?

## Why?
1. Bandits
1. Online-learning problems with distribution shift
1. Finance—detecting toxic flow

---

# Bayes' rule for sequential data
A state-space representation.

Let ${\cal D}_t = ({\bf x}_t, y_t)$.  
In a probabilistic sense, to *train* is to estimate

$$
    p(\theta \vert {\cal D}_{1:t}) \propto p(\theta) p({\cal D}_{1:t} \vert \theta)
$$
at every time $t$.

If we have an estimate for $p(\theta \vert {\cal D}_{1:t-1})$, we can use Bayes' rule to update the posterior
after observing ${\cal D}_t$:
$$
    p(\theta \vert {\cal D}_{1:t}) \propto p({\cal D}_t \vert \theta) p(\theta \vert {\cal D}_{1:t-1})
$$


---

# Bayes' rule for sequential data
R-VGA framework (Lambert et al., 2021)

We find Gaussian approximation to the posterior $q_t(\theta) = {\cal N}(\theta \vert \mu_t, \Sigma_t)$
by solving the following optimisation problem:
$$
    \mu_t, \Sigma_t =
    \argmin_{\mu, \Sigma}\text{KL}({\cal N}(\theta \vert \mu, \Sigma) || q_{t-1}(\theta) p(y_t \vert \theta, {\bf x}_t))
$$

when the likelihood $p({\cal D}_t \vert \theta_t)$ is parameterised by a neural network.

---

# Regression example
The extended Kalman filter (EKF)

Let $y_t \sim {\cal N}(f(\theta, {\bf x}_t), \sigma^2)$.

A first-order Taylor expansion of $f$ around the mean $\mu_{t-1}$ gives

$$
    f(\theta, {\bf x}_t)
    \approx f(\mu_{t-1}, {\bf x}_t) + \nabla_\theta f(\mu_{t-1}, {\bf x}_t)\Big\vert_{\theta=\mu_{t-1}}^\intercal (\theta - \mu_{t-1})
$$

---

# Our work

### Subspace EKF
$$
    \theta_t = {\bf Az}_t + {\bf b}
$$

### Low-rank EKF (LoFi)
Let ${\bf W}_t \in \mathbb{R}^{D \times d}$ be a low-rank matrix,
and let $\bm\Upsilon_t \in \mathbb{R}^{D \times D}$ be a diagonal matrix.

$$
    \bm\Sigma_t^{-1} = \bm\Upsilon_t  + {\bf W}_t {\bf W}_t^\intercal
$$

### Feature-subspace last-layer (SFL2)

---

# Future work
