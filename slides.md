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
layout: center
---

## Gradient descent
$$
    \bm\theta^{(e)} := \bm\theta^{(e-1)} - \eta \nabla_\theta \mathcal{L}(\bm\theta^{(e-1)}, {\cal D})
$$

---

# Sequential estimation

How can we efficiently train neural networks in one pass of the data?

## Why?
1. Streaming data
2. Continuously changing environment
3. Data storage and computational constraints

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

# Regression example
The extended Kalman filter (EKF)

Suppose:
1. $f: \mathbb{R}^D \times \mathbb{R}^M \to \mathbb{R}$ is a multilayered perceptron (MLP) with parameters $\theta$.  
2. the likelihood is  $p(y_t | \theta, {\bf x}_t) = {\cal N}(y_t \vert f(\theta, {\bf x}_t), \sigma^2)$.  
3. the prior is $p(\theta) = {\cal N}(\theta \vert \mu_0, \Sigma_0)$.

A first-order Taylor expansion of $f$ around the mean $\mu_{t-1}$ gives

$$
\begin{aligned}
    f(\theta, {\bf x}_t)
    \approx \hat{f}(\theta, {\bf x}_t)
    = f(\mu_{t-1}, {\bf x}_t) + \nabla_\theta f(\theta, {\bf x}_t)\Big\vert_{\theta=\mu_{t-1}}^\intercal (\theta - \mu_{t-1})
\end{aligned}
$$

Then $p(\theta \vert {\cal D}_{1:t})$ is Gaussian with mean $\mu_t$ and covariance $\Sigma_t$ given
by the Kalman filter update equations.

---

# The EKF for neural networks
Neural network posterior predictive distribution

<!-- Center the figure below -->
<img class="horizontal-center" width=500
     src="https://user-images.githubusercontent.com/4108759/159231061-377f69f9-dbee-40c4-84c9-74fe5cf8ef5f.gif">

---

# What gives?
Applying the EKF equations to neural networks is not straightforward.

1. EKF update equation is order $O(D^3)$.
1. The likelihood $p(y_t \vert \theta, {\bf x}_t)$ is not Gaussian.
2. High-dimensional parameter space ($D \gg N$).
3. Continously changing environment.
4. Choice of prior $p_0(\theta) = {\cal N}(\theta | \mu_0, \Sigma_0)$ is not trivial for neural networks.

---
layout: center
---

# Our work
Online training of modern neural networks


---

# Some assumptions
The state-space representation

$$
\begin{aligned}
    p(\bm\theta_0) &= {\cal N}(\theta_0 \vert \bm\mu_0, \bm\Upsilon_0)\\
    p(\bm\theta_t \vert \bm\theta_{t-1}) &= {\cal N}(\bm\theta_t \vert \gamma \bm\theta_{t-1}, {\bf Q}_t)\\
    p(y_t \vert \bm\theta_t, {\bf x}_t) &=
    \begin{cases}
    {\cal N}(y_t \vert f(\bm\theta_t, {\bf x}_t), r{\bf I}) & \text{Regression}\\
    \text{Mult}(y_t \vert \text{softmax}(f(\bm\theta_t, {\bf x}_t))) & \text{Classification}
    \end{cases}
\end{aligned}
$$

where $f: \mathbb{R}^D \times \mathbb{R}^C \to \mathbb{R}$ is any neural network architecture.


---

# Bayes' rule for sequential data
R-VGA framework (Lambert et al., 2021)

We recursively estimate Gaussian approximation to the posterior 
by solving the following optimisation problem:
$$
    \mu_t, \Sigma_t =
    \argmin_{\mu, \Sigma}\text{KL}({\cal N}(\theta \vert \mu, \Sigma) || c_t q_{t-1}(\theta) p(y_t \vert \theta, {\bf x}_t))
$$

where $q_{t-1}(\theta) = {\cal N}(\theta \vert \mu_{t-1}, \Sigma_{t-1})$ and
$q_0(\theta_0) = p(\theta_0)$.

---

# Overview of our methods

<!-- create a markdown table with three rows and three columns -->
| method | assumption | update cost| application
|---|---|---|---|
Subspace EKF | $\bm\theta = {\bf Az} + {\bf b}$ | $O(d^3)$ | bandits|
LoFi | $\Sigma_t = \Upsilon_t + {\bf W}_t{\bf W}_t^\intercal$ | $O((d+C)^2$) | continual learning
FSLL | $\theta = ({\bf w}, \bm\varphi = {\bf Az} + {\bf b})$ | $O(d_L^3)$ | toxic-flow prediction

---

# Low-rank EKF (LoFi)
Parameter learning with a low-rank + diagonal covariance matrix

Let ${\bf W}_t \in \mathbb{R}^{D \times d}$ be a low-rank matrix,
and let $\bm\Upsilon_t \in \mathbb{R}^{D \times D}$ be a diagonal matrix.
We take the posterior to be Gaussian with mean $\mu_t$ and precision matrix

$$
    \bm\Sigma_t^{-1} = \bm\Upsilon_t  + {\bf W}_t {\bf W}_t^\intercal
$$


---

# Subspace EKF
Parameter learning in a subspace

Let $p(y_t | \theta, {\bf x}_t) = {\cal N}(y_t \vert f(\theta, {\bf x}_t), \sigma^2)$.
$$
    \theta_t = {\bf Az}_t + {\bf b}
$$


---

# Feature-subspace last-layer (SFL2)
A financial application

Write the link function as
$f(\theta, {\bf x}_t) = \text{softmax}\left({\bf W}_t^\intercal h(\bm\varphi_t, {\bf x}_t) + b_t\right)$,
and decompose

$$
    \bm\varphi = {\bf A} {\bf z} + {\bf b}
$$

---

# Future work

1. Uncertainty calibration
1. Is the assumption of a Gaussian posterior reasonable for neural networks?