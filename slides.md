---
# try also 'default' to start simple
theme: default
background: /cover-lofi-dalle.png
---

# One-pass learning methods for Bayesian neural networks
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
How to update the parameter beliefs, one observation at a time?

Let ${\cal D}_t = ({\bf x}_t, y_t)$.  If we have an estimate for $p(\theta \vert {\cal D}_{1:t-1})$, we can use Bayes' rule to update the posterior
after observing ${\cal D}_t$:

$$
    p(\theta \vert {\cal D}_{1:t}) \propto p({\cal D}_t \vert \theta) p(\theta \vert {\cal D}_{1:t-1})
$$

---

# Bayes' rule for sequential data
(continued)

How to estimate (or approximate) $p(\theta \vert {\cal D}_{1:t})$?

1. KF and variants: EKF, UKF, SLF.
2. Particle filters: SIS, SMC, VSMC.
3. R-VGA, Expfam EKF.

---

# Regression example
The extended Kalman filter (EKF) for neural networks.

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
by the extended Kalman filter update equations.

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
3. Continuously changing environment (catastrophic forgetting).
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

# Overview of our methods

<!-- create a markdown table with three rows and three columns -->
| method | assumption | update cost| application
|---|---|---|---|
Subspace EKF | $\bm\theta = {\bf Az} + {\bf b}$ | $O(d^3)$ | bandits|
LoFi | $\Sigma_t = \Upsilon_t + {\bf W}_t{\bf W}_t^\intercal$ | $O((d+C)^2$) | continual learning
FSLL | $\theta = ({\bf w}, \bm\varphi = {\bf Az} + {\bf b})$ | $O(d_L^3)$ | toxic-flow prediction

---

# LoFi for continual learning
(An example)

<!-- add video with path "lofi-posterior-predictive -->
<video class="horizontal-center" width=500 controls muted autoplay>
  <source src="/lofi-posterior-predictive.mp4" type="video/mp4">
</video>

--

# One final thought
What about LLMs?

<img class="horizontal-center" width=700
     src="/chat-gpt-demo.png"/>

---

# References

* Duran-Martin, G., Kara, A., & Murphy, K. (2021). Efficient Online Bayesian Inference for Neural Bandits. ArXiv [Cs.LG]. Retrieved from http://arxiv.org/abs/2112.00195

* Chang P., Duran-Martin G, Shestopaloff A, Jones M, & Murphy K. (2023). Low-rank extended Kalman filtering for online learning of neural networks from streaming data. (submitted)

* Cartea A., Duran-Martin G, & Sanchez-Betancourt L. (2023). Detecting toxic flow: A sequential Bayes approach. (work in progress)

* Lambert, M., Bonnabel, S. & Bach, F. The recursive variational Gaussian approximation (R-VGA). Stat Comput 32, 10 (2022). https://doi.org/10.1007/s11222-021-10068-w

* Haykin, S. (2001). Kalman Filters. In Kalman Filtering and Neural Networks (pp. 1–21). doi:10.1002/0471221546.ch1

* Ollivier, Y. (2018). Online Natural Gradient as a Kalman Filter. ArXiv [Stat.ML]. Retrieved from http://arxiv.org/abs/1703.00209