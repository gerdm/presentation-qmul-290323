---
# try also 'default' to start simple
theme: default
# random image from a curated Unsplash collection by Anthony
background: https://source.unsplash.com/collection/94734566/slidev
---

# One-pass learning methods for Bayesian neural networks
## PhD Fire Talks

Gerardo Duran-Martin

March 2023

---

# Sequential estimation

## Research question:
How can we efficiently train neural networks in one pass?

## Why?
1. Bandits
1. Online-learning problems with distribution shift
1. Finance—detecting toxic flow

---

# Bayes' rule for sequential data
A state-space representation.

Let ${\cal D}_t = ({\bf x}_t, y_t)$
$$
\begin{aligned}
    p(\theta_{t} \vert \theta_{t-1}) &= {\cal N}(\theta_t \vert \gamma \theta_{t-1}, {\bf R}_t)\\
    p(y_t \vert \theta_t, {\bf x}_t) &= \text{ExpFam}(y_t \vert \eta(\theta_t; {\bf x}_t))
\end{aligned}
$$

We seek to estimate
$$
    p(\theta_t \vert {\cal D}_{1:t}) \propto p({\cal D}_t \vert \theta_t) p(\theta_t \vert {\cal D}_{1:t-1})
$$

---

# Subspace EKF


# Low-rank EKF (LoFi)

# feature-transform last-layer (SFL2)

# Future work 
