---
title: "Multi-Armed Bandit Problem and Its Solutions"
date: 2023-08-02T14:21:44+08:00
draft: false
tags: 
- "reinforcement-learning"
- "exploration-exploitation"
math: mathjax
---

In probability theory and decision-making under uncertainty, the multi-armed bandit problem presents a challenge where a limited set of resources must be wisely allocated among competing choices to maximize the expected gain. This is a classic reinforcement learning problem that exemplifies the exploration-exploitation tradeoff dilemma. 

Imagine a gambler facing a row of slot machines (also called [one-armed bandits](https://en.wiktionary.org/wiki/one-armed_bandit)). The gambler must make a series of decisions: which machines to play, how many times to play each machine, the order in which to play them, and whether to stick with the current machine or switch to another one. In this setup, each machine provides a random reward from an unknown probability distribution. The primary objective of the gambler (or agent) is to maximize the total reward obtained over a series of plays. 

At each trial, the agent faces a crucial tradeoff: exploitation or exploration. In exploration, the agent tries out different arms to gather information about their rewards and estimate their true potential. The more exploration is done, the better the agent can learn about each arm's payoffs. In contrast, exploitation involves making decisions based on the current knowledge to choose the arm expected to yield the highest reward.

Effectively balancing exploration and exploitation is the key challenge to maximize cumulative rewards over time. If the agent exploits too much, it may miss out on higher-reward arms. On the other hand, excessive exploration could lead to missed opportunities to gain higher rewards from known better arms.

To address the multi-armed bandit problem, various strategies and algorithms have been developed, including epsilon-greedy, optimistic initialization, upper confidence bound (UCB), Thompson sampling, and gradient bandit methods. Each of these approaches aims to tackle the exploration-exploitation dilemma and optimize the agent's decision-making.

## Epsilon-Greedy

The action value is estimated according to past experience by averaging the rewards associated with the target action $a$ that we have observed so far (up to the current time step t).

$$
\hat{Q}\_t(a) = \frac{1}{N\_t(a)} \sum\_{\tau=1}^t r\_\tau \mathbb{1}[a\_\tau = a]
$$

where $\mathbb{1}$ is a binary indicator function and $N_t(a)$ is how many times the action a has been selected so far, $N_t(a) = \sum_{\tau=1}^t \mathbb{1}[a_\tau = a]$.

The simplest action that the agent can make is to select one of the actions with the highest estimated value, that is, one of the greedy actions defined as:

$$
A_t = \arg \max_a \hat{Q}_t(a)
$$

The greedy action prioritizes exploiting existing knowledge to achieve the maximum immediate reward and does not invest any time in sampling other actions to explore if they could yield better results.

One simple alternative to this is to take greedy action most of the time, but with a small probability (say $\epsilon$), select random action instead. This method is known as the epsilon-greedy method.

$$
A_t = \begin{cases}
    \arg \max_a \hat{Q}_t(a) & \text{with probability 1 -  } \epsilon \\\
    \text{a random action} & \text{with probability } \epsilon \\
\end{cases}
$$

## Optimistic Initialization

The idea behind this method is to initialize the estimated action value, $Q_0(a)$ to a high value, higher than their actual estimated action value. The agent then updates action value by incremental averaging, starting with $N_0(a) \ge 0$ for all $a \in \mathcal{A}$,

$$
\hat{Q}_t(A_t) = \hat{Q}\_{t-1}(A_t) + {1 \over N_t(a)}(r_t - \hat{Q}\_{t-1}(A_t)), \text{ and}
$$
$$
\hat{Q}_t(a) = \hat{Q}\_{t-1}(a) \text{ for all } a \ne A_t 
$$

This method promotes systematic exploration in the initial stages. When the agent chooses actions initially, the rewards received are lower than the starting estimates. Consequently, the agent switches to other actions as it becomes "disappointed" with the received rewards. This leads to repeated trials of all actions before the value estimates eventually converge.

## Upper Confidence Bound

Exploration is important because we are not always sure how accurate our action-value estimates are. Greedy method choose an action that looks best at time $t$, but there might be better options among the other actions. Epsilon-greedy forces us to try non-greedy actions, but it does so randomly, without considering which non-greedy actions are almost as good as the greedy ones or which ones have more uncertainty.

A better approach would be to choose non-greedy actions based on their potential to be the best choices. This means considering how close their estimates are to the highest possible values and how uncertain those estimates are.

One effective way of doing this is to select actions according to:

$$
A_t = \arg \max_a [Q_t(a) + c \sqrt{\ln t \over N_t(a)}]
$$

where $\ln t$ denotes the natural logarithm of t, and the number $c \gt 0$ control degree of exploration.

The square-root term is a measure of the uncertainty in the estimate of action $a$’s value. Each time action $a$ selected, the uncertainty decrease and, conversely, each time action other than $a$ is selected, the uncertainty increase. The use of the natural logarithm means that the increments become progressively smaller over time, but they have no upper limit. As a result, all actions will eventually be chosen, but actions with lower value estimates or those that have been frequently selected before will be picked less frequently as time goes on.

## Thompson Sampling

Thompson sampling model the uncertainty in the reward distributions of arms using probability distributions, particularly the Beta distribution. Each arm is associated with a Beta distribution that represents the agent's belief or uncertainty about the true mean reward of that arm. 

The algorithm work as follow:

1. For each arm, initialize the parameters of the Beta distribution based on some prior belief. The choice of prior can influence the algorithm's behavior, but common choices are uniform or optimistic priors. For example, we can set α = 1 and β = 1, where we expect the reward probability to be 50% but we are not very confident.
2. At each time step, sample an expected reward from each arm's  $\text{Beta}(\alpha_i, \beta_i)$ distribution independently. the best action is selected among samples: $a_t = \arg\max \tilde{Q}(a)$
3. The sampled reward is used to update the parameters of the corresponding Beta distribution for that arm, incorporating the new information.
    
    $$
    \alpha_i \leftarrow \alpha_i + r_t \mathbb{1}[a_t = a_i] 
    $$
    $$
    \beta_i \leftarrow \beta_i + (1-r_t) \mathbb{1}[a_t = a_i]
    $$
    
4. Continue the process by selecting arms, pulling, observing rewards, and updating the Beta distributions at each time step.

Thompson sampling implements the idea of [probability matching](https://en.wikipedia.org/wiki/Probability_matching). Because its reward estimations $\tilde{Q}$ are sampled from posterior distributions, each of these probabilities is equivalent to the probability that the corresponding action is optimal, conditioned on observed history.

## Gradient Bandit Algorithm

So far, we have explored methods that estimate action values and use those estimates to select actions. Another approach is to consider learning a numerical preference for each action $a$, which we denote as $H_t(a)$. A higher preference value corresponds to a more frequent selection of the action, but the preference itself does not hold any direct reward interpretation. 

Initially, all action preferences are set equally, ensuring that all actions have an equal probability of being chosen. After each step (where action $A_t$ is selected, and reward $R_t$ is received), the action preferences are updated by:

$$
H_{t+1}(A_t) = H_t(A_t) + \alpha(R_t - \bar{R_t})(1 - \pi_t(A_t)), \text{ and} 
$$
$$
H_{t+1}(a) = H_t(a) - \alpha(R_t, -\bar{R_t})\pi_t(a),\text{ for all } \ a \ne A_t
$$

where $\pi_t(a) = {e^{H_t(a)} \over \sum_{b=1}^ke^{H_t(b)}}$ is the probability of taking action $a$ at time $t$, $\alpha \gt 0$ is a step-size parameter, and $\bar{R_t} \in \mathbb{R}$ is the average of all the rewards up to but not including time $t$.

If the reward ($R_t$) is higher than the baseline ($\bar{R_t}$), the probability of taking $A_t$ in the future increase and vice versa. The non-selected actions move in the opposite direction.

## References

[1] RL Course by David Silver - Lecture 9: [Exploration and Exploitation](https://youtu.be/sGuiWX07sKw)

[2] Stanford CME 241 slide - [Multi-Armed Bandits: Exploration versus Exploitation](https://stanford.edu/~ashlearn/RLForFinanceBook/MultiArmedBandits.pdf)

[3] Reinforcement Learning: An Introduction by Richard S. Sutton and Andrew G. Barto

[4] Lilian Weng - [The Multi-Armed Bandit Problem and Its Solutions](https://lilianweng.github.io/posts/2018-01-23-multi-armed-bandit/)