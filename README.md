# RL-Experiments

A collection of Reinforcement Learning implementations exploring fundamental algorithms from bandits to deep Q-learning. Each project demonstrates core RL concepts with clean, documented code and experimental analysis.

## Projects

| Project | Algorithms | Environment | Key Concepts |
|---------|-----------|-------------|--------------|
| [Multi-Armed Bandits](./multi-armed-bandits) | ε-Greedy, Decaying ε-Greedy, UCB | Ad Platform (10 arms) | Exploration vs Exploitation |
| [GridWorld](./grid-world) | Value Iteration, Policy Iteration | 10×10 Navigation Grid | Dynamic Programming, Bellman Equations |
| [CartPole DQN](./cart-pole-DQN) | DQN, Double DQN, PER | CartPole-v1 | Deep RL, Experience Replay, Target Networks |

---

## Multi-Armed Bandits

Explores the exploration-exploitation tradeoff in a simulated ad recommendation scenario.

**Algorithms:**
- **ε-Greedy**: Fixed exploration rate
- **Decaying ε-Greedy**: Time-decaying exploration
- **Upper Confidence Bound (UCB)**: Optimism under uncertainty

**Key Findings:**
- UCB achieves best long-term performance through principled exploration
- Decaying ε-greedy balances early exploration with later exploitation
- Fixed ε-greedy performance depends heavily on parameter tuning

[→ View Project](./multi-armed-bandits)

---

## GridWorld: Dynamic Programming

Implements classic model-based RL algorithms for navigation in a configurable grid environment with obstacles and stochastic transitions.

**Algorithms:**
- **Value Iteration**: Iterative Bellman optimality updates
- **Policy Iteration**: Alternating evaluation and improvement

**Features:**
- Configurable obstacles and slip probability
- Policy visualization with directional arrows
- Convergence analysis across different discount factors

**Key Findings:**
- Lower discount factors speed convergence but risk suboptimal policies
- Policy Iteration more robust to parameter choices
- Stochasticity significantly impacts convergence speed

| Value Iteration | Policy Iteration |
|:---------------:|:----------------:|
| ![VI](./grid-world/results/stage_2/0_9/stage_2_vi_0_9.png) | ![PI](./grid-world/results/stage_2/0_9/stage_2_pi_0_9.png) |

[→ View Project](./grid-world)

---

## CartPole DQN

Deep Reinforcement Learning implementations for continuous state-space control, progressively addressing DQN limitations.

**Algorithms:**
- **DQN**: Deep Q-Network with experience replay and target networks
- **Double DQN**: Reduces overestimation bias
- **Prioritized Experience Replay (PER)**: Samples important transitions more frequently

**Key Findings:**
- Double DQN exhibits smoother, more stable learning
- PER alone can amplify instability if TD errors are inflated
- DDQN + PER achieves best overall performance

![Training Curves](./cart-pole-DQN/results/training_curves.png)

[→ View Project](./cart-pole-DQN)

---

## Getting Started

```bash
git clone https://github.com/amirthalingamrajasundar/RL-Experiments.git
cd RL-Experiments
```

See the README in each project folder for specific setup instructions.

---

## Project Structure

```
RL-Experiments/
├── README.md                   # This file
├── multi-armed-bandits/
│   ├── main.py                 # Bandit algorithms & experiments
│   └── README.md
├── grid-world/
│   ├── env.py                  # GridWorld environment
│   ├── agents.py               # Value/Policy Iteration agents
│   ├── main.py                 # Experiment runner
│   ├── results/                # Policy visualizations
│   └── README.md
└── cart-pole-DQN/
    ├── agent.py                # DQN/DDQN agent implementations
    ├── model.py                # Neural network architecture
    ├── utils/memory.py         # Replay buffers (Uniform & PER)
    ├── main.py                 # Training & evaluation
    ├── results/                # Training curves & comparisons
    └── README.md
```

---

## Learning Path

These projects are ordered by increasing complexity:

1. **Multi-Armed Bandits** — Start here to understand the fundamental exploration-exploitation tradeoff
2. **GridWorld** — Learn how dynamic programming solves MDPs when the model is known
3. **CartPole DQN** — See how deep learning extends RL to continuous state spaces

---

