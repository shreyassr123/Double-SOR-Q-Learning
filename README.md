# Double Successive Over-Relaxation Q-Learning with an Extension to Deep Reinforcement Learning

This repository contains the code for the numerical simulations presented in the paper:

**Double Successive Over-Relaxation Q-Learning with an Extension to Deep Reinforcement Learning**  
- Published in *IEEE Transactions on Neural Networks and Learning Systems* → [IEEE Xplore](https://ieeexplore.ieee.org/document/11048710)  
- Preprint available on [arXiv](https://doi.org/10.48550/arXiv.2409.06356)  

## Project Structure

### Tabular Version

The `Tabular_Version` folder contains Python code to run environments and experiments for:

- **Roulette**
- **Grid World**

This folder also includes the necessary files for plotting results and analyzing performance.

### Deep RL Version

The `Deep_RL_Version` folder includes code for running:

- **CartPole Environment**
- **Lunar Lander Environment**
- **Maximization Bias Example**
- **Atari 2600 Games** (via MushroomRL)

These implementations leverage deep reinforcement learning techniques to explore the effectiveness of Double SOR Q-Learning in more complex scenarios.

---

#### Parameters for Maximization Bias Example

| Parameter                 | Value                          |
|---------------------------|--------------------------------|
| Network architecture      | Fully connected                |
| Number of hidden layers   | 2                              |
| Hidden layer sizes        | 4, 8                           |
| Activation function       | ReLU                           |
| Optimizer                 | SGD                            |
| Discount factor (γ)       | 0.999                          |
| Exploration strategy      | Epsilon-greedy                 |
| Epsilon (ε)               | 0.1                            |
| Number of states          | 1e9 + 2                        |
| Number of actions         | 2 (Left, Right)                |
| Number of training episodes | 400                          |
| Number of iterations      | 1000                           |

---

#### Network Architecture and Training Parameters for Atari

| Component/Parameter       | Details/Value                  |
|---------------------------|--------------------------------|
| Evaluation Environment    | Six Atari 2600 games using MushroomRL |
| Input Representation      | 84×84 grayscale frames, 4-frame stack, normalized to [0,1] |
| Network Architecture      | CNN: 3 conv layers + 2 fully connected layers |
| Conv Layers               | 32 filters (8×8, stride 4), 64 filters (4×4, stride 2), 64 filters (3×3, stride 1) |
| Fully Connected Layer     | 512 hidden units, output for action values |
| Loss Function             | Huber loss (smooth L1)         |
| Exploration Policy        | Epsilon-greedy (ε decays 1.0 → 0.1 over 1M steps, eval ε=0.05) |
| History length            | 4                              |
| Training frequency        | Every 4 steps                  |
| Evaluation frequency      | Every 250,000 steps            |
| Target update frequency   | Every 10,000 steps             |
| Initial replay size       | 50,000                         |
| Max replay size           | 500,000                        |
| Test samples              | 125,000                        |
| Max training steps        | 2,500,000                      |
| Batch size                | 32                             |
| Discount factor (γ)       | 0.99                           |
| Optimizer                 | Adam (lr = 0.00025)            |
| Reported Metrics          | Mean reward over 10 epochs ± std across 5 seeds |

---

#### Parameters for Rainbow, SORDQN, and DSORDQN (Atari)

| Parameter                 | Value                          |
|---------------------------|--------------------------------|
| Number of atoms (n_atoms) | 51                             |
| Value range (v_min, v_max)| (-10, 10)                      |
| Multi-step return (n)     | 3                              |
| Prioritization exponent (α)| 0.6                           |
| Priority correction factor (β) | Linear (0.4 → 1)           |
| Noisy network parameter (σ)| 0.5                           |
| SOR Parameter (w) for SORDQN & DSORDQN | 1.3               |

---

**References**  
- Hessel et al. (2018). *Rainbow: Combining improvements in deep reinforcement learning.* AAAI.  
- John & Bhatnagar (2020). *Deep reinforcement learning with successive over-relaxation and its application in autoscaling cloud resources.* IJCNN.  
- Weng et al. (2020). *The mean-squared error of double Q-learning.* NeurIPS.  

## Acknowledgements

### Tabular Version

The numerical implementation of the tabular Double SOR Q-Learning version, including the roulette and grid world environments, is based on the examples presented in the paper:

- **Double Q-Learning** by Hado van Hasselt (2010), NeurIPS.

We acknowledge the contributions of Hado van Hasselt and the NeurIPS conference for providing foundational methods that have influenced this work. The code and methodology are inspired by and extend the ideas described in this seminal paper.

### Deep RL Version

The code for the following environments is sourced from the specified GitHub pages:

- **CartPole and Lunar Lander Environments**: [Stochastic Expatriate Descent - Double DQN](https://github.com/davidrpugh/stochastic-expatriate-descent/blob/2020-04-11-double-dqn/_notebooks/2020-04-11-double-dqn.ipynb)
- **Maximization Bias Example**: [The Mean Squared Error of Double Q-Learning - Bias(nn)](https://github.com/wentaoweng/The-Mean-Squared-Error-of-Double-Q-Learning/tree/main/Bias(nn))

We appreciate the authors and maintainers of these repositories for their contributions, which have been instrumental in the development of this work. The provided code and methods have been adapted and extended to fit the context of this project.
