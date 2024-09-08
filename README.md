# Double Successive Over-Relaxation Q-Learning with an Extension to Deep Reinforcement Learning

This repository contains codes related to the paper on Double Successive Over-Relaxation Q-Learning with an extension to Deep Reinforcement Learning (DRL). The project is organized into two main folders: one for the tabular version of Double SOR Q-Learning and one for the Deep RL version.

## Project Structure

### Tabular Version

The `Tabular_Version` folder contains Python code to run environments and experiments for:

- **Roulette Environment**
- **Grid World Environment**

This folder also includes the necessary files for plotting results and analyzing performance.

### Deep RL Version

The `Deep_RL_Version` folder includes code for running:

- **CartPole Environment**
- **Lunar Lander Environment**
- **Maximization Example**

These implementations leverage deep reinforcement learning techniques to explore the effectiveness of Double SOR Q-Learning in more complex scenarios.

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
