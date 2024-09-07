

import torch
from torch import nn

import typing

import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F

import collections
import typing

import gym



def select_greedy_actions(states: torch.Tensor, q_network: nn.Module) -> torch.Tensor:
    """Select the greedy action for the current state given some Q-network."""
    _, actions = q_network(states).max(dim=1, keepdim=True)
    return actions


def evaluate_selected_actions(states: torch.Tensor,
                              actions: torch.Tensor,
                              rewards: torch.Tensor,
                              dones: torch.Tensor,
                              gamma: float,
                              q_network: nn.Module) -> torch.Tensor:
    """Compute the Q-values by evaluating the actions given the current states and Q-network."""
    next_q_values = q_network(states).gather(dim=1, index=actions)
    q_values = rewards + (gamma * next_q_values * (1 - dones))
    return q_values

def evaluate_selected_actions_sor(states: torch.Tensor,
                              actions: torch.Tensor,
                              rewards: torch.Tensor,
                              old_states: torch.Tensor,
                              old_actions: torch.Tensor,
                              dones: torch.Tensor,
                              gamma: float,
                              q_network: nn.Module) -> torch.Tensor:
    
    """Compute the Q-values by evaluating the actions given the current states and Q-network."""
    next_q_values = q_network(states).gather(dim=1, index=actions)
    old_q_values = q_network(old_states).gather(dim=1, index=old_actions)
    w = 1.3
    dones_tensor = torch.ones_like(dones, dtype=torch.bool) # Create a tensor of True values
    if (dones == dones_tensor).all():
      q_values = rewards
     
    else:
      q_values = w * (rewards + (gamma * next_q_values * (1 - dones))) + (1 - w) * old_q_values

    return q_values


def q_learning_update(states: torch.Tensor,
                      rewards: torch.Tensor,
                      dones: torch.Tensor,
                      gamma: float,
                      q_network: nn.Module) -> torch.Tensor:
    """Q-Learning update with explicitly decoupled action selection and evaluation steps."""
    actions = select_greedy_actions(states, q_network)
    q_values = evaluate_selected_actions(states, actions, rewards, dones, gamma, q_network)
    return q_values

def sorq_learning_update(states: torch.Tensor,
                      rewards: torch.Tensor,
                      old_states: torch.Tensor,
                      dones: torch.Tensor,
                      gamma: float,
                      q_network: nn.Module) -> torch.Tensor:
    """SORQ-Learning update with explicitly decoupled action selection and evaluation steps."""
    actions = select_greedy_actions(states, q_network)
    old_actions = select_greedy_actions(old_states, q_network)
    q_values = evaluate_selected_actions_sor(states, actions, rewards, old_states, old_actions, dones, gamma, q_network)
    return q_values



def double_q_learning_update(states: torch.Tensor,
                             rewards: torch.Tensor,
                             dones: torch.Tensor,
                             gamma: float,
                             q_network_1: nn.Module,
                             q_network_2: nn.Module) -> torch.Tensor:
    """Double Q-Learning uses Q-network 1 to select actions and Q-network 2 to evaluate the selected actions."""
    actions = select_greedy_actions(states, q_network_1)
    q_values = evaluate_selected_actions(states, actions, rewards, dones, gamma, q_network_2)
    return q_values

def sordouble_q_learning_update(states: torch.Tensor,
                             rewards: torch.Tensor,
                             old_states: torch.Tensor,
                             dones: torch.Tensor,
                             gamma: float,
                             q_network_1: nn.Module,
                             q_network_2: nn.Module) -> torch.Tensor:
    """SOR Double Q-Learning uses Q-network 1 to select actions and Q-network 2 to evaluate the selected actions."""
    actions = select_greedy_actions(states, q_network_1)
    old_actions = select_greedy_actions(old_states, q_network_1)
    q_values = evaluate_selected_actions_sor(states, actions, rewards, old_states, old_actions, dones, gamma, q_network_2)
    return q_values



"""Note that the function `double_q_learning_update` and `sordouble_q_learning_update` is almost identical 
to the `q_learning_update` and `sorq_learning_update` function above. All that is needed is to 
introduce a second Q-network parameter, `q_network_2`, to the function. 
This second Q-network will be use to evaluate the actions chosen using the 
original Q-network parameter, now called `q_network_1`.

### Experience Replay

Just like the DQN algorithm, the Double DQN, SORDQN, and SORDDQN algorithms use an `ExperienceReplayBuffer` 
to stabilize the learning process.
"""

import collections
import typing

import numpy as np


_field_names = [
    "state",
    "action",
    "reward",
    "next_state",
    "done"
]


Experience = collections.namedtuple("Experience", field_names=_field_names)


class ExperienceReplayBuffer:
    """Fixed-size buffer to store Experience tuples."""

    def __init__(self,
                 batch_size: int,
                 buffer_size: int = None,
                 random_state: np.random.RandomState = None) -> None:
        """
        Initialize an ExperienceReplayBuffer object.

        Parameters:
        -----------
        buffer_size (int): maximum size of buffer
        batch_size (int): size of each training batch
        random_state (np.random.RandomState): random number generator.

        """
        self._batch_size = batch_size
        self._buffer_size = buffer_size
        self._buffer = collections.deque(maxlen=buffer_size)
        self._random_state = np.random.RandomState() if random_state is None else random_state

    def __len__(self) -> int:
        return len(self._buffer)

    @property
    def batch_size(self) -> int:
        """Number of experience samples per training batch."""
        return self._batch_size

    @property
    def buffer_size(self) -> int:
        """Total number of experience samples stored in memory."""
        return self._buffer_size

    def append(self, experience: Experience) -> None:
        """Add a new experience to memory."""
        self._buffer.append(experience)

    def sample(self) -> typing.List[Experience]:
        """Randomly sample a batch of experiences from memory."""
        idxs = self._random_state.randint(len(self._buffer), size=self._batch_size)
        experiences = [self._buffer[idx] for idx in idxs]
        return experiences




class Agent:

    def choose_action(self, state: np.array) -> int:
        """Rule for choosing an action given the current state of the environment."""
        raise NotImplementedError

    def learn(self, experiences: typing.List[Experience]) -> None:
        """Update the agent's state based on a collection of recent experiences."""
        raise NotImplementedError

    def save(self, filepath) -> None:
        """Save any important agent state to a file."""
        raise NotImplementedError

    def step(self,
             state: np.array,
             action: int,
             reward: float,
             next_state: np.array,
             done: bool) -> None:
        """Update agent's state after observing the effect of its action on the environment."""
        raise NotImplementedError


class DeepQAgent(Agent):

    def __init__(self,
                 state_size: int,
                 action_size: int,
                 number_hidden_units: int,
                 optimizer_fn: typing.Callable[[typing.Iterable[nn.Parameter]], optim.Optimizer],
                 batch_size: int,
                 buffer_size: int,
                 epsilon_decay_schedule: typing.Callable[[int], float],
                 alpha: float,
                 gamma: float,
                 update_frequency: int,
                 double_dqn: bool = False,
                 sordouble_dqn: bool = False,
                 sor_dqn: bool = False,
                 seed: int = None) -> None:
        """
        Initialize a DeepQAgent.

        Parameters:
        -----------
        state_size (int): the size of the state space.
        action_size (int): the size of the action space.
        number_hidden_units (int): number of units in the hidden layers.
        optimizer_fn (callable): function that takes Q-network parameters and returns an optimizer.
        batch_size (int): number of experience tuples in each mini-batch.
        buffer_size (int): maximum number of experience tuples stored in the replay buffer.
        epsilon_decay_schdule (callable): function that takes episode number and returns epsilon.
        alpha (float): rate at which the target q-network parameters are updated.
        gamma (float): Controls how much that agent discounts future rewards (0 < gamma <= 1).
        update_frequency (int): frequency (measured in time steps) with which q-network parameters are updated.
        double_dqn (bool): whether to use vanilla DQN algorithm or use the Double DQN algorithm.
        sor_dqn (bool): whether to use vanilla DQN algorithm or use the SORDQN algorithm.
        seed (int): random seed

        """
        self._state_size = state_size
        self._action_size = action_size
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # set seeds for reproducibility
        self._random_state = np.random.RandomState() if seed is None else np.random.RandomState(seed)
        if seed is not None:
            torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        # initialize agent hyperparameters
        _replay_buffer_kwargs = {
            "batch_size": batch_size,
            "buffer_size": buffer_size,
            "random_state": self._random_state
        }
        self._memory = ExperienceReplayBuffer(**_replay_buffer_kwargs)
        self._epsilon_decay_schedule = epsilon_decay_schedule
        self._alpha = alpha
        self._gamma = gamma
        self._double_dqn = double_dqn
        self._sordouble_dqn = sordouble_dqn
        self._sor_dqn = sor_dqn
        # initialize Q-Networks
        self._update_frequency = update_frequency
        self._online_q_network = self._initialize_q_network(number_hidden_units)
        self._target_q_network = self._initialize_q_network(number_hidden_units)
        self._synchronize_q_networks(self._target_q_network, self._online_q_network)
        self._online_q_network.to(self._device)
        self._target_q_network.to(self._device)

        # initialize the optimizer
        self._optimizer = optimizer_fn(self._online_q_network.parameters())

        # initialize some counters
        self._number_episodes = 0
        self._number_timesteps = 0

    def _initialize_q_network(self, number_hidden_units: int) -> nn.Module:
        """Create a neural network for approximating the action-value function."""
        q_network = nn.Sequential(
            nn.Linear(in_features=self._state_size, out_features=number_hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=number_hidden_units, out_features=number_hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=number_hidden_units, out_features=self._action_size)
        )
        return q_network

    @staticmethod
    def _soft_update_q_network_parameters(q_network_1: nn.Module,
                                          q_network_2: nn.Module,
                                          alpha: float) -> None:
        """In-place, soft-update of q_network_1 parameters with parameters from q_network_2."""
        for p1, p2 in zip(q_network_1.parameters(), q_network_2.parameters()):
            p1.data.copy_(alpha * p2.data + (1 - alpha) * p1.data)

    @staticmethod
    def _synchronize_q_networks(q_network_1: nn.Module, q_network_2: nn.Module) -> None:
        """In place, synchronization of q_network_1 and q_network_2."""
        _ = q_network_1.load_state_dict(q_network_2.state_dict())

    def _uniform_random_policy(self, state: torch.Tensor) -> int:
        """Choose an action uniformly at random."""
        return self._random_state.randint(self._action_size)

    def _greedy_policy(self, state: torch.Tensor) -> int:
        """Choose an action that maximizes the action_values given the current state."""
        action = (self._online_q_network(state)
                      .argmax()
                      .cpu()  # action_values might reside on the GPU!
                      .item())
        return action

    def _epsilon_greedy_policy(self, state: torch.Tensor, epsilon: float) -> int:
        """With probability epsilon explore randomly; otherwise exploit knowledge optimally."""
        if self._random_state.random() < epsilon:
            action = self._uniform_random_policy(state)
        else:
            action = self._greedy_policy(state)
        return action

    def choose_action(self, state: np.array) -> int:
        """
        Return the action for given state as per current policy.

        Parameters:
        -----------
        state (np.array): current state of the environment.

        Return:
        --------
        action (int): an integer representing the chosen action.

        """
        # need to reshape state array and convert to tensor
        state_tensor = (torch.from_numpy(state)
                             .unsqueeze(dim=0)
                             .to(self._device))

        # choose uniform at random if agent has insufficient experience
        if not self.has_sufficient_experience():
            action = self._uniform_random_policy(state_tensor)
        else:
            epsilon = self._epsilon_decay_schedule(self._number_episodes)
            action = self._epsilon_greedy_policy(state_tensor, epsilon)
        return action

    def learn(self, experiences: typing.List[Experience]) -> None:
        """Update the agent's state based on a collection of recent experiences."""
        states, actions, rewards, next_states, dones = (torch.Tensor(vs).to(self._device) for vs in zip(*experiences))

        # need to add second dimension to some tensors
        actions = (actions.long()
                          .unsqueeze(dim=1))
        rewards = rewards.unsqueeze(dim=1)
        dones = dones.unsqueeze(dim=1)

        if self._double_dqn:
            target_q_values = double_q_learning_update(next_states,
                                                       rewards,
                                                       dones,
                                                       self._gamma,
                                                       self._online_q_network,
                                                       self._target_q_network)
        elif self._sor_dqn:
            target_q_values = sorq_learning_update(next_states,
                                                rewards,
                                                states,
                                                dones,
                                                self._gamma,
                                                self._target_q_network)

        elif self._sordouble_dqn:
            target_q_values = sordouble_q_learning_update(next_states,
                                                rewards,
                                                states,
                                                dones,
                                                self._gamma,
                                                self._online_q_network,
                                                self._target_q_network)

        else:
            target_q_values = q_learning_update(next_states,
                                                rewards,
                                                dones,
                                                self._gamma,
                                                self._target_q_network)

        online_q_values = (self._online_q_network(states)
                               .gather(dim=1, index=actions))

        # compute the mean squared loss
        loss = F.mse_loss(online_q_values, target_q_values)

        # updates the parameters of the online network
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        self._soft_update_q_network_parameters(self._target_q_network,
                                               self._online_q_network,
                                               self._alpha)

    def has_sufficient_experience(self) -> bool:
        """True if agent has enough experience to train on a batch of samples; False otherwise."""
        return len(self._memory) >= self._memory.batch_size

    def save(self, filepath: str) -> None:
        """
        Saves the state of the DeepQAgent.

        Parameters:
        -----------
        filepath (str): filepath where the serialized state should be saved.

        Notes:
        ------
        The method uses `torch.save` to serialize the state of the q-network,
        the optimizer, as well as the dictionary of agent hyperparameters.

        """
        checkpoint = {
            "q-network-state": self._online_q_network.state_dict(),
            "optimizer-state": self._optimizer.state_dict(),
            "agent-hyperparameters": {
                "alpha": self._alpha,
                "batch_size": self._memory.batch_size,
                "buffer_size": self._memory.buffer_size,
                "gamma": self._gamma,
                "update_frequency": self._update_frequency
            }
        }
        torch.save(checkpoint, filepath)

    def step(self,
             state: np.array,
             action: int,
             reward: float,
             next_state: np.array,
             done: bool) -> None:
        """
        Updates the agent's state based on feedback received from the environment.

        Parameters:
        -----------
        state (np.array): the previous state of the environment.
        action (int): the action taken by the agent in the previous state.
        reward (float): the reward received from the environment.
        next_state (np.array): the resulting state of the environment following the action.
        done (bool): True is the training episode is finised; false otherwise.

        """
        experience = Experience(state, action, reward, next_state, done)
        self._memory.append(experience)

        if done:
            self._number_episodes += 1
        else:
            self._number_timesteps += 1

            # every so often the agent should learn from experiences
            if self._number_timesteps % self._update_frequency == 0 and self.has_sufficient_experience():
                experiences = self._memory.sample()
                self.learn(experiences)





def _train_for_at_most(agent: Agent, env: gym.Env, max_timesteps: int) -> int:
    """Train agent for a maximum number of timesteps."""
    state = env.reset()
    score = 0
    for t in range(max_timesteps):
        action = agent.choose_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.step(state, action, reward, next_state, done)
        state = next_state
        score += reward
        if done:
            break
    return score


def _train_until_done(agent: Agent, env: gym.Env) -> float:
    """Train the agent until the current episode is complete."""
    state = env.reset()
    score = 0
    done = False
    while not done:
        action = agent.choose_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.step(state, action, reward, next_state, done)
        state = next_state
        score += reward
    return score


def train(agent: Agent,
          env: gym.Env,
          checkpoint_filepath: str,
          target_score: float,
          number_episodes: int,
          maximum_timesteps=None) -> typing.List[float]:
    """
    Reinforcement learning training loop.

    Parameters:
    -----------
    agent (Agent): an agent to train.
    env (gym.Env): an environment in which to train the agent.
    checkpoint_filepath (str): filepath used to save the state of the trained agent.
    number_episodes (int): maximum number of training episodes.
    maximum_timsteps (int): maximum number of timesteps per episode.

    Returns:
    --------
    scores (list): collection of episode scores from training.

    """
    scores = []
    most_recent_scores = collections.deque(maxlen=100)
    for i in range(number_episodes):
        if maximum_timesteps is None:
            score = _train_until_done(agent, env)
        else:
            score = _train_for_at_most(agent, env, maximum_timesteps)
        scores.append(score)
        most_recent_scores.append(score)

        average_score = sum(most_recent_scores) / len(most_recent_scores)
        if average_score >= target_score:
            print(f"\nEnvironment solved in {i:d} episodes!\tAverage Score: {average_score:.2f}")
            agent.save(checkpoint_filepath)
            break
        if (i + 1) % 100 == 0:
            print(f"\rEpisode {i + 1}\tAverage Score: {average_score:.2f}")

    return scores





import gym

''' One can change the enviorments here'''

#env = gym.make('CartPole-v1')
env = gym.make('LunarLander-v2')

_ = env.seed(42)


"""### Creating a `DeepQAgent`

Before creating an instance of the `DeepQAgent` we need to define an $\epsilon$-decay schedule 
and choose an optimizer.

#### Epsilon decay schedule


"""

def power_decay_schedule(episode_number: int,
                         decay_factor: float,
                         minimum_epsilon: float) -> float:
    """Power decay schedule found in other practical applications."""
    return max(decay_factor**episode_number, minimum_epsilon)

_epsilon_decay_schedule_kwargs = {
    "decay_factor": 0.99,
    "minimum_epsilon": 1e-2,
}
epsilon_decay_schedule = lambda n: power_decay_schedule(n, **_epsilon_decay_schedule_kwargs)

"""#### Choosing an optimizer

As is the case in training any neural network, the choice of optimizer and the tuning of its 
hyper-parameters (in particular the learning rate) is important. 
"""

from torch import optim


_optimizer_kwargs = {
    "lr": 1e-3,
    "betas": (0.9, 0.999),
    "eps": 1e-08,
    "weight_decay": 0,
    "amsgrad": False,
}
optimizer_fn = lambda parameters: optim.Adam(parameters, **_optimizer_kwargs)




"""Training the DeepQAgent using SOR Double DQN

### Training the `DeepQAgent` using Double DQN


Now we are finally ready to train the `DQN`, `SORDQN`, `DDQN`, and `SORDDQN`. 
The target score for the `LunarLander-v2` environment is 200 points on average for 
at least 100 consecutive episodes. 
If any algorithm is able to "solve" the environment, then training will terminate early.

Similarly, The target score for the `CartPole-v1` environment is 195 points on average for 
at least 100 consecutive episodes.

"""


#########################################################

print("The Performance of SOR Q-Learning Algorithm")

'''   SORDQN ALGORITHM '''

_agent_kwargs = {
    "state_size": env.observation_space.shape[0],
    "action_size": env.action_space.n,
    "number_hidden_units": 64,
    "optimizer_fn": optimizer_fn,
    "epsilon_decay_schedule": epsilon_decay_schedule,
    "batch_size": 64,
    "buffer_size": 100000,
    "alpha": 1e-3,
    "gamma": 0.99,
    "update_frequency": 4,
    "double_dqn": False,  # True uses Double DQN; False uses DQN
    "sordouble_dqn": False,  # True uses SORDQN; False uses DQN
    "sor_dqn": True,  # True uses SORDQN; False uses DQN
    "seed": 2024,
}


sor_dqn_agent = DeepQAgent(**_agent_kwargs)

sordqn_scores = train(sor_dqn_agent,
                          env,
                          "sor-dqn-checkpoint.pth",
                          number_episodes=5000,
                          target_score=200)


print("-----------------------------------------------------------")

#########################################################

'''   SORDDQN ALGORITHM '''

print("The Performance of the proposed SOR Double Q-Learning Algorithm")


_agent_kwargs = {
    "state_size": env.observation_space.shape[0],
    "action_size": env.action_space.n,
    "number_hidden_units": 64,
    "optimizer_fn": optimizer_fn,
    "epsilon_decay_schedule": epsilon_decay_schedule,
    "batch_size": 64,
    "buffer_size": 100000,
    "alpha": 1e-3,
    "gamma": 0.99,
    "update_frequency": 4,
    "double_dqn": False,  # True uses Double DQN; False uses DQN
    "sordouble_dqn": True,  # True uses SORDQN; False uses DQN
    "sor_dqn": False,  # True uses SORDQN; False uses DQN
    "seed": 2024,
}

sordouble_dqn_agent = DeepQAgent(**_agent_kwargs)

sordouble_dqn_scores = train(sordouble_dqn_agent,
                          env,
                          "sordouble-dqn-checkpoint.pth",
                          number_episodes=5000,
                          target_score=200)


print("-----------------------------------------------------------")
#########################################################

print("The Performance of the Double Q-Learning Algorithm")



'''   DDQN ALGORITHM '''

_agent_kwargs = {
    "state_size": env.observation_space.shape[0],
    "action_size": env.action_space.n,
    "number_hidden_units": 64,
    "optimizer_fn": optimizer_fn,
    "epsilon_decay_schedule": epsilon_decay_schedule,
    "batch_size": 64,
    "buffer_size": 100000,
    "alpha": 1e-3,
    "gamma": 0.99,
    "update_frequency": 4,
    "double_dqn": True,  # True uses Double DQN; False uses DQN
    "sordouble_dqn": False,  # True uses SORDQN; False uses DQN
    "sor_dqn": False,  # True uses SORDQN; False uses DQN
    "seed": 2024,
}
double_dqn_agent = DeepQAgent(**_agent_kwargs)

double_dqn_scores = train(double_dqn_agent,
                          env,
                          "double-dqn-checkpoint.pth",
                          number_episodes=5000,
                          target_score=200 # hack to insure that training never terminates early
                         )


print("-----------------------------------------------------------")

#########################################################

print("At last, the Performance of the Q-Learning Algorithm")



'''   DQN ALGORITHM '''

_agent_kwargs = {
    "state_size": env.observation_space.shape[0],
    "action_size": env.action_space.n,
    "number_hidden_units": 64,
    "optimizer_fn": optimizer_fn,
    "epsilon_decay_schedule": epsilon_decay_schedule,
    "batch_size": 64,
    "buffer_size": 100000,
    "alpha": 1e-3,
    "gamma": 0.99,
    "update_frequency": 4,
    "double_dqn": False,  # True uses Double DQN; False uses DQN
    "sordouble_dqn": False,  # True uses SORDQN; False uses DQN
    "sor_dqn": False,  # True uses SORDQN; False uses DQN
    "seed": 2024,
}
dqn_agent = DeepQAgent(**_agent_kwargs)

dqn_scores = train(dqn_agent,
                   env,
                   "dqn-checkpoint.pth",
                   number_episodes=5000,
                   target_score=200)


print("-----------------------------------------------------------")

#########################################################

