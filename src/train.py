"""Training functions"""

import numpy as np
import os
import pickle as pkl
import random
import time
import torch
import torch.nn as nn
import typing

from copy import deepcopy
from gymnasium.wrappers import TimeLimit
from typing import Any, Dict, List

from env_hiv import HIVPatient
import names as names
import config as config

### UTILIY FUNCTIONS


def check_device(device: str) -> str:
    """
    Check if desired device is available.

    Args:
        device (str): Device to check.

    Returns:
        str: Final device.
    """
    if (device == names.CUDA) and (torch.cuda.is_available()):
        return names.CUDA
    else:
        return names.CPU


def greedy_action(
    params: Dict[str, Any], network: nn.Module, state: List[float]
) -> int:
    """
    Choose the greedy action.

    Args:
        params (Dict[str, Any]): Parameters of the agent.
        network (nn.Module): Neural network.
        state (List[float]): State at previous time.

    Returns:
        int: Greedy action to take.
    """
    device = check_device(device=params[names.DEVICE])
    with torch.no_grad():
        Q = network(torch.Tensor(state).unsqueeze(0).to(device))
        return torch.argmax(Q).item()


def create_criterion(loss: str):
    """
    Create loss function.

    Args:
        loss (str): Name of the loss function.

    Raises:
        ValueError: If name of loss function is unknown.

    Returns:
        Loss function.
    """
    if loss == names.SMOOTH_L1:
        return nn.SmoothL1Loss()
    else:
        raise ValueError("This loss function is not supported")


def create_optimizer(optimizer: str, network: nn.Module, lr: float):
    """
    Create optimizer.

    Args:
        optimizer (str): Optimizer name.
        network (nn.Module): Neural network.
        lr (float): Learning rate.

    Raises:
        ValueError: If the optimizer name is unknown.

    Returns:
        Optimizer.
    """
    if optimizer == names.ADAM:
        return torch.optim.Adam(network.parameters(), lr=lr)
    else:
        raise ValueError("This optimizer is not supported")


_Memory = typing.TypeVar(name="_Memory", bound="Memory")


class Memory:
    def __init__(self: _Memory, params: Dict[str, Any]) -> None:
        """
        Initialize class instance.

        Args:
            self (_Memory): Class instance.
            params (Dict[str, Any]): Parameters of the agent.
        """
        self.max_memory = params[names.MEMORY_CAPACITY]
        self.curr_memory = []
        self.position = 0
        self.device = params[names.DEVICE]

    def append(
        self: _Memory,
        state: List[float],
        action: int,
        reward: float,
        next_state: int,
        done: bool,
    ) -> None:
        """
        Add last state, action, reward... to memory.

        Args:
            self (_Memory): Class instance.
            state (List[float]): Previous state.
            action (int): Action.
            reward (float): Reward.
            next_state (int): Next state.
            done (bool): Whether the simulation is over or not.
        """
        if len(self.curr_memory) < self.max_memory:
            self.curr_memory.append(None)
        self.curr_memory[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.max_memory

    def sample(self: _Memory, batch_size: int) -> List:
        """
        Sample elements from the memory.

        Args:
            self (_Memory): Class instance.
            batch_size (int): Number of elements to sample.

        Returns:
            List: List of samples.
        """
        batch = random.sample(self.curr_memory, batch_size)
        return list(
            map(lambda x: torch.Tensor(np.array(x)).to(self.device), list(zip(*batch)))
        )

    def __len__(self: _Memory) -> int:
        return len(self.curr_memory)


_DQN = typing.TypeVar(name="_DQN", bound="DQN")


class DQN:
    def __init__(self: _DQN, params: Dict[str, Any]) -> None:
        """
        Initialize class instance.

        Args:
            self (_DQN): Class instance.
            params (Dict[str, Any]): Parameters of teh agent.
        """
        self.params = params
        self.memory = Memory(params=self.params)
        self.network = self.create_network().to(self.params[names.DEVICE])
        self.target_network = deepcopy(self.network).to(self.params[names.DEVICE])
        self.criterion = create_criterion(loss=self.params[names.CRITERION])
        self.optimizer = create_optimizer(
            optimizer=self.params[names.OPTIMIZER],
            network=self.network,
            lr=self.params[names.LEARNING_RATE],
        )
        self.best_model = self.network
        self.best_reward = -float("inf")
        self.epoch_rewards = []

    def create_network(self: _DQN) -> nn.Module:
        """
        Create the network of the DQN.

        Args:
            self (_DQN): CLass instance.

        Returns:
            nn.Module: Neural network.
        """
        layers = [
            nn.Linear(
                self.params[names.STATE_SPACE_DIMENSION], self.params[names.HIDDEN_SIZE]
            ),
            nn.ReLU(),
        ]
        for _ in range(self.params[names.NB_LAYERS]):
            layers.append(
                nn.Linear(
                    self.params[names.HIDDEN_SIZE], self.params[names.HIDDEN_SIZE]
                )
            )
            layers.append(nn.ReLU())
        layers.append(
            nn.Linear(
                self.params[names.HIDDEN_SIZE],
                self.params[names.ACTION_SPACE_DIMENSION],
            )
        )
        network = nn.Sequential(*layers)
        return network

    def gradient_step(self: _DQN) -> None:
        """
        Make a gradient step.

        Args:
            self (_DQN): Class instance.
        """
        if len(self.memory) > self.params[names.BATCH_SIZE]:
            X, A, R, Y, D = self.memory.sample(self.params[names.BATCH_SIZE])
            QYmax = self.target_network(Y).max(1)[0].detach()
            update = torch.addcmul(R, 1 - D, QYmax, value=self.params[names.GAMMA])
            QXA = self.network(X).gather(1, A.to(torch.long).unsqueeze(1))
            loss = self.criterion(QXA, update.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def train(self: _DQN, env: HIVPatient):
        """
        Train the model.

        Args:
            self (_DQN): Class instance.
            env (FastHIVPatient): Environment.
        """
        epoch = 0
        epoch_cum_reward = 0
        state, _ = env.reset()
        epsilon = self.params[names.EPSILON_MAX]
        step = 0
        max_reward = -float("inf")
        while epoch < self.params[names.NB_EPOCHS]:
            start_time = time.time()
            if step > self.params[names.EPSILON_DECAY_DELAY]:
                epsilon = max(
                    self.params[names.EPSILON_MIN],
                    epsilon - self.params[names.EPSILON_STEP],
                )
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = greedy_action(
                    params=self.params, network=self.network, state=state
                )
            next_state, reward, done, trunc, _ = env.step(action)
            self.memory.append(state, action, reward, next_state, done)
            epoch_cum_reward += reward
            for _ in range(self.params[names.GRADIENT_STEPS]):
                self.gradient_step()
            if self.params[names.UPDATE_STRATEGY] == names.EMA:
                target_state_dict = self.target_network.state_dict()
                network_state_dict = self.network.state_dict()
                tau = self.params[names.UPDATE_TAU]
                for key in network_state_dict:
                    target_state_dict[key] = (
                        tau * network_state_dict[key]
                        + (1 - tau) * target_state_dict[key]
                    )
                self.target_network.load_state_dict(target_state_dict)
            step += 1
            if done or trunc:
                end_time = time.time()
                print(f"Epoch {epoch+1} --------------------------------------")
                print(
                    f"Reward : {epoch_cum_reward/1e10:.2f}.1e10 --- Time : {end_time-start_time:.2f} seconds"
                )
                self.epoch_rewards.append(epoch_cum_reward)
                if epoch_cum_reward > max_reward:
                    self.best_model = self.network
                    self.best_reward = float(epoch_cum_reward)
                    max_reward = epoch_cum_reward
                epoch += 1
                state, _ = env.reset()
                epoch_cum_reward = 0
            else:
                state = next_state
        print("Training done.")
        print(f"Best reward : {self.best_reward/1e10:.2f}.1e10")


_ProjectAgent = typing.TypeVar(name="_ProjectAgent", bound="ProjectAgent")


class ProjectAgent:
    def __init__(
        self: _ProjectAgent, id_experiment: int = config.ID_BEST_MODEL
    ) -> None:
        """
        Initialize class instance.

        Args:
            self (_ProjectAgent): Class instance.
            id_experiment (int, optional): ID of the experiment. Defaults to config.ID_BEST_MODEL.
        """
        self.id_experiment = id_experiment
        self.env = TimeLimit(
            env=HIVPatient(domain_randomization=False), max_episode_steps=200
        )
        self.params = config.EXPERIMENTS[id_experiment]
        self.params[names.DEVICE] = check_device(device=self.params[names.DEVICE])
        self.update_params(
            state_space_dimension=6,
            action_space_dimension=4,
        )
        if self.params[names.MODEL] == names.DQN:
            self.model = DQN(params=self.params)
        self.best_model = None
        self.folder = os.path.join("src", "saved_models")
        os.makedirs(self.folder, exist_ok=True)
        print("Agent created.")

    def update_params(
        self: _ProjectAgent,
        state_space_dimension: int,
        action_space_dimension: int,
    ) -> None:
        """
        Update the parameters of the agent given the environment.

        Args:
            self (_ProjectAgent): Class instance.
            state_space_dimension (int): Dimension of the state space.
            action_space_dimension (int): Dimension of the action space.
        """
        self.params[names.STATE_SPACE_DIMENSION] = state_space_dimension
        self.params[names.ACTION_SPACE_DIMENSION] = action_space_dimension
        self.params[names.EPSILON_STEP] = (
            self.params[names.EPSILON_MAX] - self.params[names.EPSILON_MIN]
        ) / self.params[names.EPSILON_DECAY_PERIOD]

    def act(
        self: _ProjectAgent, observation: List[float], use_random: bool = False
    ) -> int:
        """
        Decide whith action to take given an observation.

        Args:
            self (_ProjectAgent): Class instance.
            observation (List[float]): Observation.
            use_random (bool, optional): Whether to do a random aciton or not. Defaults to False.

        Returns:
            int: Action to take.
        """
        if use_random:
            return self.env.action_space.sample()
        state_tensor = (
            torch.FloatTensor(observation).unsqueeze(0).to(self.params[names.DEVICE])
        )
        Q_values = self.best_model(state_tensor)
        return torch.argmax(Q_values, dim=1).item()

    def save(self: _ProjectAgent) -> None:
        """
        Save the agent.

        Args:
            self (_ProjectAgent): Class instance.
        """
        self.best_model = self.model.best_model
        self.model = None
        with open(
            os.path.join(self.folder, f"agent_{self.id_experiment}.pkl"), "wb"
        ) as file:
            pkl.dump(self, file)
        print("Agent saved.")

    def load(self: _ProjectAgent) -> _ProjectAgent:
        """
        Load a pre-trained agent.

        Args:
            self (_ProjectAgent): Class isnatnce.

        Returns:
            _ProjectAgent: Pre-trained agent.
        """
        with open(
            os.path.join(self.folder, f"agent_{self.id_experiment}.pkl"), "rb"
        ) as file:
            agent = pkl.load(file)
            self.best_model = agent.best_model
        print("Agent loaded.")


if __name__ == "__main__":
    agent = ProjectAgent(id_experiment=8)
    agent.model.train(env=agent.env)
    agent.save()
