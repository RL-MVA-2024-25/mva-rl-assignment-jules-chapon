"""Training functions"""

import numpy as np
import os
import random
import time
import torch
import torch.nn as nn
import typing

from copy import deepcopy
from typing import Any, Dict, List

from env_hiv import HIVPatient


### NAMES

MODEL = "model"
DQN = "DQN"
STATE_SPACE_DIMENSION = "state_space_dimension"
ACTION_SPACE_DIMENSION = "action_space_dimension"
DEVICE = "device"
CUDA = "cuda"
CPU = "cpu"
GAMMA = "gamma"
MEMORY_CAPACITY = "memory_capacity"
EPSILON_MIN = "epsilon_min"
EPSILON_MAX = "epsilon_max"
EPSILON_DECAY_PERIOD = "epsilon_decay_period"
EPSILON_DECAY_DELAY = "epsilon_decay_delay"
EPSILON_STEP = "epsilon_step"
GRADIENT_STEPS = "gradient_steps"
UPDATE_FREQUENCY = "update_frequency"
UPDATE_TAU = "update_tau"
UPDATE_STRATEGY = "update_strategy"
EMA = "ema"
NB_LAYERS = "nb_layers"
HIDDEN_SIZE = "hidden_size"
NB_EPOCHS = "nb_epochs"
LEARNING_RATE = "learning_rate"
BATCH_SIZE = "batch_size"
CRITERION = "criterion"
SMOOTH_L1 = "smooth_l1"
OPTIMIZER = "optimizer"
ADAM = "adam"

### CONFIG

ID_BEST_MODEL = 8

EXPERIMENTS = {
    8: {
        MODEL: DQN,
        DEVICE: CPU,
        LEARNING_RATE: 0.0005,
        HIDDEN_SIZE: 256,
        NB_LAYERS: 2,
        GAMMA: 0.90,
        MEMORY_CAPACITY: 40000,
        EPSILON_MIN: 0.07,
        EPSILON_MAX: 1.0,
        EPSILON_DECAY_PERIOD: 40000,
        EPSILON_DECAY_DELAY: 500,
        BATCH_SIZE: 1000,
        GRADIENT_STEPS: 2,
        UPDATE_STRATEGY: EMA,
        UPDATE_FREQUENCY: 40,
        UPDATE_TAU: 0.005,
        CRITERION: SMOOTH_L1,
        OPTIMIZER: ADAM,
        NB_EPOCHS: 500,
    },
}


### UTILIY FUNCTIONS


def check_device(device: str) -> str:
    """
    Check if desired device is available.

    Args:
        device (str): Device to check.

    Returns:
        str: Final device.
    """
    if (device == CUDA) and (torch.cuda.is_available()):
        return CUDA
    else:
        return CPU


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
    device = check_device(device=params[DEVICE])
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
    if loss == SMOOTH_L1:
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
    if optimizer == ADAM:
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
        self.max_memory = params[MEMORY_CAPACITY]
        self.curr_memory = []
        self.position = 0
        self.device = params[DEVICE]

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
        self.network = self.create_network().to(self.params[DEVICE])
        self.target_network = deepcopy(self.network).to(self.params[DEVICE])
        self.criterion = create_criterion(loss=self.params[CRITERION])
        self.optimizer = create_optimizer(
            optimizer=self.params[OPTIMIZER],
            network=self.network,
            lr=self.params[LEARNING_RATE],
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
            nn.Linear(self.params[STATE_SPACE_DIMENSION], self.params[HIDDEN_SIZE]),
            nn.ReLU(),
        ]
        for _ in range(self.params[NB_LAYERS]):
            layers.append(nn.Linear(self.params[HIDDEN_SIZE], self.params[HIDDEN_SIZE]))
            layers.append(nn.ReLU())
        layers.append(
            nn.Linear(
                self.params[HIDDEN_SIZE],
                self.params[ACTION_SPACE_DIMENSION],
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
        if len(self.memory) > self.params[BATCH_SIZE]:
            X, A, R, Y, D = self.memory.sample(self.params[BATCH_SIZE])
            QYmax = self.target_network(Y).max(1)[0].detach()
            update = torch.addcmul(R, 1 - D, QYmax, value=self.params[GAMMA])
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
            env (HIVPatient): Environment.
        """
        epoch = 0
        epoch_cum_reward = 0
        state, _ = env.reset()
        epsilon = self.params[EPSILON_MAX]
        step = 0
        max_reward = -float("inf")
        while epoch < self.params[NB_EPOCHS]:
            start_time = time.time()
            if step > self.params[EPSILON_DECAY_DELAY]:
                epsilon = max(
                    self.params[EPSILON_MIN],
                    epsilon - self.params[EPSILON_STEP],
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
            for _ in range(self.params[GRADIENT_STEPS]):
                self.gradient_step()
            if self.params[UPDATE_STRATEGY] == EMA:
                target_state_dict = self.target_network.state_dict()
                network_state_dict = self.network.state_dict()
                tau = self.params[UPDATE_TAU]
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
    def __init__(self: _ProjectAgent) -> None:
        """
        Initialize class instance.

        Args:
            self (_ProjectAgent): Class instance.
        """
        self.id_experiment = 8
        self.params = EXPERIMENTS[self.id_experiment]
        self.params[DEVICE] = check_device(device=self.params[DEVICE])
        self.update_params(
            state_space_dimension=6,
            action_space_dimension=4,
        )
        if self.params[MODEL] == DQN:
            self.model = DQN(params=self.params)

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
        self.params[STATE_SPACE_DIMENSION] = state_space_dimension
        self.params[ACTION_SPACE_DIMENSION] = action_space_dimension
        self.params[EPSILON_STEP] = (
            self.params[EPSILON_MAX] - self.params[EPSILON_MIN]
        ) / self.params[EPSILON_DECAY_PERIOD]

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
            return 0
        self.model.best_model.eval()
        with torch.no_grad():
            state_tensor = (
                torch.FloatTensor(observation).unsqueeze(0).to(self.params[DEVICE])
            )
            Q_values = self.model.best_model(state_tensor)
        return torch.argmax(Q_values, dim=1).item()

    def save(self: _ProjectAgent) -> None:
        """
        Save the agent.

        Args:
            self (_ProjectAgent): Class instance.
        """
        folder = os.path.join("src", "saved_models")
        os.makedirs(folder, exist_ok=True)
        torch.save(
            self.model.best_model.state_dict(),
            os.path.join(folder, f"agent_{self.id_experiment}.pth"),
        )

    def load(self: _ProjectAgent) -> None:
        """
        Load a pre-trained agent.

        Args:
            self (_ProjectAgent): Class isnatnce.
        """
        folder = os.path.join("src", "saved_models")
        self.model.best_model.load_state_dict(
            torch.load(
                os.path.join(folder, f"agent_{self.id_experiment}.pth"),
                weights_only=True,
                map_location=torch.device(self.params[DEVICE]),
            )
        )


# if __name__ == "__main__":
#     from fast_env import FastHIVPatient
#     agent = ProjectAgent(id_experiment=2)
#     agent.model.train(
#         env=TimeLimit(
#             env=FastHIVPatient(domain_randomization=False), max_episode_steps=200
#         )
#     )
#     agent.save()
