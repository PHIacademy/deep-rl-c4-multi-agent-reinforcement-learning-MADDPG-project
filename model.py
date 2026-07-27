"""
model.py

Defines the Actor and Critic neural networks used by each DDPG agent inside
the MADDPG algorithm.

- Actor:  maps a single agent's LOCAL observation -> that agent's action.
          Used for both training and (decentralized) execution.
- Critic: maps the CONCATENATED observations and actions of ALL agents
          -> a single Q-value. Only used during (centralized) training.

Both classes instantiate plain nn.Module networks; maddpg_agents.py is
responsible for keeping a "local" and a "target" copy of each and for the
soft-update logic between them.
"""

import numpy as np
import torch
import torch.nn as nn

from hyperparameters import (
    ACTOR_FC1_UNITS,
    ACTOR_FC2_UNITS,
    CRITIC_FCS1_UNITS,
    CRITIC_FC2_UNITS,
    NON_LIN,
    USE_BATCH_NORM,
)


def hidden_init(layer):
    """
    Returns the +/- bound for a uniform initialization of `layer`'s weights,
    based on the number of incoming connections (fan-in). This is the
    initialization scheme used in the original DDPG paper.
    """
    fan_in = layer.weight.data.size()[0]
    lim = 1.0 / np.sqrt(fan_in)
    return (-lim, lim)


class Actor(nn.Module):
    """Actor (Policy) network: state -> action."""

    def __init__(self, state_size, action_size, seed,
                 fc1_units=ACTOR_FC1_UNITS, fc2_units=ACTOR_FC2_UNITS):
        """
        Params
        ------
        state_size (int): dimension of a single agent's local observation
        action_size (int): dimension of a single agent's action
        seed (int): random seed
        fc1_units (int): number of units in the 1st hidden layer
        fc2_units (int): number of units in the 2nd hidden layer
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.fc1 = nn.Linear(state_size, fc1_units)
        self.bn1 = nn.BatchNorm1d(fc1_units) if USE_BATCH_NORM else nn.Identity()
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

        self.non_lin = NON_LIN
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Maps a (batch of) state(s) -> action(s) in [-1, 1] (tanh output)."""
        x = self.non_lin(self.fc1(state))
        x = self.bn1(x)
        x = self.non_lin(self.fc2(x))
        return torch.tanh(self.fc3(x))


class Critic(nn.Module):
    """
    Critic (Value) network: concatenated (states, actions) of ALL agents
    -> a single Q-value.

    Unlike vanilla DDPG (which concatenates the action only at the first
    hidden layer), here states AND actions of every agent are concatenated
    directly at the network's input, which empirically converges a bit
    faster for this environment.
    """

    def __init__(self, full_state_size, full_action_size, seed,
                 fcs1_units=CRITIC_FCS1_UNITS, fc2_units=CRITIC_FC2_UNITS):
        """
        Params
        ------
        full_state_size (int): sum of the observation sizes of ALL agents
        full_action_size (int): sum of the action sizes of ALL agents
        seed (int): random seed
        fcs1_units (int): number of units in the 1st hidden layer
        fc2_units (int): number of units in the 2nd hidden layer
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.fcs1 = nn.Linear(full_state_size + full_action_size, fcs1_units)
        self.bn1 = nn.BatchNorm1d(fcs1_units) if USE_BATCH_NORM else nn.Identity()
        self.fc2 = nn.Linear(fcs1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)

        self.non_lin = NON_LIN
        self.reset_parameters()

    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, full_states, full_actions):
        """
        Maps (all agents' states, all agents' actions) -> Q-value.

        full_states:  tensor of shape (batch, full_state_size)
        full_actions: tensor of shape (batch, full_action_size)
        """
        x = torch.cat((full_states, full_actions), dim=1)
        x = self.non_lin(self.fcs1(x))
        x = self.bn1(x)
        x = self.non_lin(self.fc2(x))
        return self.fc3(x)
