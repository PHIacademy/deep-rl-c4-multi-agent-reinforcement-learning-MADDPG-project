"""
maddpg_agents.py

Implements the Multi-Agent Deep Deterministic Policy Gradient (MADDPG)
algorithm on top of the networks defined in model.py.

Contents
--------
- OUNoise:       Ornstein-Uhlenbeck process used for action exploration.
- ReplayBuffer:  shared experience replay buffer, used by all agents.
- DDPGAgent:     a single agent's local/target Actor + local/target Critic,
                 plus its own optimizers and act()/soft_update() logic.
- MADDPG:        owns all DDPGAgents, exposes act()/step()/save()/load(),
                 and implements the centralized-critic learn() update:

                     Q_targets = r + gamma * critic_target(next_full_state,
                                                            actor_target(next_state))
                     where:
                         actor_target(state)             -> action        (per agent, local obs)
                         critic_target(full_state, full_action) -> Q-value (all agents)
"""

import copy
import random
from collections import namedtuple, deque

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from model import Actor, Critic
from hyperparameters import (
    SEED,
    BUFFER_SIZE,
    BATCH_SIZE,
    LR_ACTOR,
    LR_CRITIC,
    WEIGHT_DECAY,
    CLIP_CRITIC_GRADIENT,
    GAMMA,
    TAU,
    ADD_OU_NOISE,
    MU,
    THETA,
    SIGMA,
    NOISE_START,
    NOISE_REDUCTION,
    UPDATE_EVERY_NB_STEPS,
    MULTIPLE_LEARN_PER_UPDATE,
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ==============================================================================
# Ornstein-Uhlenbeck noise process
# ==============================================================================
class OUNoise:
    """Ornstein-Uhlenbeck process for temporally-correlated exploration noise."""

    def __init__(self, size, seed=SEED, mu=MU, theta=THETA, sigma=SIGMA):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state back to the mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(len(x))
        self.state = x + dx
        return self.state


# ==============================================================================
# Replay buffer (shared across all agents)
# ==============================================================================
class ReplayBuffer:
    """
    Fixed-size buffer storing full multi-agent transitions:
    (states, actions, rewards, next_states, dones), where each field holds
    one entry per agent, e.g. states has shape (num_agents, state_size).
    """

    def __init__(self, buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE, seed=SEED):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple(
            "Experience",
            field_names=["states", "actions", "rewards", "next_states", "dones"],
        )
        self.seed = random.seed(seed)

    def add(self, states, actions, rewards, next_states, dones):
        e = self.experience(states, actions, rewards, next_states, dones)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences, returned as per-agent tensors."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(
            np.stack([e.states for e in experiences if e is not None])
        ).float().to(device)
        actions = torch.from_numpy(
            np.stack([e.actions for e in experiences if e is not None])
        ).float().to(device)
        rewards = torch.from_numpy(
            np.stack([e.rewards for e in experiences if e is not None])
        ).float().to(device)
        next_states = torch.from_numpy(
            np.stack([e.next_states for e in experiences if e is not None])
        ).float().to(device)
        dones = torch.from_numpy(
            np.stack([e.dones for e in experiences if e is not None]).astype(np.uint8)
        ).float().to(device)

        # Each of the above has shape (batch, num_agents, feature_size)
        # except rewards/dones, which have shape (batch, num_agents).
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)


# ==============================================================================
# Single DDPG agent (local + target Actor, local + target Critic)
# ==============================================================================
class DDPGAgent:
    """
    One agent's networks. The Actor only ever sees this agent's own local
    observation. The Critic sees the full (all-agent) state/action, and is
    only used during training (centralized training, decentralized execution).
    """

    def __init__(self, state_size, action_size, full_state_size, full_action_size,
                 seed=SEED):
        self.state_size = state_size
        self.action_size = action_size

        # Actor networks (local obs -> own action)
        self.actor_local = Actor(state_size, action_size, seed).to(device)
        self.actor_target = Actor(state_size, action_size, seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic networks (full state + full action -> Q-value)
        self.critic_local = Critic(full_state_size, full_action_size, seed).to(device)
        self.critic_target = Critic(full_state_size, full_action_size, seed).to(device)
        self.critic_optimizer = optim.Adam(
            self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY
        )

        # Start local and target networks with identical weights
        self.hard_update(self.actor_target, self.actor_local)
        self.hard_update(self.critic_target, self.critic_local)

        # Exploration noise
        self.noise = OUNoise(action_size, seed)
        self.noise_scale = NOISE_START

        # Most recent training diagnostics (updated every learn() call, None until then)
        self.last_critic_loss = None
        self.last_actor_loss = None
        self.last_q_expected_mean = None  # avg predicted Q over the batch; a rapidly
                                           # growing magnitude here is the classic sign
                                           # of Q-value divergence / overestimation

    def act(self, state, add_noise=True):
        """Return this agent's action for a single local observation."""
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy().squeeze(0)
        self.actor_local.train()
        if add_noise and ADD_OU_NOISE:
            action += self.noise_scale * self.noise.sample()
        return np.clip(action, -1, 1)

    def reset_noise(self):
        self.noise.reset()

    def decay_noise(self):
        self.noise_scale *= NOISE_REDUCTION

    @staticmethod
    def hard_update(target, local):
        for target_param, local_param in zip(target.parameters(), local.parameters()):
            target_param.data.copy_(local_param.data)

    @staticmethod
    def soft_update(target, local, tau=TAU):
        for target_param, local_param in zip(target.parameters(), local.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


# ==============================================================================
# MADDPG coordinator
# ==============================================================================
class MADDPG:
    """
    Owns one DDPGAgent per agent in the environment and coordinates the
    centralized-critic training update across them, while keeping actors
    fully decentralized (each actor only ever sees its own local state).
    """

    def __init__(self, num_agents, state_size, action_size, seed=SEED):
        self.num_agents = num_agents
        self.state_size = state_size
        self.action_size = action_size

        full_state_size = state_size * num_agents
        full_action_size = action_size * num_agents

        self.agents = [
            DDPGAgent(state_size, action_size, full_state_size, full_action_size, seed)
            for _ in range(num_agents)
        ]

        self.memory = ReplayBuffer(seed=seed)
        self.t_step = 0  # counts episodes, used for the learning schedule

    # -------------------------------------------------------------------- #
    def act(self, states, add_noise=True):
        """
        states: array of shape (num_agents, state_size)
        returns: array of shape (num_agents, action_size)
        """
        actions = [
            agent.act(states[i], add_noise=add_noise) for i, agent in enumerate(self.agents)
        ]
        return np.stack(actions)

    def reset_noise(self):
        for agent in self.agents:
            agent.reset_noise()

    # -------------------------------------------------------------------- #
    def step(self, states, actions, rewards, next_states, dones, episode_done=False):
        """
        Store one transition in the shared replay buffer and, according to
        the learning schedule (learn MULTIPLE_LEARN_PER_UPDATE times every
        UPDATE_EVERY_NB_STEPS environment steps), trigger the learning
        updates. Noise is decayed once per episode, independent of learning.
        """
        self.memory.add(states, actions, rewards, next_states, dones)

        # t_step now counts environment steps (not episodes), so learning
        # frequency scales with how much experience is actually collected
        # rather than with episode boundaries.
        self.t_step += 1

        if self.t_step % UPDATE_EVERY_NB_STEPS == 0 and len(self.memory) > BATCH_SIZE:
            for _ in range(MULTIPLE_LEARN_PER_UPDATE):
                for agent_index in range(self.num_agents):
                    experiences = self.memory.sample()
                    self.learn(experiences, agent_index)

        if episode_done:
            for agent in self.agents:
                agent.decay_noise()

    # -------------------------------------------------------------------- #
    def learn(self, experiences, agent_index):
        """
        Update the Critic and Actor of a single agent (agent_index) using a
        batch of full (all-agent) experiences, following the centralized
        training / decentralized execution scheme of MADDPG.
        """
        states, actions, rewards, next_states, dones = experiences
        agent = self.agents[agent_index]
        batch_size = states.shape[0]

        # ---- flatten the "all agents" dimension for the critic's input ---
        full_states = states.reshape(batch_size, -1)
        full_actions = actions.reshape(batch_size, -1)
        full_next_states = next_states.reshape(batch_size, -1)

        # ---------------------------- update critic ------------------------
        # Target actions: each agent's TARGET actor acting on its own next_state
        next_actions = torch.cat(
            [self.agents[i].actor_target(next_states[:, i, :]) for i in range(self.num_agents)],
            dim=1,
        )
        with torch.no_grad():
            Q_targets_next = agent.critic_target(full_next_states, next_actions)

        Q_targets = (
            rewards[:, agent_index].unsqueeze(1)
            + GAMMA * Q_targets_next * (1 - dones[:, agent_index].unsqueeze(1))
        )
        Q_expected = agent.critic_local(full_states, full_actions)

        critic_loss = F.mse_loss(Q_expected, Q_targets)

        agent.critic_optimizer.zero_grad()
        critic_loss.backward()
        if CLIP_CRITIC_GRADIENT:
            torch.nn.utils.clip_grad_norm_(agent.critic_local.parameters(), 1.0)
        agent.critic_optimizer.step()

        # ---------------------------- update actor --------------------------
        # For the actor loss, this agent's own action comes from its LOCAL
        # actor (with gradient); every other agent's action is taken as-is
        # from the sampled batch (detached, no gradient flows through them).
        actions_pred = [
            self.agents[i].actor_local(states[:, i, :]) if i == agent_index
            else actions[:, i, :].detach()
            for i in range(self.num_agents)
        ]
        actions_pred = torch.cat(actions_pred, dim=1)

        actor_loss = -agent.critic_local(full_states, actions_pred).mean()

        agent.actor_optimizer.zero_grad()
        actor_loss.backward()
        agent.actor_optimizer.step()

        # ---------------------------- soft-update targets --------------------
        agent.soft_update(agent.critic_target, agent.critic_local, TAU)
        agent.soft_update(agent.actor_target, agent.actor_local, TAU)

        # ---------------------------- record diagnostics ----------------------
        agent.last_critic_loss = critic_loss.item()
        agent.last_actor_loss = actor_loss.item()
        agent.last_q_expected_mean = Q_expected.mean().item()

    # -------------------------------------------------------------------- #
    def get_diagnostics(self):
        """
        Returns the most recent (critic_loss, actor_loss, mean_Q) for each
        agent, averaged across agents. Any value is None until the first
        learn() call has happened for every agent.
        """
        critic_losses = [a.last_critic_loss for a in self.agents]
        actor_losses = [a.last_actor_loss for a in self.agents]
        q_means = [a.last_q_expected_mean for a in self.agents]

        if any(v is None for v in critic_losses):
            return None, None, None

        return (
            float(np.mean(critic_losses)),
            float(np.mean(actor_losses)),
            float(np.mean(q_means)),
        )

    # -------------------------------------------------------------------- #
    def warmup(self, env, brain_name, n_episodes=None, max_t=None, seed=SEED):
        """
        Populate the shared replay buffer with `n_episodes` episodes of pure
        uniform-random actions -- no learning happens during this phase.

        This exists purely to avoid a cold-start failure mode: if learning
        starts as soon as the buffer holds BATCH_SIZE transitions, the
        critic can lock onto a trivial "predict ~0 Q everywhere" solution
        before the buffer has any diverse (especially positive-reward)
        experience, which then starves the actor of a useful gradient.
        Random play first gives the buffer a wider, less degenerate mix.

        Returns
        -------
        dict with 'episodes', 'buffer_size', and 'positive_reward_episodes'
        (how many warmup episodes scored > 0), useful as a quick sanity check
        that the buffer isn't just full of identical zero-reward transitions.
        """
        from hyperparameters import NB_EPISODES, NB_STEPS  # local import avoids a hard
                                                             # dependency at module load time
        n_episodes = NB_EPISODES if n_episodes is None else n_episodes
        max_t = NB_STEPS if max_t is None else max_t
        rng = np.random.RandomState(seed)

        positive_reward_episodes = 0

        for _ in range(n_episodes):
            env_info = env.reset(train_mode=True)[brain_name]
            states = env_info.vector_observations
            episode_scores = np.zeros(self.num_agents)

            for _t in range(max_t):
                actions = rng.uniform(-1, 1, size=(self.num_agents, self.action_size))
                env_info = env.step(actions)[brain_name]
                next_states = env_info.vector_observations
                rewards = np.array(env_info.rewards)
                dones = np.array(env_info.local_done)

                self.memory.add(states, actions, rewards, next_states, dones)

                states = next_states
                episode_scores += rewards

                if np.any(dones):
                    break

            if np.max(episode_scores) > 0:
                positive_reward_episodes += 1

        return {
            "episodes": n_episodes,
            "buffer_size": len(self.memory),
            "positive_reward_episodes": positive_reward_episodes,
        }

    # -------------------------------------------------------------------- #
    def save(self, path_prefix="checkpoint"):
        """Save each agent's local Actor and Critic weights to disk."""
        for i, agent in enumerate(self.agents):
            torch.save(agent.actor_local.state_dict(), f"{path_prefix}_actor_agent{i}.pth")
            torch.save(agent.critic_local.state_dict(), f"{path_prefix}_critic_agent{i}.pth")

    def load(self, path_prefix="checkpoint"):
        """Load each agent's local (and target, hard-synced) weights from disk."""
        for i, agent in enumerate(self.agents):
            agent.actor_local.load_state_dict(
                torch.load(f"{path_prefix}_actor_agent{i}.pth", map_location=device)
            )
            agent.critic_local.load_state_dict(
                torch.load(f"{path_prefix}_critic_agent{i}.pth", map_location=device)
            )
            agent.hard_update(agent.actor_target, agent.actor_local)
            agent.hard_update(agent.critic_target, agent.critic_local)
