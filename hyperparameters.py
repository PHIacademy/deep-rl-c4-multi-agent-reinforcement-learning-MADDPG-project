"""
hyperparameters.py

Central place for every tunable constant used by model.py and maddpg_agents.py.
Keeping these in one file makes it easy to tune the agent without touching any
class/logic code -- just edit a value here and restart the notebook kernel.

Values below are the ones empirically found to work well for the Unity Tennis
(2-agent, 24-dim local obs, 2-dim continuous action) environment, based on
architecture/parameter choices commonly reported for this project (e.g.
concatenating states+actions at the critic's input, learning a small number
of times every few environment steps, etc).
"""

import torch.nn.functional as F

# ----------------------------------------------------------------------------
# General / reproducibility
# ----------------------------------------------------------------------------
SEED = 10                          # Random seed (numpy, torch, env)

# ----------------------------------------------------------------------------
# Training loop
# ----------------------------------------------------------------------------
NB_EPISODES = 10000                 # Max number of training episodes
NB_STEPS = 1000                     # Max number of steps per episode
WARMUP_EPISODES = 300               # Episodes of pure random play used to pre-fill the replay
                                     # buffer with diverse experience *before* any learning
                                     # happens (prevents the critic from locking onto a
                                     # trivial "predict ~0" solution from too little/too
                                     # uniform early data)
SOLVED_SCORE = 0.5                  # Target average score (over 100 episodes) to solve the env
SCORES_WINDOW = 100                 # Window size (episodes) for the rolling average

# ----------------------------------------------------------------------------
# Learning schedule
# ----------------------------------------------------------------------------
# NOTE: previously this was episode-based (learn 3x every 4 episodes). Tennis
# episodes are very short early on (~10-20 steps before the ball drops), so
# that schedule gave far too few gradient updates per unit of collected
# experience -- likely the main reason training plateaued. Learning every
# few TIMESTEPS (as basically every published Tennis solution does) gives a
# much higher and steadier ratio of learning updates to experience collected.
UPDATE_EVERY_NB_STEPS = 2           # Number of environment steps between learning updates
MULTIPLE_LEARN_PER_UPDATE = 1       # Number of consecutive learning steps performed each update

# ----------------------------------------------------------------------------
# Replay buffer
# ----------------------------------------------------------------------------
BUFFER_SIZE = int(1e5)              # Replay buffer size
BATCH_SIZE = 200                    # Minibatch size sampled per learning step

# ----------------------------------------------------------------------------
# Network architecture
# ----------------------------------------------------------------------------
ACTOR_FC1_UNITS = 400               # Units in Actor hidden layer 1
ACTOR_FC2_UNITS = 300               # Units in Actor hidden layer 2
CRITIC_FCS1_UNITS = 400             # Units in Critic hidden layer 1
CRITIC_FC2_UNITS = 300              # Units in Critic hidden layer 2
NON_LIN = F.relu                    # Non-linearity used throughout the networks
USE_BATCH_NORM = True               # Apply BatchNorm after the activation of the first hidden layer

# ----------------------------------------------------------------------------
# Optimization
# ----------------------------------------------------------------------------
LR_ACTOR = 1e-4                     # Learning rate for the Actor networks
LR_CRITIC = 1e-4                    # Learning rate for the Critic networks. Lowered from 1e-3:
                                     # combined with per-step learning (many more updates overall)
                                     # a critic LR this high is a common cause of a rise-then-collapse
                                     # pattern from Q-value divergence. The reference solution that
                                     # solved cleanly in ~1166 episodes used 1e-4 for both networks.
WEIGHT_DECAY = 0                    # L2 weight decay (Critic optimizer)
CLIP_CRITIC_GRADIENT = True         # Clip critic gradients (norm 1.0) during optimization --
                                     # a standard DDPG stabilizer against occasional critic-loss
                                     # spikes poisoning the actor update

# ----------------------------------------------------------------------------
# RL parameters
# ----------------------------------------------------------------------------
GAMMA = 0.995                       # Discount factor
TAU = 1e-3                          # Soft update rate for target networks

# ----------------------------------------------------------------------------
# Exploration noise (Ornstein-Uhlenbeck process)
# ----------------------------------------------------------------------------
ADD_OU_NOISE = True                 # Whether to add OU noise to actions during training
MU = 0.0                            # OU noise mean
THETA = 0.15                        # OU noise mean-reversion rate
SIGMA = 0.2                         # OU noise volatility
NOISE_START = 1.0                   # Initial noise amplitude (scales the OU sample)
NOISE_REDUCTION = 0.999             # Multiplicative decay applied to the noise amplitude each
                                     # episode. Previously 1.0 (no decay) -- with constant
                                     # full-strength noise the policy can rally the ball back
                                     # and forth but struggles to lock in the last bit of
                                     # precision needed to consistently clear +0.5 average.
                                     # 0.999/episode roughly halves noise by ~episode 700,
                                     # while still exploring heavily during early training.
