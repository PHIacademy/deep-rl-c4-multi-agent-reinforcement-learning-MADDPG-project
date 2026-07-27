# Project: Collaboration and Competition — Deep Reinforcement Learning Nanodegree

### Introduction

For this project, two agents are trained to control rackets and bounce a ball over a net, using **Multi-Agent Deep Deterministic Policy Gradient (MADDPG)** — a centralized-training, decentralized-execution extension of DDPG.

| Random Agents | Trained Agents |
| --- | --- |
| [![Random Agents](./Training-Results/random_agent.gif)](./Training-Results/random_agent.gif) | [![Trained Agents](./Training-Results/trained_agent.gif)](./Training-Results/trained_agent.gif) |

If an agent hits the ball over the net, it receives a reward of `+0.1`. If an agent lets the ball hit the ground, or hits the ball out of bounds, it receives a reward of `-0.01`. Thus, the goal of each agent is to keep the ball in play.

**State space**: 8 variables per agent, corresponding to the position and velocity of the ball and racket (each agent receives its own local observation).

**Action space**: 2 continuous actions per agent, corresponding to movement toward/away from the net, and jumping.

**Solve criteria**: The task is episodic. After each episode, the rewards each agent received (undiscounted) are summed to get a score per agent, and the **maximum** of the two scores is taken as that episode's score. The environment is considered solved when the average of these scores over **100 consecutive episodes** is at least **+0.5**.

---

### Getting Started

#### 1. Install Python 3.6 in a virtual environment

The Unity ML-Agents API used in this project (`unityagents`, ml-agents 0.4) requires Python 3.6.

```
winget install -e --id Python.Python.3.6 --accept-source-agreements --accept-package-agreements --override "InstallAllUsers=0 PrependPath=0 Include_doc=0 Include_test=0 Include_tcltk=0 Include_tools=0 Include_launcher=0 AssociateFiles=0 Shortcuts=0 CompileAll=0 Include_pip=1 SimpleInstall=1"
```

Create and activate the project virtual environment:

```
cd /d/python/Tennis_project
py -3.6 -m venv .venv
source .venv/Scripts/activate
python --version   # should print Python 3.6.8
```

Pin installer tooling for 3.6 compatibility:

```
python -m pip install --upgrade "pip<22" "setuptools<60" wheel
```

#### 2. Get the project code and the `unityagents` package

Clone the Udacity `Value-based-methods` repository and install the `unityagents` package it ships with (this is what provides `from unityagents import UnityEnvironment`):

```
cd /d/python/Tennis_project
git clone https://github.com/udacity/Value-based-methods.git
cd Value-based-methods/python
pip install .
```

#### 3. Install the remaining dependencies

```
pip install torch matplotlib "protobuf==3.20.3" "ipykernel<6"
```

Quick validation:

```
python -c "import torch, matplotlib, unityagents; print(torch.__version__, matplotlib.__version__)"
```

#### 4. Download the Unity environment

Download the pre-built Tennis environment matching your OS (you do **not** need to install Unity itself):

- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

Unzip it into the project folder (or any folder of your choice), then point the `file_name` argument in [`Tennis_workspace_version_solved.ipynb`](./Tennis_workspace_version_solved.ipynb) at the extracted `.exe` / `.app` / `.x86_64`. For example, on Windows (64-bit), if you unzipped the environment into the project folder alongside the notebook:

```
env = UnityEnvironment(file_name="Tennis_Windows_x86_64/Tennis.exe")
```

On Mac:

```
env = UnityEnvironment(file_name="Tennis.app")
```

Adjust the path to match wherever you actually placed the downloaded environment on your machine.

#### 5. Register the Jupyter kernel

```
python -m ipykernel install --user --name drlnd --display-name "drlnd"
```

> In VS Code, the kernel may show as `.venv (3.6.8)` instead of `drlnd` — both point to the same environment and are fine to use.

---

### File Structure

| File | Description |
| --- | --- |
| [`Tennis_workspace_version_solved.ipynb`](./Tennis_workspace_version_solved.ipynb) | Main notebook — launches the environment, trains the MADDPG agents, plots scores, and evaluates the trained agents |
| [`maddpg_agents.py`](./maddpg_agents.py) | `MADDPG` coordinator, `DDPGAgent` (per-agent local/target Actor + Critic), `OUNoise`, and the shared `ReplayBuffer` |
| [`model.py`](./model.py) | `Actor` and `Critic` network definitions used by each agent |
| [`hyperparameters.py`](./hyperparameters.py) | All tunable constants (network sizes, learning rates, discount factor, noise, learning schedule, etc.) |
| [`checkpoint_actor_agent0.pth`](./checkpoint_actor_agent0.pth) | Saved local Actor weights for agent 0 |
| [`checkpoint_actor_agent1.pth`](./checkpoint_actor_agent1.pth) | Saved local Actor weights for agent 1 |
| [`checkpoint_critic_agent0.pth`](./checkpoint_critic_agent0.pth) | Saved local Critic weights for agent 0 |
| [`checkpoint_critic_agent1.pth`](./checkpoint_critic_agent1.pth) | Saved local Critic weights for agent 1 |
| [`Report.md`](./Report.md) | Learning algorithm, hyperparameters, network architecture, rewards plot, and ideas for future work |

---

### Instructions

Open [`Tennis_workspace_version_solved.ipynb`](./Tennis_workspace_version_solved.ipynb) and run the cells **in order**. The notebook is split into two flows depending on what you want to do:

#### Flow A — Train the agents from scratch

Run:

1. **Section 1** — Start the Environment (launches Unity)
2. The "get the default brain" cell
3. **Section 2** — Examine the State and Action Spaces
4. **Section 5** — creates the `MADDPG` coordinator and runs the training loop (`maddpg_train`)

Make sure [`hyperparameters.py`](./hyperparameters.py), [`model.py`](./model.py), and [`maddpg_agents.py`](./maddpg_agents.py) are saved in the **same folder** as the notebook before running these cells.

Training runs until either the environment is solved (average score ≥ +0.5 over 100 consecutive episodes) or `NB_EPISODES` is reached. On success, the trained weights are automatically saved to the four `checkpoint_*.pth` files, and a plot of score-per-episode is displayed.

> Section 3 (random-action demo) is optional — it's only there to sanity-check that the environment is wired up correctly before training. You can skip straight from Section 2 to Section 5.

#### Flow B — Watch a previously-trained agent (no training)

If you already have the four `checkpoint_*.pth` files from a prior training run (or want to watch the agents right after training, in the same notebook run):

1. Run the imports cell, then **Section 1**.
2. Run the "get the default brain" cell and **Section 2** (needed to reconstruct `state_size`/`action_size`).
3. **Section 7 — Evaluate the Trained Agent** — loads all four checkpoints into a fresh `MADDPG` coordinator via `maddpg.load(path_prefix='checkpoint')` and runs several episodes with noise-free (greedy) actions, so you can watch the trained agents rally.

```python
maddpg.load(path_prefix='checkpoint')
```

4. `env.close()` shuts down the Unity window when you're done.

---

### Results

The agents solved the environment (average score ≥ +0.5 over 100 consecutive episodes, after taking the max over both agents) in **2502 episodes**, reaching a final rolling average of **0.511**. See [`score_plot.png`](./Training-Results/score_plot.png) and Section 6 of [`Tennis_workspace_version_solved.ipynb`](./Tennis_workspace_version_solved.ipynb) for the full training curve.

Running the trained agents greedily (no exploration noise) over 5 evaluation episodes (Section 7) achieved an **average score of 1.74**, well above the +0.5 solve threshold.

---

### Notes

- See [`Report.md`](./Report.md) for a full description of the learning algorithm, hyperparameters, network architecture, rewards plot, and ideas for future work.
- If Unity fails to launch with `Error: Global Illumination requires a graphics device to render albedo.`, this is expected behavior of `no_graphics=True` and can be ignored; it only affects rendering, not training.
