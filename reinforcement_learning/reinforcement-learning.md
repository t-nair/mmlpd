# Reinforcement Learning Hub
## Full Project Database
* [Jump to tables by topic](#tables-by-topic)

| Topic                                     | Project Idea                                                               | Difficulty    |
|-------------------------------------------|----------------------------------------------------------------------------|---------------|
| Deep Q-Networks (DQN)                     | Train a DQN to play a simple Atari game (e.g. Pong).                      | Beginner      |
| Deep Q-Networks (DQN)                     | Use DQN to learn cart-pole balancing in OpenAI Gym.                       | Beginner      |
| Deep Q-Networks (DQN)                     | Implement Double DQN to reduce value bias.                               | Intermediate  |
| Deep Q-Networks (DQN)                     | Add experience replay to improve DQN training.                            | Intermediate  |
| Deep Q-Networks (DQN)                     | Use DQN to solve MountainCar-v0 environment.                              | Intermediate  |
| Deep Q-Networks (DQN)                     | Extend DQN to Dueling DQN architecture for better stability.              | Advanced      |
| Deep Q-Networks (DQN)                     | Apply DQN to a custom game environment.                                  | Advanced      |
| Deep Q-Networks (DQN)                     | Research prioritized experience replay for faster learning.                | Advanced      |
| Deep Q-Networks (DQN)                     | Integrate convolutional encoders in DQN for image input.                   | Advanced      |
| Deep Q-Networks (DQN)                     | Combine DQN with imitation learning to pretrain policy.                    | Advanced      |
| Proximal Policy Optimization (PPO)        | Apply PPO to train an agent in CartPole.                                 | Beginner      |
| Proximal Policy Optimization (PPO)        | Use stable-baselines to run PPO on a simple environment.                  | Beginner      |
| Proximal Policy Optimization (PPO)        | Tune PPO hyperparameters for faster convergence.                          | Intermediate  |
| Proximal Policy Optimization (PPO)        | Use PPO for continuous control (e.g. LunarLanderContinuous).             | Intermediate  |
| Proximal Policy Optimization (PPO)        | Implement PPO with Generalized Advantage Estimation.                      | Intermediate  |
| Proximal Policy Optimization (PPO)        | Develop a custom environment (e.g. robotic control) for PPO.              | Advanced      |
| Proximal Policy Optimization (PPO)        | Use distributed PPO across multiple workers for speed.                    | Advanced      |
| Proximal Policy Optimization (PPO)        | Research clipping parameters’ effect on PPO stability.                    | Advanced      |
| Proximal Policy Optimization (PPO)        | Combine PPO with curiosity-driven exploration.                           | Advanced      |
| Proximal Policy Optimization (PPO)        | Implement meta-learning (MAML) on top of PPO policy.                     | Advanced      |
| Actor-Critic (A3C/A2C)                    | Train an A2C agent on the CartPole environment.                          | Beginner      |
| Actor-Critic (A3C/A2C)                    | Use an asynchronous runner to train A3C on Pong.                         | Beginner      |
| Actor-Critic (A3C/A2C)                    | Implement entropy regularization for exploration.                         | Intermediate  |
| Actor-Critic (A3C/A2C)                    | Use a recurrent policy (LSTM) in actor-critic agent.                    | Intermediate  |
| Actor-Critic (A3C/A2C)                    | Compare A2C vs DQN on a grid-world task.                               | Intermediate  |
| Actor-Critic (A3C/A2C)                    | Develop a multi-threaded A3C for faster training.                       | Advanced      |
| Actor-Critic (A3C/A2C)                    | Research IMPALA (Importance-weighted Actor-Learner) architecture.        | Advanced      |
| Actor-Critic (A3C/A2C)                    | Apply actor-critic to train an agent in MuJoCo.                          | Advanced      |
| Actor-Critic (A3C/A2C)                    | Integrate self-play in actor-critic for games like Go.                  | Advanced      |
| Actor-Critic (A3C/A2C)                    | Combine PPO and A3C ideas for a hybrid agent.                           | Advanced      |
| Soft Actor-Critic (SAC)                   | Train a SAC agent on Pendulum-v0.                                        | Beginner      |
| Soft Actor-Critic (SAC)                   | Use SAC to solve a continuous control (e.g. Walker2d).                   | Beginner      |
| Soft Actor-Critic (SAC)                   | Tune SAC’s entropy temperature parameter.                               | Intermediate  |
| Soft Actor-Critic (SAC)                   | Compare SAC vs DDPG on a benchmark task.                               | Intermediate  |
| Soft Actor-Critic (SAC)                   | Use twin Q-networks as in SAC for stability.                           | Intermediate  |
| Soft Actor-Critic (SAC)                   | Implement SAC for a custom robotics control problem.                     | Advanced      |
| Soft Actor-Critic (SAC)                   | Research automatic entropy tuning in SAC.                                | Advanced      |
| Soft Actor-Critic (SAC)                   | Combine SAC with HER (Hindsight Experience Replay).                     | Advanced      |
| Soft Actor-Critic (SAC)                   | Use SAC for multi-agent cooperative scenarios.                           | Advanced      |
| Soft Actor-Critic (SAC)                   | Explore batch-constrained SAC for offline RL.                            | Advanced      |
| Policy Gradients (REINFORCE, A2C)         | Implement REINFORCE to solve CartPole.                                 | Beginner      |
| Policy Gradients (REINFORCE, A2C)         | Use Monte Carlo returns for a simple policy gradient.                    | Beginner      |
| Policy Gradients (REINFORCE, A2C)         | Add a baseline to REINFORCE to reduce variance.                        | Intermediate  |
| Policy Gradients (REINFORCE, A2C)         | Apply A2C (advantage actor-critic) on a toy game.                      | Intermediate  |
| Policy Gradients (REINFORCE, A2C)         | Use reward shaping to speed up learning.                               | Intermediate  |
| Policy Gradients (REINFORCE, A2C)         | Research variance-reduction techniques for policy gradients.            | Advanced      |
| Policy Gradients (REINFORCE, A2C)         | Implement trust region policy optimization (TRPO) for comparison.        | Advanced      |
| Policy Gradients (REINFORCE, A2C)         | Use policy gradients with curriculum learning tasks.                    | Advanced      |
| Policy Gradients (REINFORCE, A2C)         | Develop a hierarchical policy gradient agent (options).                | Advanced      |
| Policy Gradients (REINFORCE, A2C)         | Integrate meta-learning to adapt learning rates in PG.                 | Advanced      |
| Multi-Agent Reinforcement Learning        | Train two agents in a simple cooperative game (e.g. Pursuit).           | Beginner      |
| Multi-Agent Reinforcement Learning        | Implement self-play between two simple agents.                           | Beginner      |
| Multi-Agent Reinforcement Learning        | Use MADDPG (Multi-Agent DDPG) for cooperative tasks.                    | Intermediate  |
| Multi-Agent Reinforcement Learning        | Train agents with a shared reward signal (cooperative).                | Intermediate  |
| Multi-Agent Reinforcement Learning        | Implement competitive self-play with a reward zero-sum game.            | Intermediate  |
| Multi-Agent Reinforcement Learning        | Develop communication protocols among agents.                            | Advanced      |
| Multi-Agent Reinforcement Learning        | Research CTDE (Centralized Training Decentralized Execution).            | Advanced      |
| Multi-Agent Reinforcement Learning        | Use Independent Q-Learning in a multi-agent grid world.                 | Advanced      |
| Multi-Agent Reinforcement Learning        | Explore federated reinforcement learning across agents.                  | Advanced      |
| Multi-Agent Reinforcement Learning        | Investigate game-theoretic equilibrium in MARL.                          | Advanced      |
| Hierarchical Reinforcement Learning       | Implement subgoal division in a simple maze.                             | Beginner      |
| Hierarchical Reinforcement Learning       | Use option-critic framework on a two-room navigation.                   | Beginner      |
| Hierarchical Reinforcement Learning       | Create high-level and low-level policies for a task (e.g. Fetch).      | Intermediate  |
| Hierarchical Reinforcement Learning       | Use subroutines (options) to break down tasks in Atari.                | Intermediate  |
| Hierarchical Reinforcement Learning       | Apply feudal networks (manager-worker) on a game.                       | Intermediate  |
| Hierarchical Reinforcement Learning       | Train HRL on tasks requiring long-term planning (Montezuma’s Revenge).   | Advanced      |
| Hierarchical Reinforcement Learning       | Research skill discovery methods for HRL.                                | Advanced      |
| Hierarchical Reinforcement Learning       | Combine HRL with intrinsic motivation (curiosity).                      | Advanced      |
| Hierarchical Reinforcement Learning       | Implement MAXQ decomposition for a robot task.                           | Advanced      |
| Hierarchical Reinforcement Learning       | Use hierarchical policies in multi-agent coordination.                  | Advanced      |

## Tables by Topic
### Deep Q-Networks (DQN)
| Project Idea                                                               | Difficulty    |
|----------------------------------------------------------------------------|---------------|
| Train a DQN to play a simple Atari game (e.g. Pong).                      | Beginner      |
| Use DQN to learn cart-pole balancing in OpenAI Gym.                       | Beginner      |
| Implement Double DQN to reduce value bias.                               | Intermediate  |
| Add experience replay to improve DQN training.                            | Intermediate  |
| Use DQN to solve MountainCar-v0 environment.                              | Intermediate  |
| Extend DQN to Dueling DQN architecture for better stability.              | Advanced      |
| Apply DQN to a custom game environment.                                  | Advanced      |
| Research prioritized experience replay for faster learning.                | Advanced      |
| Integrate convolutional encoders in DQN for image input.                   | Advanced      |
| Combine DQN with imitation learning to pretrain policy.                    | Advanced      |

### Proximal Policy Optimization (PPO)
| Project Idea                                                               | Difficulty    |
|----------------------------------------------------------------------------|---------------|
| Apply PPO to train an agent in CartPole.                                 | Beginner      |
| Use stable-baselines to run PPO on a simple environment.                  | Beginner      |
| Tune PPO hyperparameters for faster convergence.                          | Intermediate  |
| Use PPO for continuous control (e.g. LunarLanderContinuous).             | Intermediate  |
| Implement PPO with Generalized Advantage Estimation.                      | Intermediate  |
| Develop a custom environment (e.g. robotic control) for PPO.              | Advanced      |
| Use distributed PPO across multiple workers for speed.                    | Advanced      |
| Research clipping parameters’ effect on PPO stability.                    | Advanced      |
| Combine PPO with curiosity-driven exploration.                           | Advanced      |
| Implement meta-learning (MAML) on top of PPO policy.                     | Advanced      |

### Actor-Critic (A3C/A2C)
| Project Idea                                                               | Difficulty    |
|----------------------------------------------------------------------------|---------------|
| Train an A2C agent on the CartPole environment.                          | Beginner      |
| Use an asynchronous runner to train A3C on Pong.                         | Beginner      |
| Implement entropy regularization for exploration.                         | Intermediate  |
| Use a recurrent policy (LSTM) in actor-critic agent.                    | Intermediate  |
| Compare A2C vs DQN on a grid-world task.                               | Intermediate  |
| Develop a multi-threaded A3C for faster training.                       | Advanced      |
| Research IMPALA (Importance-weighted Actor-Learner) architecture.        | Advanced      |
| Apply actor-critic to train an agent in MuJoCo.                          | Advanced      |
| Integrate self-play in actor-critic for games like Go.                  | Advanced      |
| Combine PPO and A3C ideas for a hybrid agent.                           | Advanced      |

### Soft Actor-Critic (SAC)
| Project Idea                                                               | Difficulty    |
|----------------------------------------------------------------------------|---------------|
| Train a SAC agent on Pendulum-v0.                                        | Beginner      |
| Use SAC to solve a continuous control (e.g. Walker2d).                   | Beginner      |
| Tune SAC’s entropy temperature parameter.                               | Intermediate  |
| Compare SAC vs DDPG on a benchmark task.                               | Intermediate  |
| Use twin Q-networks as in SAC for stability.                           | Intermediate  |
| Implement SAC for a custom robotics control problem.                     | Advanced      |
| Research automatic entropy tuning in SAC.                                | Advanced      |
| Combine SAC with HER (Hindsight Experience Replay).                     | Advanced      |
| Use SAC for multi-agent cooperative scenarios.                           | Advanced      |
| Explore batch-constrained SAC for offline RL.                            | Advanced      |

### Policy Gradients (REINFORCE, A2C)
| Project Idea                                                               | Difficulty    |
|----------------------------------------------------------------------------|---------------|
| Implement REINFORCE to solve CartPole.                                 | Beginner      |
| Use Monte Carlo returns for a simple policy gradient.                    | Beginner      |
| Add a baseline to REINFORCE to reduce variance.                        | Intermediate  |
| Apply A2C (advantage actor-critic) on a toy game.                      | Intermediate  |
| Use reward shaping to speed up learning.                               | Intermediate  |
| Research variance-reduction techniques for policy gradients.            | Advanced      |
| Implement trust region policy optimization (TRPO) for comparison.        | Advanced      |
| Use policy gradients with curriculum learning tasks.                    | Advanced      |
| Develop a hierarchical policy gradient agent (options).                | Advanced      |
| Integrate meta-learning to adapt learning rates in PG.                 | Advanced      |

### Multi-Agent Reinforcement Learning
| Project Idea                                                               | Difficulty    |
|----------------------------------------------------------------------------|---------------|
| Train two agents in a simple cooperative game (e.g. Pursuit).           | Beginner      |
| Implement self-play between two simple agents.                           | Beginner      |
| Use MADDPG (Multi-Agent DDPG) for cooperative tasks.                    | Intermediate  |
| Train agents with a shared reward signal (cooperative).                | Intermediate  |
| Implement competitive self-play with a reward zero-sum game.            | Intermediate  |
| Develop communication protocols among agents.                            | Advanced      |
| Research CTDE (Centralized Training Decentralized Execution).            | Advanced      |
| Use Independent Q-Learning in a multi-agent grid world.                 | Advanced      |
| Explore federated reinforcement learning across agents.                  | Advanced      |
| Investigate game-theoretic equilibrium in MARL.                          | Advanced      |

### Hierarchical Reinforcement Learning
| Project Idea                                                               | Difficulty    |
|----------------------------------------------------------------------------|---------------|
| Implement subgoal division in a simple maze.                             | Beginner      |
| Use option-critic framework on a two-room navigation.                   | Beginner      |
| Create high-level and low-level policies for a task (e.g. Fetch).      | Intermediate  |
| Use subroutines (options) to break down tasks in Atari.                | Intermediate  |
| Apply feudal networks (manager-worker) on a game.                       | Intermediate  |
| Train HRL on tasks requiring long-term planning (Montezuma’s Revenge).   | Advanced      |
| Research skill discovery methods for HRL.                                | Advanced      |
| Combine HRL with intrinsic motivation (curiosity).                      | Advanced      |
| Implement MAXQ decomposition for a robot task.                           | Advanced      |
| Use hierarchical policies in multi-agent coordination.                  | Advanced      |

[![follow banner](https://github.com/user-attachments/assets/d1b3ca08-dfea-403d-b4f1-613cedb83e11)](https://linktr.ee/mlinguist)
