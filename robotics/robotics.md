# Robotics Hub
## Full Project Database
* [Jump to tables by topic](#tables-by-topic)

| Topic                                     | Project Idea                                                               | Difficulty    |
|-------------------------------------------|----------------------------------------------------------------------------|---------------|
| Robot Operating System (ROS)              | Simulate a robot arm in Gazebo using ROS.                                | Beginner      |
| Robot Operating System (ROS)              | Use ROS to move a turtlebot to a target position.                         | Beginner      |
| Robot Operating System (ROS)              | Implement a ROS node for processing camera images.                        | Intermediate  |
| Robot Operating System (ROS)              | Publish and subscribe topics for robot sensor data.                        | Intermediate  |
| Robot Operating System (ROS)              | Create a ROS launch file for multi-node setups.                            | Intermediate  |
| Robot Operating System (ROS)              | Develop a custom ROS message type and node.                               | Advanced      |
| Robot Operating System (ROS)              | Build a mobile robot SLAM stack in ROS.                                   | Advanced      |
| Robot Operating System (ROS)              | Integrate ROS with a drone autopilot (e.g. PX4).                           | Advanced      |
| Robot Operating System (ROS)              | Implement real-time obstacle avoidance in ROS.                             | Advanced      |
| Robot Operating System (ROS)              | Contribute a new ROS package for a novel sensor.                          | Advanced      |
| SLAM (Simultaneous Localization and Mapping) | Use ORB-SLAM to map a small indoor environment.                            | Beginner      |
| SLAM (Simultaneous Localization and Mapping) | Visualize a 2D map from a robot’s laser scan.                             | Beginner      |
| SLAM (Simultaneous Localization and Mapping) | Implement a particle-filter-based SLAM (FastSLAM).                        | Intermediate  |
| SLAM (Simultaneous Localization and Mapping) | Integrate IMU data into a visual SLAM algorithm.                        | Intermediate  |
| SLAM (Simultaneous Localization and Mapping) | Test ORB-SLAM on a live camera feed.                                     | Intermediate  |
| SLAM (Simultaneous Localization and Mapping) | Develop a 3D SLAM system using an RGB-D camera.                           | Advanced      |
| SLAM (Simultaneous Localization and Mapping) | Combine LiDAR and visual SLAM for robust mapping.                         | Advanced      |
| SLAM (Simultaneous Localization and Mapping) | Implement loop-closure detection for improved accuracy.                   | Advanced      |
| SLAM (Simultaneous Localization and Mapping) | Create a multi-robot SLAM system for mapping large areas.                 | Advanced      |
| SLAM (Simultaneous Localization and Mapping) | Research SLAM in difficult environments (underwater, etc.).                | Advanced      |
| Path Planning (A*, RRT)                   | Implement A* search on a 2D grid maze.                                    | Beginner      |
| Path Planning (A*, RRT)                   | Plan a collision-free path for a point robot.                             | Beginner      |
| Path Planning (A*, RRT)                   | Use RRT to plan a random exploration path.                                | Intermediate  |
| Path Planning (A*, RRT)                   | Combine A* with robot kinematic constraints.                              | Intermediate  |
| Path Planning (A*, RRT)                   | Optimize a path for minimal time or distance.                             | Intermediate  |
| Path Planning (A*, RRT)                   | Implement RRT* for asymptotically optimal paths.                           | Advanced      |
| Path Planning (A*, RRT)                   | Use D* Lite for dynamic re-planning in changing maps.                     | Advanced      |
| Path Planning (A*, RRT)                   | Plan coordinated paths for multiple robots.                               | Advanced      |
| Path Planning (A*, RRT)                   | Integrate path planning with ROS navigation stack.                        | Advanced      |
| Path Planning (A*, RRT)                   | Research kinodynamic planning for manipulator arms.                       | Advanced      |
| Control (PID / MPC)                       | Tune a PID controller for a balancing robot.                              | Beginner      |
| Control (PID / MPC)                       | Implement a PID speed controller for a DC motor.                          | Beginner      |
| Control (PID / MPC)                       | Design a PID temperature controller (analogous).                          | Intermediate  |
| Control (PID / MPC)                       | Implement a simple Model Predictive Control for a system.                 | Intermediate  |
| Control (PID / MPC)                       | Use PID for a line-following robot.                                       | Intermediate  |
| Control (PID / MPC)                       | Develop MPC for a quadrotor altitude control.                             | Advanced      |
| Control (PID / MPC)                       | Combine PID with feedforward for precise control.                         | Advanced      |
| Control (PID / MPC)                       | Apply LQR (Linear Quadratic Regulator) for optimal control.               | Advanced      |
| Control (PID / MPC)                       | Design robust control to handle model uncertainty.                        | Advanced      |
| Control (PID / MPC)                       | Research nonlinear control (sliding mode) for robot tasks.                | Advanced      |
| Inverse Kinematics (IK)                   | Compute joint angles for a 2-link arm to reach a point.                   | Beginner      |
| Inverse Kinematics (IK)                   | Use geometric IK for a 3DOF planar manipulator.                           | Beginner      |
| Inverse Kinematics (IK)                   | Implement a numeric IK solver (CCD) for a robot arm.                      | Intermediate  |
| Inverse Kinematics (IK)                   | Integrate IK with ROS MoveIt for arm planning.                            | Intermediate  |
| Inverse Kinematics (IK)                   | Solve IK for a humanoid leg reaching a target foot position.              | Intermediate  |
| Inverse Kinematics (IK)                   | Develop an analytical IK solution for a complex arm.                      | Advanced      |
| Inverse Kinematics (IK)                   | Include joint limits and singularity avoidance in IK.                     | Advanced      |
| Inverse Kinematics (IK)                   | Use IKFast or Trac-IK for fast IK computation.                           | Advanced      |
| Inverse Kinematics (IK)                   | Research whole-body IK for humanoid robots.                              | Advanced      |
| Inverse Kinematics (IK)                   | Implement IK for redundant robots (e.g. 7-DOF arm).                       | Advanced      |
| Trajectory Planning                       | Plan a straight-line joint trajectory for a robot arm.                    | Beginner      |
| Trajectory Planning                       | Interpolate waypoints for a mobile robot path.                            | Beginner      |
| Trajectory Planning                       | Use cubic splines for smooth trajectory between points.                   | Intermediate  |
| Trajectory Planning                       | Implement time-scaling to meet velocity/acceleration limits.              | Intermediate  |
| Trajectory Planning                       | Compute trajectories that avoid obstacles in joint space.                 | Intermediate  |
| Trajectory Planning                       | Develop minimum-time trajectories under torque limits.                    | Advanced      |
| Trajectory Planning                       | Implement collision-free trajectories for multiple arms.                  | Advanced      |
| Trajectory Planning                       | Optimize via polynomial trajectories for legged robots.                   | Advanced      |
| Trajectory Planning                       | Combine trajectory planning with vision feedback (visual servoing).      | Advanced      |
| Trajectory Planning                       | Research kinodynamic constraints in trajectory generation.                | Advanced      |
| Gazebo / Simulation                       | Load a robot model (URDF) in Gazebo.                                      | Beginner      |
| Gazebo / Simulation                       | Simulate simple sensor (lidar/camera) in Gazebo.                          | Beginner      |
| Gazebo / Simulation                       | Create a Gazebo world with obstacles.                                   | Intermediate  |
| Gazebo / Simulation                       | Run a ROS robot in Gazebo for testing algorithms.                         | Intermediate  |
| Gazebo / Simulation                       | Use PyBullet to simulate a custom robot.                                | Intermediate  |
| Gazebo / Simulation                       | Integrate physics-based simulation for soft robotics.                     | Advanced      |
| Gazebo / Simulation                       | Simulate multi-robot interaction in a Gazebo environment.                 | Advanced      |
| Gazebo / Simulation                       | Connect Gazebo simulation to reinforcement learning (OpenAI Gym).        | Advanced      |
| Gazebo / Simulation                       | Develop a real-to-sim pipeline for sensor emulation.                     | Advanced      |
| Gazebo / Simulation                       | Research simulation-to-reality transfer (Domain Randomization).           | Advanced      |
| Robot Learning / Adaptation               | Use imitation learning to mimic a simple demonstration.                    | Beginner      |
| Robot Learning / Adaptation               | Apply Gaussian process regression to predict robot dynamics.              | Beginner      |
| Robot Learning / Adaptation               | Implement covariance matrix adaptation evolution strategy (CMA-ES) for tuning.| Intermediate  |
| Robot Learning / Adaptation               | Use RL (e.g. PPO) to learn a locomotion policy.                          | Intermediate  |
| Robot Learning / Adaptation               | Apply Meta-Learning (MAML) for fast adaptation on tasks.               | Intermediate  |
| Robot Learning / Adaptation               | Research domain randomization for robust visual policies.                | Advanced      |
| Robot Learning / Adaptation               | Implement online adaptation of control gains with RL.                    | Advanced      |
| Robot Learning / Adaptation               | Combine policy distillation for multi-skill learning.                     | Advanced      |
| Robot Learning / Adaptation               | Use model-based RL for sample-efficient learning.                       | Advanced      |
| Robot Learning / Adaptation               | Explore active learning to improve robot perception models.             | Advanced      |

## Tables by Topic
### Robot Operating System (ROS)
| Project Idea                                                      | Difficulty    |
|-------------------------------------------------------------------|---------------|
| Simulate a robot arm in Gazebo using ROS.                         | Beginner      |
| Use ROS to move a turtlebot to a target position.                  | Beginner      |
| Implement a ROS node for processing camera images.                 | Intermediate  |
| Publish and subscribe topics for robot sensor data.                 | Intermediate  |
| Create a ROS launch file for multi-node setups.                     | Intermediate  |
| Develop a custom ROS message type and node.                        | Advanced      |
| Build a mobile robot SLAM stack in ROS.                            | Advanced      |
| Integrate ROS with a drone autopilot (e.g. PX4).                   | Advanced      |
| Implement real-time obstacle avoidance in ROS.                      | Advanced      |
| Contribute a new ROS package for a novel sensor.                   | Advanced      |

### SLAM (Simultaneous Localization and Mapping)
| Project Idea                                                      | Difficulty    |
|-------------------------------------------------------------------|---------------|
| Use ORB-SLAM to map a small indoor environment.                   | Beginner      |
| Visualize a 2D map from a robot’s laser scan.                    | Beginner      |
| Implement a particle-filter-based SLAM (FastSLAM).                | Intermediate  |
| Integrate IMU data into a visual SLAM algorithm.                | Intermediate  |
| Test ORB-SLAM on a live camera feed.                             | Intermediate  |
| Develop a 3D SLAM system using an RGB-D camera.                  | Advanced      |
| Combine LiDAR and visual SLAM for robust mapping.                | Advanced      |
| Implement loop-closure detection for improved accuracy.          | Advanced      |
| Create a multi-robot SLAM system for mapping large areas.        | Advanced      |
| Research SLAM in difficult environments (underwater, etc.).       | Advanced      |

### Path Planning (A*, RRT)
| Project Idea                                                      | Difficulty    |
|-------------------------------------------------------------------|---------------|
| Implement A* search on a 2D grid maze.                            | Beginner      |
| Plan a collision-free path for a point robot.                    | Beginner      |
| Use RRT to plan a random exploration path.                        | Intermediate  |
| Combine A* with robot kinematic constraints.                     | Intermediate  |
| Optimize a path for minimal time or distance.                    | Intermediate  |
| Implement RRT* for asymptotically optimal paths.                  | Advanced      |
| Use D* Lite for dynamic re-planning in changing maps.            | Advanced      |
| Plan coordinated paths for multiple robots.                      | Advanced      |
| Integrate path planning with ROS navigation stack.               | Advanced      |
| Research kinodynamic planning for manipulator arms.              | Advanced      |

### Control (PID / MPC)
| Project Idea                                                      | Difficulty    |
|-------------------------------------------------------------------|---------------|
| Tune a PID controller for a balancing robot.                      | Beginner      |
| Implement a PID speed controller for a DC motor.                  | Beginner      |
| Design a PID temperature controller (analogous).                  | Intermediate  |
| Implement a simple Model Predictive Control for a system.         | Intermediate  |
| Use PID for a line-following robot.                               | Intermediate  |
| Develop MPC for a quadrotor altitude control.                     | Advanced      |
| Combine PID with feedforward for precise control.                 | Advanced      |
| Apply LQR (Linear Quadratic Regulator) for optimal control.       | Advanced      |
| Design robust control to handle model uncertainty.                | Advanced      |
| Research nonlinear control (sliding mode) for robot tasks.        | Advanced      |

### Inverse Kinematics (IK)
| Project Idea                                                      | Difficulty    |
|-------------------------------------------------------------------|---------------|
| Compute joint angles for a 2-link arm to reach a point.           | Beginner      |
| Use geometric IK for a 3DOF planar manipulator.                   | Beginner      |
| Implement a numeric IK solver (CCD) for a robot arm.              | Intermediate  |
| Integrate IK with ROS MoveIt for arm planning.                    | Intermediate  |
| Solve IK for a humanoid leg reaching a target foot position.      | Intermediate  |
| Develop an analytical IK solution for a complex arm.              | Advanced      |
| Include joint limits and singularity avoidance in IK.             | Advanced      |
| Use IKFast or Trac-IK for fast IK computation.                   | Advanced      |
| Research whole-body IK for humanoid robots.                      | Advanced      |
| Implement IK for redundant robots (e.g. 7-DOF arm).               | Advanced      |

### Trajectory Planning
| Project Idea                                                      | Difficulty    |
|-------------------------------------------------------------------|---------------|
| Plan a straight-line joint trajectory for a robot arm.            | Beginner      |
| Interpolate waypoints for a mobile robot path.                    | Beginner      |
| Use cubic splines for smooth trajectory between points.           | Intermediate  |
| Implement time-scaling to meet velocity/acceleration limits.      | Intermediate  |
| Compute trajectories that avoid obstacles in joint space.         | Intermediate  |
| Develop minimum-time trajectories under torque limits.            | Advanced      |
| Implement collision-free trajectories for multiple arms.          | Advanced      |
| Optimize via polynomial trajectories for legged robots.           | Advanced      |
| Combine trajectory planning with vision feedback (visual servoing).| Advanced      |
| Research kinodynamic constraints in trajectory generation.        | Advanced      |


### Gazebo / Simulation
| Project Idea                                                      | Difficulty    |
|-------------------------------------------------------------------|---------------|
| Load a robot model (URDF) in Gazebo.                              | Beginner      |
| Simulate simple sensor (lidar/camera) in Gazebo.                 | Beginner      |
| Create a Gazebo world with obstacles.                           | Intermediate  |
| Run a ROS robot in Gazebo for testing algorithms.                  | Intermediate  |
| Use PyBullet to simulate a custom robot.                        | Intermediate  |
| Integrate physics-based simulation for soft robotics.             | Advanced      |
| Simulate multi-robot interaction in a Gazebo environment.         | Advanced      |
| Connect Gazebo simulation to reinforcement learning (OpenAI Gym).| Advanced      |
| Develop a real-to-sim pipeline for sensor emulation.             | Advanced      |
| Research simulation-to-reality transfer (Domain Randomization).   | Advanced      |

### Robot Learning / Adaptation
| Project Idea                                                               | Difficulty    |
|----------------------------------------------------------------------------|---------------|
| Use imitation learning to mimic a simple demonstration.                    | Beginner      |
| Apply Gaussian process regression to predict robot dynamics.              | Beginner      |
| Implement covariance matrix adaptation evolution strategy (CMA-ES) for tuning.| Intermediate  |
| Use RL (e.g. PPO) to learn a locomotion policy.                          | Intermediate  |
| Apply Meta-Learning (MAML) for fast adaptation on tasks.               | Intermediate  |
| Research domain randomization for robust visual policies.                | Advanced      |
| Implement online adaptation of control gains with RL.                    | Advanced      |
| Combine policy distillation for multi-skill learning.                     | Advanced      |
| Use model-based RL for sample-efficient learning.                       | Advanced      |
| Explore active learning to improve robot perception models.             | Advanced      |

[![follow banner](https://github.com/user-attachments/assets/d1b3ca08-dfea-403d-b4f1-613cedb83e11)](https://linktr.ee/mlinguist)
