# Federated Learning Hub
## Full Project Database
* [Jump to tables by topic](#tables-by-topic)

| Topic                                     | Project Idea                                                               | Difficulty    |
|-------------------------------------------|----------------------------------------------------------------------------|---------------|
| Federated Averaging (FedAvg)              | Simulate FedAvg training across two data silos.                            | Beginner      |
| Federated Averaging (FedAvg)              | Implement simple federated averaging on synthetic data.                    | Beginner      |
| Federated Averaging (FedAvg)              | Compare FedAvg vs local training for convergence speed.                    | Intermediate  |
| Federated Averaging (FedAvg)              | Use FedAvg with PyTorch across multiple clients.                           | Intermediate  |
| Federated Averaging (FedAvg)              | Measure communication rounds needed for FedAvg.                            | Intermediate  |
| Federated Averaging (FedAvg)              | Add client weighting to FedAvg based on data size.                        | Advanced      |
| Federated Averaging (FedAvg)              | Research momentum or adaptive updates in FedAvg.                            | Advanced      |
| Federated Averaging (FedAvg)              | Use FedAvg for cross-device image classifier training.                     | Advanced      |
| Federated Averaging (FedAvg)              | Extend FedAvg with personalized learning rates per client.                 | Advanced      |
| Federated Averaging (FedAvg)              | Combine FedAvg with secure aggregation for privacy.                        | Advanced      |
| Secure Aggregation                        | Implement secret sharing of model updates.                               | Beginner      |
| Secure Aggregation                        | Use PySyft to demo secure aggregation.                                    | Beginner      |
| Secure Aggregation                        | Add noise to model updates for basic privacy.                            | Intermediate  |
| Secure Aggregation                        | Encrypt gradients with Paillier for secure sum.                          | Intermediate  |
| Secure Aggregation                        | Use threshold encryption to aggregate only.                              | Intermediate  |
| Secure Aggregation                        | Develop a secure protocol for untrusted servers.                           | Advanced      |
| Secure Aggregation                        | Research compression + secure aggregation trade-offs.                      | Advanced      |
| Secure Aggregation                        | Implement blockchain logging for aggregation audit.                        | Advanced      |
| Secure Aggregation                        | Evaluate network overhead of secure aggregation.                           | Advanced      |
| Secure Aggregation                        | Combine secure aggregation with differential privacy.                       | Advanced      |
| Differential Privacy                      | Add Gaussian noise to gradients for privacy.                             | Beginner      |
| Differential Privacy                      | Use DP-SGD to train a private ML model.                                 | Beginner      |
| Differential Privacy                      | Compute privacy budget (ε) for a training run.                           | Intermediate  |
| Differential Privacy                      | Compare utility vs privacy trade-off in FL.                              | Intermediate  |
| Differential Privacy                      | Apply local differential privacy on user data.                           | Intermediate  |
| Differential Privacy                      | Implement RDP (Rényi DP) for tighter analysis.                           | Advanced      |
| Differential Privacy                      | Integrate DP with FedAvg (DP-FedAvg).                                    | Advanced      |
| Differential Privacy                      | Use advanced composition to link privacy budgets.                        | Advanced      |
| Differential Privacy                      | Apply DP in heterogeneous federated scenarios.                           | Advanced      |
| Differential Privacy                      | Research optimal noise calibration for DP in FL.                         | Advanced      |
| Federated Multi-Task Learning             | Train separate models per client and average lightly.                    | Beginner      |
| Federated Multi-Task Learning             | Use personalized layers for each client while sharing others.            | Beginner      |
| Federated Multi-Task Learning             | Implement clustered federated learning (client grouping).                | Intermediate  |
| Federated Multi-Task Learning             | Use meta-learning (MAML) for federated personalization.                 | Intermediate  |
| Federated Multi-Task Learning             | Implement FedPer (personalized layers) in practice.                     | Intermediate  |
| Federated Multi-Task Learning             | Research clustered federated optimization algorithms.                    | Advanced      |
| Federated Multi-Task Learning             | Develop client clustering based on data similarity.                      | Advanced      |
| Federated Multi-Task Learning             | Interpolate between global and personal models.                          | Advanced      |
| Federated Multi-Task Learning             | Compare global vs personalized ensemble performance.                     | Advanced      |
| Federated Multi-Task Learning             | Use federated multi-task for mobile keyboard prediction.                 | Advanced      |
| Cross-Silo vs Cross-Device FL             | Simulate a small number of silos (e.g. hospitals).                       | Beginner      |
| Cross-Silo vs Cross-Device FL             | Simulate many devices (e.g. smartphones) with partial data.            | Beginner      |
| Cross-Silo vs Cross-Device FL             | Study impact of uneven data distribution (skew).                         | Intermediate  |
| Cross-Silo vs Cross-Device FL             | Implement asynchronous updates for many clients.                         | Intermediate  |
| Cross-Silo vs Cross-Device FL             | Use participant sampling (e.g. 10% devices) each round.                 | Intermediate  |
| Cross-Silo vs Cross-Device FL             | Research fairness across clients in cross-silo FL.                       | Advanced      |
| Cross-Silo vs Cross-Device FL             | Develop solutions for client drop-out and stragglers.                    | Advanced      |
| Cross-Silo vs Cross-Device FL             | Analyze FL convergence under intermittent connectivity.                  | Advanced      |
| Cross-Silo vs Cross-Device FL             | Implement hierarchical FL (region→cloud aggregation).                     | Advanced      |
| Cross-Silo vs Cross-Device FL             | Compare privacy/performance trade-offs in each setting.                  | Advanced      |
| Open-Source FL Tools (TensorFlow Federated, PySyft) | Explore TensorFlow Federated examples on sample data.               | Beginner      |
| Open-Source FL Tools (TensorFlow Federated, PySyft) | Use PySyft to simulate federated model updates.                    | Beginner      |
| Open-Source FL Tools (TensorFlow Federated, PySyft) | Integrate TFF with a Keras model for FL demo.                       | Intermediate  |
| Open-Source FL Tools (TensorFlow Federated, PySyft) | Combine Syft with SMPC for secure model training.                   | Intermediate  |
| Open-Source FL Tools (TensorFlow Federated, PySyft) | Deploy a simple FL API service with PyGrid.                         | Intermediate  |
| Open-Source FL Tools (TensorFlow Federated, PySyft) | Contribute a new feature to an open FL library.                    | Advanced      |
| Open-Source FL Tools (TensorFlow Federated, PySyft) | Benchmark different FL frameworks (TFF vs PySyft).                  | Advanced      |
| Open-Source FL Tools (TensorFlow Federated, PySyft) | Implement a cross-platform FL experiment (Python/Rust).             | Advanced      |
| Open-Source FL Tools (TensorFlow Federated, PySyft) | Research federated learning APIs for mobile deployment.             | Advanced      |
| Open-Source FL Tools (TensorFlow Federated, PySyft) | Securely share a pretrained model across libraries via ONNX.        | Advanced      |

## Tables by Topic
### Federated Averaging (FedAvg)
| Project Idea                                                               | Difficulty    |
|----------------------------------------------------------------------------|---------------|
| Simulate FedAvg training across two data silos.                            | Beginner      |
| Implement simple federated averaging on synthetic data.                    | Beginner      |
| Compare FedAvg vs local training for convergence speed.                    | Intermediate  |
| Use FedAvg with PyTorch across multiple clients.                           | Intermediate  |
| Measure communication rounds needed for FedAvg.                            | Intermediate  |
| Add client weighting to FedAvg based on data size.                        | Advanced      |
| Research momentum or adaptive updates in FedAvg.                            | Advanced      |
| Use FedAvg for cross-device image classifier training.                     | Advanced      |
| Extend FedAvg with personalized learning rates per client.                 | Advanced      |
| Combine FedAvg with secure aggregation for privacy.                        | Advanced      |

### Secure Aggregation
| Project Idea                                                               | Difficulty    |
|----------------------------------------------------------------------------|---------------|
| Implement secret sharing of model updates.                               | Beginner      |
| Use PySyft to demo secure aggregation.                                    | Beginner      |
| Add noise to model updates for basic privacy.                            | Intermediate  |
| Encrypt gradients with Paillier for secure sum.                          | Intermediate  |
| Use threshold encryption to aggregate only.                              | Intermediate  |
| Develop a secure protocol for untrusted servers.                           | Advanced      |
| Research compression + secure aggregation trade-offs.                      | Advanced      |
| Implement blockchain logging for aggregation audit.                        | Advanced      |
| Evaluate network overhead of secure aggregation.                           | Advanced      |
| Combine secure aggregation with differential privacy.                       | Advanced      |

### Differential Privacy
| Project Idea                                                               | Difficulty    |
|----------------------------------------------------------------------------|---------------|
| Add Gaussian noise to gradients for privacy.                             | Beginner      |
| Use DP-SGD to train a private ML model.                                 | Beginner      |
| Compute privacy budget (ε) for a training run.                           | Intermediate  |
| Compare utility vs privacy trade-off in FL.                              | Intermediate  |
| Apply local differential privacy on user data.                           | Intermediate  |
| Implement RDP (Rényi DP) for tighter analysis.                           | Advanced      |
| Integrate DP with FedAvg (DP-FedAvg).                                    | Advanced      |
| Use advanced composition to link privacy budgets.                        | Advanced      |
| Apply DP in heterogeneous federated scenarios.                           | Advanced      |
| Research optimal noise calibration for DP in FL.                         | Advanced      |

### Federated Multi-Task Learning
| Project Idea                                                               | Difficulty    |
|----------------------------------------------------------------------------|---------------|
| Train separate models per client and average lightly.                    | Beginner      |
| Use personalized layers for each client while sharing others.            | Beginner      |
| Implement clustered federated learning (client grouping).                | Intermediate  |
| Use meta-learning (MAML) for federated personalization.                 | Intermediate  |
| Implement FedPer (personalized layers) in practice.                     | Intermediate  |
| Research clustered federated optimization algorithms.                    | Advanced      |
| Develop client clustering based on data similarity.                      | Advanced      |
| Interpolate between global and personal models.                          | Advanced      |
| Compare global vs personalized ensemble performance.                     | Advanced      |
| Use federated multi-task for mobile keyboard prediction.                 | Advanced      |

### Cross-Silo vs Cross-Device FL
| Project Idea                                                               | Difficulty    |
|----------------------------------------------------------------------------|---------------|
| Simulate a small number of silos (e.g. hospitals).                       | Beginner      |
| Simulate many devices (e.g. smartphones) with partial data.            | Beginner      |
| Study impact of uneven data distribution (skew).                         | Intermediate  |
| Implement asynchronous updates for many clients.                         | Intermediate  |
| Use participant sampling (e.g. 10% devices) each round.                 | Intermediate  |
| Research fairness across clients in cross-silo FL.                       | Advanced      |
| Develop solutions for client drop-out and stragglers.                    | Advanced      |
| Analyze FL convergence under intermittent connectivity.                  | Advanced      |
| Implement hierarchical FL (region→cloud aggregation).                     | Advanced      |
| Compare privacy/performance trade-offs in each setting.                  | Advanced      |

### Open-Source FL Tools (TensorFlow Federated, PySyft)
| Project Idea                                                               | Difficulty    |
|----------------------------------------------------------------------------|---------------|
| Explore TensorFlow Federated examples on sample data.               | Beginner      |
| Use PySyft to simulate federated model updates.                    | Beginner      |
| Integrate TFF with a Keras model for FL demo.                       | Intermediate  |
| Combine Syft with SMPC for secure model training.                   | Intermediate  |
| Deploy a simple FL API service with PyGrid.                         | Intermediate  |
| Contribute a new feature to an open FL library.                    | Advanced      |
| Benchmark different FL frameworks (TFF vs PySyft).                  | Advanced      |
| Implement a cross-platform FL experiment (Python/Rust).             | Advanced      |
| Research federated learning APIs for mobile deployment.             | Advanced      |
| Securely share a pretrained model across libraries via ONNX.        | Advanced      |
