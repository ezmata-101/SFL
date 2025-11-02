# SFL — SplitFed: When Federated Learning Meets Split Learning

**SplitFed (SFL)** system combines the parallelism of **Federated Learning (FL)** with the model-splitting and privacy benefits of **Split Learning (SL)**. The base work shares the idea of SFL and provides a simple simulation. This repository implements a more realistic SFL system with socket-based communication.

1) **Baselines**  
   - **SL**: client–server split model with relay-style training  
   - **FL**: parallel client training with FedAvg

**Reports & slides** are in `submissions/` directory.

## How to run
Read `server.py` and `client.py` to find out how to run :\. 