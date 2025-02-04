import torch

SERVER_ADDRESS = "localhost"
SERVER_PORT = 12345
TOTAL_GLOBAL_EPOCH = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GLOBAL_TOTAL_CLIENTS = 3