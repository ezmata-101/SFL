import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sfl_model import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

GLOBAL_SHARED_SPLIT_LAYER_TENSOR = None
GLOBAL_SHARED_LABELS = None
GLOBAL_SHARED_GRAD_FROM_SERVER = None
GLOBAL_SHARED_EPOCH = 0
GLOBAL_SHARED_I = 0

server_model = ServerModel().to(device)
server_optimizer = optim.SGD(server_model.parameters(), lr=0.01,
            momentum=0.9, weight_decay=5e-4)
loss_criteria = nn.CrossEntropyLoss()

def server_train():
    global GLOBAL_SHARED_SPLIT_LAYER_TENSOR, GLOBAL_SHARED_LABELS, GLOBAL_SHARED_GRAD_FROM_SERVER, GLOBAL_SHARED_EPOCH, GLOBAL_SHARED_I
    if GLOBAL_SHARED_SPLIT_LAYER_TENSOR is None or GLOBAL_SHARED_LABELS is None:
        print("No data from client")
        return

    server_optimizer.zero_grad()
    split_layer_tensor = GLOBAL_SHARED_SPLIT_LAYER_TENSOR
    labels = GLOBAL_SHARED_LABELS

    server_output = server_model(split_layer_tensor)
    loss = loss_criteria(server_output, labels)

    loss.backward()

    server_optimizer.step()
    split_layer_tensor.retain_grad()

    GLOBAL_SHARED_GRAD_FROM_SERVER = split_layer_tensor.grad
    GLOBAL_SHARED_SPLIT_LAYER_TENSOR = None
    GLOBAL_SHARED_LABELS = None

client_model = ClientModel().to(device)
client_optimizer = optim.SGD(client_model.parameters(), lr=0.01,
            momentum=0.9, weight_decay=5e-4)

def client_train():
    global GLOBAL_SHARED_SPLIT_LAYER_TENSOR, GLOBAL_SHARED_LABELS, GLOBAL_SHARED_GRAD_FROM_SERVER, GLOBAL_SHARED_EPOCH, GLOBAL_SHARED_I

    for epoch in range(5):
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            client_optimizer.zero_grad()
            split_layer_tensor = client_model(images)
            # split_layer_tensor.retain_grad()
            split_layer_tensor = split_layer_tensor.detach().requires_grad_(True)
            GLOBAL_SHARED_SPLIT_LAYER_TENSOR = split_layer_tensor
            GLOBAL_SHARED_LABELS = labels
            GLOBAL_SHARED_EPOCH = epoch
            GLOBAL_SHARED_I = i

            server_train()

            grads = GLOBAL_SHARED_GRAD_FROM_SERVER

            split_layer_tensor.backward(grads)
            client_optimizer.step()

            print(f"Client Epoch {epoch + 1}, Batch {i + 1} completed")

        print(f"Client Epoch {epoch + 1} completed")

    print("Client training done")

def train():
    client_train()

def test():
    pass

train()
