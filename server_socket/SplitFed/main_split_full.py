import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from argparse import ArgumentParser, Namespace

from SplitFed.models import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

GLOBAL_SHARED_SPLIT_LAYER_TENSOR = None
GLOBAL_SHARED_LABELS = None
GLOBAL_SHARED_GRAD_FROM_SERVER = None
GLOBAL_SHARED_EPOCH = 0
GLOBAL_SHARED_I = 0

FEATURE_LIST = None
LABEL_LIST = None

CLIENT_NO = -1

def server_train():
    global GLOBAL_SHARED_SPLIT_LAYER_TENSOR, GLOBAL_SHARED_LABELS, GLOBAL_SHARED_GRAD_FROM_SERVER, GLOBAL_SHARED_EPOCH, GLOBAL_SHARED_I

    server_model = ServerModel().to(device)
    server_optimizer = optim.Adam(server_model.parameters(), lr=0.001)
    loss_criteria = nn.CrossEntropyLoss()

    while True:
        print("Waiting for client")
        input("Press Enter to continue...")
        GLOBAL_SHARED_SPLIT_LAYER_TENSOR = torch.load(f"split_layer_tensor_{GLOBAL_SHARED_EPOCH}_{GLOBAL_SHARED_I}.pt")
        GLOBAL_SHARED_LABELS = torch.load(f"labels_{GLOBAL_SHARED_EPOCH}_{GLOBAL_SHARED_I}.pt")

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

        torch.save(split_layer_tensor.grad, f"grad_from_server_{GLOBAL_SHARED_EPOCH}_{GLOBAL_SHARED_I}.pt")

        GLOBAL_SHARED_SPLIT_LAYER_TENSOR = None
        GLOBAL_SHARED_LABELS = None

client_model = ClientModel().to(device)
client_optimizer = optim.Adam(client_model.parameters(), lr=0.001)

def client_train(comm_with_server, comm_with_fed_server = None):
    global GLOBAL_SHARED_SPLIT_LAYER_TENSOR, GLOBAL_SHARED_LABELS, GLOBAL_SHARED_GRAD_FROM_SERVER, GLOBAL_SHARED_EPOCH, GLOBAL_SHARED_I
    client_model.train()
    for epoch in range(5):
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            client_optimizer.zero_grad()
            split_layer_tensor = client_model(images)
            split_layer_tensor = split_layer_tensor.detach().requires_grad_(True)

            grads = comm_with_server(labels, split_layer_tensor, epoch, i)

            split_layer_tensor.backward(grads)
            client_optimizer.step()

        print(f"Client Epoch {epoch + 1} completed")

    if comm_with_fed_server:
        comm_with_fed_server(client_model, -1)
    print("Client training done")


def server_test():
    global FEATURE_LIST, LABEL_LIST
    if FEATURE_LIST is None or LABEL_LIST is None:
        print("No data from client")
        return

    server_model.eval()
    correct = 0
    total = LABEL_LIST.size(0)

    with torch.no_grad():
        server_output = server_model(FEATURE_LIST)
        loss = loss_criteria(server_output, LABEL_LIST)
        print(f"Server test loss: {loss.item()}")

        _, predicted = torch.max(server_output.data, 1)
        correct += (predicted == LABEL_LIST).sum().item()

    print(f"Server test accuracy: {correct / total}")


def test_split_model(final_client_model, server_model):
    final_client_model.eval()
    server_model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            split_layer_tensor = final_client_model(images)
            server_output = server_model(split_layer_tensor)
            _, predicted = torch.max(server_output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Test accuracy: {correct / total}")


def client_test():
    global FEATURE_LIST, LABEL_LIST
    client_model.eval()
    feature_list = []
    label_list = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            split_layer_tensor = client_model(images)
            feature_list.append(split_layer_tensor)
            label_list.append(labels)

    FEATURE_LIST, LABEL_LIST = torch.cat(feature_list), torch.cat(label_list)

    server_test()



def train():
    client_train()

def test():
    client_test()

def parse_arg() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--mode", type=str, default="server", help="server or client")
    parser.add_argument("--client_no", type=int, default=-1, help="client number")
    return parser.parse_args()

def main(mode, client_no):
    global CLIENT_NO
    CLIENT_NO = client_no
    if mode == "server":
        server_train()

    if mode == "client":
        client_train()
        client_test()

if __name__ == "__main__":
    args = parse_arg()
    main(args.mode, args.client_no)
