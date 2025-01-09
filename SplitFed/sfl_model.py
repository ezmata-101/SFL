import torch
import torch.nn as nn
import torch.nn.functional as F


# input_layer = Input(shape=(28, 28))  # Input layer
# # flatten_layer = Flatten()(input_layer)  # Flattening the input
# # hidden_layer_1 = Dense(128, activation="relu")(flatten_layer)  # First hidden layer
# # client_model = Model(inputs=input_layer, outputs=hidden_layer_1)
# #
# # # Server-side model
# # server_input = Input(shape=(128,))  # Input from the client-side model
# # hidden_layer_2 = Dense(64, activation="relu")(server_input)  # Second hidden layer
# # output_layer = Dense(10, activation="softmax")(hidden_layer_2)  # Output layer
# # server_model = Model(inputs=server_input, outputs=output_layer)

# for mnist dataset
class ClientModel(nn.Module):
    def __init__(self):
        super(ClientModel, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(784, 128)
        self.activation1 = nn.ReLU()

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.activation1(x)
        return x


class ServerModel(nn.Module):
    def __init__(self):
        super(ServerModel, self).__init__()
        self.fc1 = nn.Linear(128, 64)
        self.activation1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 10)
        self.activation2 = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation1(x)
        x = self.fc2(x)
        x = self.activation2(x)
        return x


class CompleteModel(nn.Module):
    def __init__(self):
        super(CompleteModel, self).__init__()
        self.client = ClientModel()
        self.server = ServerModel()

    def forward(self, x):
        x = self.client.forward(x)
        x = self.server.forward(x)
        return x


