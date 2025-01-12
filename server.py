import socket
import threading
import os
import torch
import torch.nn as nn
import torch.optim as optim

from datetime import datetime


from SplitFed.main_split_full import *

from common import *
from communication import *


device = DEVICE


clients = {}
clients_nos = {}
clients_lock = threading.Lock()
server_socket = None

current_date_time = str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S")).replace("-", "")

server_file_directory_path = "shared_files/server/server_files"
client_file_directory_path = "shared_files/server/client_files"

server_log_file = f"server_log_{current_date_time}.txt"
specific_client_file_directory_path = None

CLIENT_COUNT = 0

GLOBAL_EPOCH = TOTAL_GLOBAL_EPOCH
TOTAL_CLIENTS = 3
CLIENT_EPOCH_COUNT = [0] * GLOBAL_EPOCH
CLIENT_MODEL_PATHS = [[None] * GLOBAL_EPOCH for _ in range(TOTAL_CLIENTS)]
CLIENT_VALIDATION_LOADER_PATHS = [[None] * GLOBAL_EPOCH for _ in range(TOTAL_CLIENTS)]
epoch_count_lock = threading.Lock()
CLIENT_WEIGHT = [1 / TOTAL_CLIENTS] * TOTAL_CLIENTS
AGGREGATION_DONE = [False] * GLOBAL_EPOCH
VALIDATION_SAFETY_LOCK = threading.Lock()

def start_training(client_socket, client_name):
    global specific_client_file_directory_path, device, CLIENT_EPOCH_COUNT, CLIENT_MODEL_PATHS, CLIENT_VALIDATION_LOADER_PATHS
    print(f"Client {client_name} training started.")

    server_model = ServerModel().to(device)
    server_optimizer = optim.Adam(server_model.parameters(), lr=0.001)
    loss_criteria = nn.CrossEntropyLoss()

    final_client_model_path = None
    final_validation_loader_path = None

    while True:
        try:
            msg = receive_message_as_json(client_socket)
            if msg.message_type == MessageType.REQUEST_TO_SEND_FILE:
                labels_path = msg.content
                receive_file(client_socket, specific_client_file_directory_path, labels_path)
                slt_message = receive_message_as_json(client_socket)
                if slt_message.message_type == MessageType.REQUEST_TO_SEND_FILE:
                    slt_path = slt_message.content
                    receive_file(client_socket, specific_client_file_directory_path, slt_path)
                    labels = torch.load(f"{specific_client_file_directory_path}/{labels_path}")
                    split_layer_tensor = torch.load(f"{specific_client_file_directory_path}/{slt_path}")

                    server_optimizer.zero_grad()
                    server_output = server_model(split_layer_tensor)
                    loss = loss_criteria(server_output, labels)

                    # print to log file
                    with open(f"shared_files/server/{server_log_file}", "a") as log_file:
                        log_file.write(f"{clients_nos[client_name]},{labels_path},loss,{loss}\n")

                    loss.backward()
                    server_optimizer.step()
                    split_layer_tensor.retain_grad()
                    torch.save(split_layer_tensor.grad, f"{specific_client_file_directory_path}/grads_{labels_path}")

                    send_file(client_socket, specific_client_file_directory_path, f"grads_{labels_path}")
                else:
                    print("Invalid message 1.")

            elif msg.message_type == MessageType.REQUEST_TO_SEND_FINAL_MODEL:
                    final_model_path = msg.content
                    receive_file(client_socket, specific_client_file_directory_path, final_model_path)
                    final_client_model_path = f"{specific_client_file_directory_path}/{final_model_path}"

            elif msg.message_type == MessageType.TRAINING_DONE:
                send_message_as_json(client_socket, MessageType.TRAINING_DONE)
                print(f"Client {client_name} training done.")

                if final_client_model_path:
                    test(final_client_model_path, server_model)
                break

            elif msg.message_type == MessageType.REQUEST_TO_SEND_VALIDATION_LOADER:
                validation_loader_path = msg.content
                final_validation_loader_path = f"{specific_client_file_directory_path}/{validation_loader_path}"
                receive_file(client_socket, specific_client_file_directory_path, validation_loader_path)

            elif msg.message_type == MessageType.REQUEST_TO_SEND_MODEL:
                model_path = msg.content
                global_epoch = int(model_path.split("_")[2].split(".")[0])
                receive_file(client_socket, specific_client_file_directory_path, model_path)
                final_client_model_path = f"{specific_client_file_directory_path}/{model_path}"

                val_acc = validate_model(final_client_model_path, server_model, final_validation_loader_path)
                if val_acc > 0:
                    with epoch_count_lock:
                        CLIENT_EPOCH_COUNT[global_epoch] += 1
                        CLIENT_MODEL_PATHS[clients_nos[client_name]][global_epoch] = final_client_model_path
                        CLIENT_VALIDATION_LOADER_PATHS[clients_nos[client_name]][global_epoch] = final_validation_loader_path
                        print(f"Client {client_name} validated model for epoch {global_epoch}, accuracy: {val_acc}")
                        # print to log file
                        with open(f"shared_files/server/{server_log_file}", "a") as log_file:
                            log_file.write(f"{clients_nos[client_name]},{model_path},val_acc,{val_acc}\n")

                        if CLIENT_EPOCH_COUNT[global_epoch] == TOTAL_CLIENTS:
                            send_fed_avg_model_to_clients(client_socket, global_epoch, send = False)

                    send_message_as_json(client_socket, MessageType.VALIDATION_DONE, "Server", client_name, val_acc)
                    if receive_message_as_json(client_socket).message_type == MessageType.VALIDATION_RECEIVED:
                        print(f"Client {client_name} received validation accuracy.")
                else:
                    print(f"Client {client_name} failed to validate model for epoch {global_epoch}.")
                    send_message_as_json(client_socket, MessageType.INVALID_MESSAGE, "Server", client_name, "Validation failed.")

            elif msg.message_type == MessageType.REQUEST_FOR_AGGREGATED_MODEL:
                epoch = int(msg.content)
                with epoch_count_lock:
                    if CLIENT_EPOCH_COUNT[epoch] == TOTAL_CLIENTS:
                        send_fed_avg_model_to_clients(client_socket, epoch)
                    else:
                        send_message_as_json(client_socket, MessageType.INVALID_MESSAGE, "Server", client_name, "Not all clients have validated their models.")
            else:
                print("Invalid message 2: ", msg.message_type, msg.sender, msg.receiver, msg.content)
                send_message_as_json(client_socket, MessageType.INVALID_MESSAGE)
        except ConnectionResetError:
            print(f"Client {client_name} forcibly disconnected.")
            break

def send_fed_avg_model_to_clients(client_socket, epoch, send = True):
    global clients, clients_lock, AGGREGATION_DONE
    if not AGGREGATION_DONE[epoch]:
        fed_avg_model = get_fed_avg_model(epoch)
        torch.save(fed_avg_model.state_dict(), f"shared_files/server/fed_avg_model_{epoch}.pt")
        AGGREGATION_DONE[epoch] = True
    if send:
        send_file(client_socket, "shared_files/server", f"fed_avg_model_{epoch}.pt", MessageType.SEND_AGGREGATED_MODEL)

def get_fed_avg_model(epoch):
    global device, CLIENT_WEIGHT, CLIENT_MODEL_PATHS, TOTAL_CLIENTS
    fed_avg_client_model = ClientModel().to(device)
    for i in range(TOTAL_CLIENTS):
        temp_client_model = ClientModel().to(device)
        temp_client_model.load_state_dict(torch.load(CLIENT_MODEL_PATHS[i][epoch]))
        for fed_avg_param, client_param in zip(fed_avg_client_model.parameters(), temp_client_model.parameters()):
            fed_avg_param.data += client_param.data * CLIENT_WEIGHT[i]
    return fed_avg_client_model

def validate_model(client_model_path, server_model, validation_loader_path):
    try:
        temp_client_model = ClientModel().to(device)
        temp_client_model.load_state_dict(torch.load(client_model_path))

        temp_server_model = ServerModel().to(device)
        temp_server_model.load_state_dict(server_model.state_dict())

        validation_loader = torch.load(validation_loader_path)
        validation_accuracy = validate_split_model(temp_client_model, temp_server_model, validation_loader)
        return validation_accuracy
    except Exception as e:
        print(f"Error: {e}")
        return -1

def test(client_model_path, server_model):
    final_client_model = ClientModel().to(device)
    final_client_model.load_state_dict(torch.load(client_model_path))

    accuracy = test_split_model(final_client_model, server_model)
#     write to log file
    with open(f"shared_files/server/{server_log_file}", "a") as log_file:
        log_file.write(f"{client_model_path},test_acc,{accuracy}\n")

def client_message_handler(client_socket, client_name):
    global clients, clients_lock
    while True:
        try:
            message = receive_message_as_json(client_socket)
            if message:
                if message.message_type == MessageType.SEND:
                    send_to_client(message.receiver, message.content)
                elif message.message_type == MessageType.UNREGISTER:
                    unregister_client(client_name)
                    break
                elif message.message_type == MessageType.REQUEST_TO_START:
                    send_message_as_json(client_socket, MessageType.START_TRAINING)
                    start_training(client_socket, message.sender)
            else:
                print(f"{client_name} has forcibly disconnected.")
                break
        except ConnectionResetError:
            print(f"{client_name} has forcibly disconnected.")
            break
    with clients_lock:
        if client_name in clients:
            del clients[client_name]
    client_socket.close()
    print(f"{client_name} disconnected.")

def send_to_client(client_name, message):
    global clients, clients_lock

    with clients_lock:
        if client_name in clients:
            send_message_as_json(
                client_socket=clients[client_name],
                message_type=MessageType.MESSAGE,
                sender="Server",
                receiver=client_name,
                content=message
            )
        else:
            print(f"Client {client_name} not found.")


def unregister_client(client_name):
    global clients, clients_lock
    with clients_lock:
        if client_name in clients:
            del clients[client_name]
            print(f"{client_name} disconnected.")
        else:
            print(f"Client {client_name} not found.")


def unregister_all_clients():
    global clients, clients_lock
    with clients_lock:
        for client_name, client_socket in clients.items():
            client_socket.close()
        clients = {}
    print("All clients disconnected.")


def register_client(client_socket, client_address):
    global clients, clients_lock, specific_client_file_directory_path, CLIENT_COUNT

    print(f"Connection from {client_address}")

    send_message_as_json(client_socket, MessageType.REGISTER)
    client_name = receive_message_as_json(client_socket).content

    while True:
        if "exit" in client_name:
            return False
        with clients_lock:
            if client_name not in clients:
                clients[client_name] = client_socket
                break
            else:
                send_message_as_json(client_socket, MessageType.REGISTER)
                client_name = receive_message_as_json(client_socket).content

    print(f"{client_name} connected.")
    send_message_as_json(client_socket, MessageType.REGISTERED, content=CLIENT_COUNT)
    clients_nos[client_name] = CLIENT_COUNT
    CLIENT_COUNT += 1
    specific_client_file_directory_path = f"{client_file_directory_path}/{client_name}"
    if not os.path.exists(specific_client_file_directory_path):
        os.makedirs(specific_client_file_directory_path)

    threading.Thread(target=client_message_handler, args=(client_socket, client_name), daemon=True).start()

    return True

def start_server(host=SERVER_ADDRESS, port=SERVER_PORT):
    global server_socket, clients, clients_lock
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen()
    print(f"Server is listening on {host}:{port}...")

    while True:
        try:
            client_socket, client_address = server_socket.accept()
            ret = register_client(client_socket, client_address)
            if not ret:
                break

        except KeyboardInterrupt:
            print("\nServer shutting down...")
            break
        except Exception as e:
            print(f"Error: {e}")

    unregister_all_clients()
    server_socket.close()
    print("Server shut down.")


def input_command_to_send():
    global clients, clients_lock, server_socket
    while True:
        try:
            command = input("\nEnter command: ")
            if command == "exit":
                break
            elif command.startswith("send"):
                command_parts = command.split(" ")
                if len(command_parts) >= 3:
                    client_name = command_parts[1]
                    message = " ".join(command_parts[2:])
                    send_to_client(client_name, message)
                else:
                    print("Invalid command.")
            else:
                print("Invalid command.")
        except KeyboardInterrupt:
            print("\nServer shutting down...")
            break

    with clients_lock:
        for client_name, client_socket in clients.items():
            client_socket.close()

    temp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    temp_socket.connect((SERVER_ADDRESS, SERVER_PORT))
    _ = temp_socket.recv(1024).decode()
    send_message_as_json(temp_socket, MessageType.UNREGISTER, "exit", "exit")
    temp_socket.close()


# Run the server
if __name__ == "__main__":
    start_server()


