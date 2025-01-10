import socket
import threading
import os
import torch
import torch.nn as nn
import torch.optim as optim

from SplitFed.main_split_full import *

from common import *
from communication import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

clients = {}
clients_lock = threading.Lock()
server_socket = None

server_file_directory_path = "shared_files/server/server_files"
client_file_directory_path = "shared_files/server/client_files"
specific_client_file_directory_path = None

CLIENT_COUNT = 0

def start_training(client_socket, client_name):
    global specific_client_file_directory_path
    print(f"Client {client_name} training started.")

    server_model = ServerModel().to(device)
    server_optimizer = optim.Adam(server_model.parameters(), lr=0.001)
    loss_criteria = nn.CrossEntropyLoss()

    while True:
        try:
            msg = receive_message_as_json(client_socket)
            print(msg)
            if msg.message_type == MessageType.REQUEST_TO_SEND_FILE:
                labels_path = msg.content
                receive_file(client_socket, specific_client_file_directory_path, labels_path)
                slt_message = receive_message_as_json(client_socket)
                print(slt_message)
                if slt_message.message_type == MessageType.REQUEST_TO_SEND_FILE:
                    slt_path = slt_message.content
                    receive_file(client_socket, specific_client_file_directory_path, slt_path)
                    labels = torch.load(f"{specific_client_file_directory_path}/{labels_path}")
                    split_layer_tensor = torch.load(f"{specific_client_file_directory_path}/{slt_path}")

                    server_optimizer.zero_grad()
                    server_output = server_model(split_layer_tensor)
                    loss = loss_criteria(server_output, labels)
                    loss.backward()
                    server_optimizer.step()
                    split_layer_tensor.retain_grad()
                    torch.save(split_layer_tensor.grad, f"{specific_client_file_directory_path}/grads_{labels_path}")

                    send_file(client_socket, specific_client_file_directory_path, f"grads_{labels_path}")
                else:
                    print("Invalid message 1.")
            else:
                print("Invalid message 2.")
        except ConnectionResetError:
            print(f"Client {client_name} forcibly disconnected.")
            break


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
    CLIENT_COUNT += 1
    send_message_as_json(client_socket, MessageType.REGISTERED, content=CLIENT_COUNT)
    specific_client_file_directory_path = f"{client_file_directory_path}/{client_name}"
    if not os.path.exists(specific_client_file_directory_path):
        os.makedirs(specific_client_file_directory_path)

    threading.Thread(target=client_message_handler, args=(client_socket, client_name), daemon=True).start()

    # client_message_handler(client_socket, client_name)

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
    # server_thread = threading.Thread(target=start_server)
    # server_thread.start()

    start_server()

    # input_command_to_send()


