import socket
import uuid
import torch

from common import *
from communication import *

from SplitFed.main_split_full import client_train

client_socket = None
UNIQUE_CLIENT_NAME = str(uuid.uuid4()).split("-")[0]
CLIENT_NO = -1
print(f"Client name: {UNIQUE_CLIENT_NAME}")

server_file_directory_path = None
client_file_directory_path = None

def receive_message():
    global client_socket
    while True:
        try:
            message = receive_message_as_json(client_socket)
            if message:
                print(message)
            else:
                print("Server has forcibly disconnected.")
                break
        except ConnectionResetError:
            print("Server has forcibly disconnected.")
            break

    client_socket.close()
    print("Disconnected from server.")

def communicate_with_server_during_training(labels, split_layer_tensor, epoch, i):
    global client_socket

    torch.save(labels, f"{client_file_directory_path}/labels_{epoch}_{i}.pt")
    torch.save(split_layer_tensor, f"{client_file_directory_path}/split_layer_tensor_{epoch}_{i}.pt")

    send_file(client_socket, client_file_directory_path, f"labels_{epoch}_{i}.pt")
    send_file(client_socket, client_file_directory_path, f"split_layer_tensor_{epoch}_{i}.pt")

    grads_message = receive_message_as_json(client_socket)
    if grads_message.message_type == MessageType.REQUEST_TO_SEND_FILE:
        receive_file(client_socket, server_file_directory_path, f"grads_{epoch}_{i}.pt")
        grads = torch.load(f"{server_file_directory_path}/grads_{epoch}_{i}.pt")


        return grads

    return None

def communicate_with_fed_server(client_model, epoch):
    global client_socket
    if epoch == -1:
        torch.save(client_model.state_dict(), f"{client_file_directory_path}/client_model_final.pt")
        send_file(client_socket, client_file_directory_path, file_path=f"client_model_final.pt", message_type=MessageType.REQUEST_TO_SEND_FINAL_MODEL)
        return


def start_training():
    global server_file_directory_path, client_file_directory_path
    print("Starting training.")
    send_message_as_json(client_socket, MessageType.REQUEST_TO_START, UNIQUE_CLIENT_NAME, "")
    resp = receive_message_as_json(client_socket)
    if resp.message_type == MessageType.START_TRAINING:
        print("Training started.")
        client_train(communicate_with_server_during_training, communicate_with_fed_server)
        send_message_as_json(client_socket, MessageType.TRAINING_DONE, UNIQUE_CLIENT_NAME, "")
        if receive_message_as_json(client_socket).message_type == MessageType.TRAINING_DONE:
            print("Training done.")
    else:
        print("Failed to start training.")
        return

def register_client():
    global UNIQUE_CLIENT_NAME, server_file_directory_path, client_file_directory_path, CLIENT_NO, client_socket

    while True:
        try:
            response = receive_message_as_json(client_socket)
            if response.message_type == MessageType.REGISTER:
                UNIQUE_CLIENT_NAME = str(uuid.uuid4()).split("-")[0]
                send_message_as_json(client_socket, MessageType.REGISTER, content=UNIQUE_CLIENT_NAME, sender="", receiver="")

            elif response.message_type == MessageType.REGISTERED:
                CLIENT_NO = int(response.content)
                print(f"Registered to server as: {UNIQUE_CLIENT_NAME}, client_id: {response}, client_no: {CLIENT_NO}")
                server_file_directory_path = f"shared_files/client/{UNIQUE_CLIENT_NAME}/server_files"
                client_file_directory_path = f"shared_files/client/{UNIQUE_CLIENT_NAME}/client_files"

                if not os.path.exists(server_file_directory_path):
                    os.makedirs(server_file_directory_path)
                if not os.path.exists(client_file_directory_path):
                    os.makedirs(client_file_directory_path)

                start_training()

                break

            else:
                print("Failed to register to server.")
                break
        except ConnectionResetError:
            print("Server has forcibly disconnected.")
            exit(1)

    print("Registered to server as: ", UNIQUE_CLIENT_NAME)


def start_client():
    global client_socket
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((SERVER_ADDRESS, SERVER_PORT))
    print("Connected to server.")
    register_client()

    while True:
        message = input()
        if message:
            send_message_as_json(client_socket, MessageType.SEND, UNIQUE_CLIENT_NAME, "", message)
        else:
            break

    client_socket.close()


if __name__ == "__main__":
    start_client()
