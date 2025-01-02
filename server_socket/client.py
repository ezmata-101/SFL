import socket
import threading
import uuid
from common import *
from communication import *


UNIQUE_CLIENT_NAME = str(uuid.uuid4()).split("-")[0]
print(f"Client name: {UNIQUE_CLIENT_NAME}")

def receive_message(client_socket):
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

def register_client(client_socket):
    global UNIQUE_CLIENT_NAME

    while True:
        try:
            response = receive_message_as_json(client_socket)
            if response.message_type == MessageType.REGISTER:
                UNIQUE_CLIENT_NAME = str(uuid.uuid4()).split("-")[0]
                send_message_as_json(client_socket, MessageType.REGISTER, content=UNIQUE_CLIENT_NAME, sender="", receiver="")

            elif response.message_type == MessageType.REGISTERED:
                print("Registered to server.")
                break

            else:
                print("Failed to register to server.")
                break
        except ConnectionResetError:
            print("Server has forcibly disconnected.")
            exit(1)

    print("Registered to server as: ", UNIQUE_CLIENT_NAME)


def start_client():
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((SERVER_ADDRESS, SERVER_PORT))
    print("Connected to server.")
    register_client(client_socket)
    threading.Thread(target=receive_message, args=(client_socket,), daemon=True).start()

    while True:
        message = input()
        if message:
            send_message_as_json(client_socket, MessageType.SEND, UNIQUE_CLIENT_NAME, "", message)
        else:
            break

    client_socket.close()


if __name__ == "__main__":
    start_client()
