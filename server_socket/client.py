import socket
import threading
from common import *


def receive_message(client_socket):
    while True:
        try:
            message = client_socket.recv(1024).decode()
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

def start_client():
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((SERVER_ADDRESS, SERVER_PORT))
    print("Connected to server.")
    threading.Thread(target=receive_message, args=(client_socket,), daemon=True).start()

    while True:
        message = input()
        if message:
            client_socket.sendall(message.encode())
        else:
            break

    client_socket.close()


if __name__ == "__main__":
    start_client()
