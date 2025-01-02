import socket
import threading


from common import *


clients = {}
clients_lock = threading.Lock()
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

def client_message_handler(client_socket, client_name):
    global clients, clients_lock
    while True:
        try:
            message = client_socket.recv(1024).decode()
            if message:
                print(f"Message from {client_name}: {message}")
                client_socket.sendall(b"Message received by server.")

                if "unregister" in message:
                    unregister_client(client_name)
                    break

                if message.startswith("send"):
                    message_parts = message.split(" ")
                    try:
                        receiver = message_parts[1]
                        message = "".join(message_parts[2:])
                        send_to_client(receiver, f"Message from {client_name}: {message}")
                    except IndexError:
                        print("Invalid Send command.")


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
            client_socket = clients[client_name]
            client_socket.sendall(message.encode())
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
    global clients, clients_lock
    client_socket.sendall(b"Enter your name: ")
    client_name = client_socket.recv(1024).decode()

    while True:
        if "exit" in client_name:
            return False
        with clients_lock:
            if client_name in clients:
                client_socket.sendall(b"Name already taken. Enter a different name: ")
                client_name = client_socket.recv(1024).decode()
            else:
                clients[client_name] = client_socket
                print(f"{client_name} connected.")
                break

    print(f"{client_name} connected.")
    client_socket.sendall(b"Welcome to the server!")

    threading.Thread(target=client_message_handler, args=(client_socket, client_name), daemon=True).start()

    return True

def start_server(host=SERVER_ADDRESS, port=SERVER_PORT):
    global server_socket, clients, clients_lock
    server_socket.bind((host, port))
    server_socket.listen(5)
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
    _ = temp_socket.sendall(b"exit")
    temp_socket.close()


# Run the server
if __name__ == "__main__":
    server_thread = threading.Thread(target=start_server)
    server_thread.start()

    input_command_to_send()


