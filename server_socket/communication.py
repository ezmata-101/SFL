from enum import Enum
import json
import os

class MessageType(Enum):
    REGISTER = 1
    REGISTERED = 2
    UNREGISTER = 3
    SEND = 4
    MESSAGE = 5
    REQUEST_TO_START = 6
    START_TRAINING = 7
    TRAINING_LABELS = 8
    TRAINING_FEATURES = 9
    TRAINING_GRADS = 10
    REQUEST_TO_SEND_FILE = 11
    SEND_FILE = 12



class Message:
    def __init__(self, message_type, sender, receiver, content):
        self.message_type = message_type
        self.sender = sender
        self.receiver = receiver
        self.content = content

    def to_string(self):
        return f"{self.message_type.value} {self.sender} {self.receiver} {self.content}"

    def register(self):
        return Message(MessageType.REGISTER, self.sender, self.receiver, self.content)

    def unregister(self):
        return Message(MessageType.UNREGISTER, self.sender, self.receiver, self.content)

    def to_dict(self):
        return {
            "message_type": self.message_type.value,
            "sender": self.sender,
            "receiver": self.receiver,
            "content": self.content
        }

    def from_dict(self, message_dict):
        self.message_type = MessageType(message_dict["message_type"])
        self.sender = message_dict["sender"]
        self.receiver = message_dict["receiver"]
        self.content = message_dict["content"]
        return self

    def __str__(self):
        return f"{self.message_type} {self.sender} {self.receiver} {self.content}"



def send_message_as_json(client_socket, message_type, sender="", receiver = "", content = ""):
    message = Message(message_type, sender, receiver, content)
    client_socket.sendall(json.dumps(message.to_dict()).encode())

def receive_message_as_json(client_socket):
    message = client_socket.recv(1024)
    try:
        message_dict = json.loads(message.decode('utf-8'))
    except UnicodeDecodeError:
        return None
    message = Message(MessageType(message_dict["message_type"]), message_dict["sender"], message_dict["receiver"], message_dict["content"])
    return message

def send_file(client_socket, directory_path, file_path):
    send_message_as_json(client_socket, MessageType.REQUEST_TO_SEND_FILE, "", "", file_path)
    if receive_message_as_json(client_socket).message_type == MessageType.SEND_FILE:
        file_size  = os.path.getsize(f"{directory_path}/{file_path}")

        client_socket.sendall(str(file_size).encode())
        client_socket.recv(4096)

        with open(f"{directory_path}/{file_path}", "rb") as file:
            while True:
                data = file.read(4096)
                if not data:
                    break
                client_socket.sendall(data)
    if receive_message_as_json(client_socket).content == "File received":
        return True

def receive_file(client_socket, directory_path, file_path):
    send_message_as_json(client_socket, MessageType.SEND_FILE, "", "", file_path)

    file_size = int(client_socket.recv(4096).decode())
    client_socket.sendall("File size received".encode())

    received = 0
    with open(f"{directory_path}/{file_path}", "wb") as file:
        while True:
            data = client_socket.recv(4096)
            received += len(data)
            file.write(data)
            if received >= file_size:
                break
    send_message_as_json(client_socket, MessageType.SEND_FILE, "", "", "File received")
    return True



