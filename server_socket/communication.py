from enum import Enum
import json

class MessageType(Enum):
    REGISTER = 1
    REGISTERED = 2
    UNREGISTER = 3
    SEND = 4
    MESSAGE = 5


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
    message_dict = json.loads(client_socket.recv(1024).decode())
    message = Message(MessageType(message_dict["message_type"]), message_dict["sender"], message_dict["receiver"], message_dict["content"])
    return message
