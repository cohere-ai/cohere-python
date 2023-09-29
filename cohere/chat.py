from enum import Enum

from cohere.responses.base import CohereObject


class Role(str, Enum):
    CHATBOT = "chatbot"
    USER = "user"


class ChatHistoryEntry(CohereObject):
    def __init__(self, message: str, role: Role):
        self.message = message
        self.role = role

    def __iter__(self):
        yield "message", self.message
        yield "role", self.role.value
