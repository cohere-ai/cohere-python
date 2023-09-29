from dataclasses import dataclass
from enum import Enum

from cohere.responses.base import CohereObject


class Role(str, Enum):
    CHATBOT = "chatbot"
    USER = "user"

    def __str__(self):
        return self.value


@dataclass
class ChatHistoryEntry(CohereObject):
    message: str
    role: Role
