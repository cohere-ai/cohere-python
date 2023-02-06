from concurrent.futures import Future
from typing import Any, Dict, Optional
from cohere.response import AsyncAttribute, CohereObject


class Chat(CohereObject):
    """
    A chat object.

    Attributes:
        query (str): The query text.
        persona (str): The persona name.
        reply (str): The reply text.
        session_id (str): The session ID.

    Methods:
        respond(response: str) -> Chat: Respond to the chat.

    Example:
        >>> chat = client.chat(query="Hello", persona="Alice")
        >>> chat.reply
        "Hello, how are you?"
        >>> chat.session_id
        "1234567890"
        >>> chat = chat.respond("I'm fine, thanks.")
        >>> chat.reply
        "That's good to hear."
        >>> chat.session_id
        "1234567890"
    """

    def __init__(self,
                 query: str,
                 persona: str,
                 response: Optional[Dict[str, Any]] = None,
                 *,
                 _future: Optional[Future] = None,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.query = query
        self.persona = persona

        if _future is not None:
            self._init_from_future(_future)
        else:
            assert response is not None
            self.reply = self._reply(response)
            self.session_id = self._session_id(response)

    def _init_from_future(self, future: Future):
        self.reply = AsyncAttribute(future, self._reply)
        self.session_id = AsyncAttribute(future, self._session_id)

    def _reply(self, response: Dict[str, Any]) -> str:
        return response['reply']

    def _session_id(self, response: Dict[str, Any]) -> str:
        return response['session_id']

    def respond(self, response: str) -> "Chat":
        return self.client.chat(query=response, session_id=self.session_id, persona=self.persona)
