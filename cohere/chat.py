from concurrent.futures import Future
from typing import Any, Dict, Optional
from cohere.response import AsyncAttribute, CohereObject


class Chat(CohereObject):

    def __init__(self,
                 query: str,
                 persona: str,
                 return_chatlog: bool = False,
                 response: Optional[Dict[str, Any]] = None,
                 *,
                 _future: Optional[Future] = None,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.query = query
        self.persona = persona

        if _future is not None:
            self._init_from_future(_future, return_chatlog)
        else:
            assert response is not None
            self.reply = self._reply(response)
            self.session_id = self._session_id(response)

            if return_chatlog:
                self.chatlog = self._chatlog

    def _init_from_future(self, future: Future, return_chatlog: bool):
        if return_chatlog:
            self.chatlog = AsyncAttribute(future, self._chatlog)

        self.reply = AsyncAttribute(future, self._reply)
        self.session_id = AsyncAttribute(future, self._session_id)

    def _reply(self, response: Dict[str, Any]) -> str:
        return response['reply']

    def _session_id(self, response: Dict[str, Any]) -> str:
        return response['session_id']

    def _chatlog(self, response: Dict[str, Any]) -> str:
        return response['chatlog']

    def respond(self, response: str) -> "Chat":
        return self.client.chat(query=response, session_id=self.session_id, persona=self.persona)
