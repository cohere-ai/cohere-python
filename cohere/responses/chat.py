from concurrent.futures import Future
from typing import Any, Dict, Optional
from cohere.responses.base import CohereObject


class Chat(CohereObject):

    def __init__(self,
                 query: str,
                 persona: str,
                 return_chatlog: bool = False,
                 response: Optional[Dict[str, Any]] = None,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.query = query
        self.persona = persona
        assert response is not None
        self.reply = self._reply(response)
        self.session_id = self._session_id(response)

        if return_chatlog:
            self.chatlog = self._chatlog(response)

    def _reply(self, response: Dict[str, Any]) -> str:
        return response['reply']

    def _session_id(self, response: Dict[str, Any]) -> str:
        return response['session_id']

    def _chatlog(self, response: Dict[str, Any]) -> str:
        return response['chatlog']

