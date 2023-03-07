from typing import Any, Dict, Optional, List
from cohere.responses.base import CohereObject
from cohere.responses.meta_response import Meta


class Chat(CohereObject):

    def __init__(self,
                 query: str,
                 persona: str,
                 reply: str,
                 session_id: str,
                 meta: Optional[List[Meta]] = None,
                 prompt: Optional[str] = None,
                 chatlog: Optional[List[Dict[str, str]]] = None,
                 client=None,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.query = query
        self.persona = persona
        self.reply = reply
        self.session_id = session_id
        self.prompt = prompt # optional 
        self.chatlog = chatlog # optional 
        self.client = client
        self.meta = meta

    @classmethod
    def from_dict(cls,response: Dict[str, Any], query: str, persona: str,client) -> "Chat":
        return cls(
            query = query,
            persona=persona,
            session_id=response['session_id'],
            reply=response['reply'],
            prompt=response.get('prompt'), # optional 
            chatlog=response.get('chatlog'), # optional 
            client=client,
            meta=response['meta']
        )

    def respond(self, response: str) -> "Chat":
        return self.client.chat(query=response, session_id=self.session_id, persona=self.persona,
                                return_chatlog = self.chatlog is not None, return_prompt = self.prompt is not None)


class AsyncChat(Chat):
    async def respond(self, response: str) -> "AsyncChat":
        return await self.client.chat(query=response, session_id=self.session_id, persona=self.persona,
                                     return_chatlog = self.chatlog is not None, return_prompt = self.prompt is not None)

