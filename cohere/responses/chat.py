import json
from typing import Any, Dict, Generator, List, NamedTuple, Optional

import requests

from cohere.responses.base import CohereObject


class Chat(CohereObject):
    def __init__(
        self,
        response_id: str,
        query: str,
        reply: str,
        conversation_id: str,
        meta: Optional[Dict[str, Any]] = None,
        prompt: Optional[str] = None,
        chatlog: Optional[List[Dict[str, str]]] = None,
        client=None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.response_id = response_id
        self.query = query
        self.reply = reply
        self.conversation_id = conversation_id
        self.prompt = prompt  # optional
        self.chatlog = chatlog  # optional
        self.client = client
        self.meta = meta

    @classmethod
    def from_dict(cls, response: Dict[str, Any], query: str, client) -> "Chat":
        return cls(
            id=response["response_id"],
            response_id=response["response_id"],
            query=query,
            conversation_id=response["conversation_id"],
            reply=response["reply"],
            prompt=response.get("prompt"),  # optional
            chatlog=response.get("chatlog"),  # optional
            client=client,
            meta=response.get("meta"),
        )

    def respond(self, response: str) -> "Chat":
        return self.client.chat(
            query=response,
            conversation_id=self.conversation_id,
            return_chatlog=self.chatlog is not None,
            return_prompt=self.prompt is not None,
        )


class AsyncChat(Chat):
    async def respond(self, response: str) -> "AsyncChat":
        return await self.client.chat(
            query=response,
            conversation_id=self.conversation_id,
            return_chatlog=self.chatlog is not None,
            return_prompt=self.prompt is not None,
        )


StreamingText = NamedTuple("StreamingText", [("index", Optional[int]), ("text", str)])


class StreamingChat(CohereObject):
    def __init__(self, response):
        self.response = response
        self.texts = []

    def _make_response_item(self, line) -> Any:
        streaming_item = json.loads(line)
        index = streaming_item.get("index", 0)
        text = streaming_item.get("text")

        while len(self.texts) <= index:
            self.texts.append("")

        if text is None:
            return None

        self.texts[index] += text
        return StreamingText(index=index, text=text)

    def __iter__(self) -> Generator[StreamingText, None, None]:
        if not isinstance(self.response, requests.Response):
            raise ValueError("For AsyncClient, use `async for` to iterate through the `StreamingChat`")

        for line in self.response.iter_lines():
            item = self._make_response_item(line)
            if item is not None:
                yield item

    async def __aiter__(self) -> Generator[StreamingText, None, None]:
        async for line in self.response.content:
            item = self._make_response_item(line)
            if item is not None:
                yield item
