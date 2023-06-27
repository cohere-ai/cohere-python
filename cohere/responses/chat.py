import json
from enum import Enum
from typing import Any, Dict, Generator, List, NamedTuple, Optional

import requests

from cohere.responses.base import CohereObject


class Mode(str, Enum):
    CHAT = "chat"
    SEARCH_QUERY_GENERATION = "search_query_generation"
    AUGMENTED_GENERATION = "augmented_generation"


class Chat(CohereObject):
    def __init__(
        self,
        response_id: Optional[str],
        generation_id: Optional[str],
        message: Optional[str],
        text: Optional[str],
        conversation_id: Optional[str],
        meta: Optional[Dict[str, Any]] = None,
        prompt: Optional[str] = None,
        chatlog: Optional[List[Dict[str, str]]] = None,
        preamble: Optional[str] = None,
        token_count: Optional[Dict[str, int]] = None,
        client=None,
        is_search_required: Optional[bool] = None,
        queries: Optional[List[str]] = None,
        citations: Optional[List[Dict[str, str]]] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.response_id = response_id
        self.generation_id = generation_id
        self.query = message  # to be deprecated
        self.message = message
        self.text = text
        self.conversation_id = conversation_id
        self.prompt = prompt  # optional
        self.chatlog = chatlog  # optional
        self.preamble = preamble  # optional
        self.client = client
        self.token_count = token_count
        self.meta = meta
        self.queries = queries
        self.citations = citations
        self.is_search_required = is_search_required

    @classmethod
    def from_dict(cls, response: Dict[str, Any], message: str, client) -> "Chat":
        return cls(
            id=response.get("response_id"),
            response_id=response.get("response_id"),
            generation_id=response.get("generation_id"),
            message=message,
            conversation_id=response["conversation_id"],
            text=response.get("text"),
            prompt=response.get("prompt"),  # optional
            chatlog=response.get("chatlog"),  # optional
            preamble=response.get("preamble"),  # option
            client=client,
            token_count=response.get("token_count"),
            meta=response.get("meta"),
            queries=response.get("queries"),
            is_search_required=response.get("is_search_required"),
            citations=response.get("citations"),
        )

    def respond(self, response: str, max_tokens: int = None) -> "Chat":
        return self.client.chat(
            message=response,
            conversation_id=self.conversation_id,
            return_chatlog=self.chatlog is not None,
            return_prompt=self.prompt is not None,
            return_preamble=self.preamble is not None,
            max_tokens=max_tokens,
        )


class AsyncChat(Chat):
    async def respond(self, response: str, max_tokens: int = None) -> "AsyncChat":
        return await self.client.chat(
            message=response,
            conversation_id=self.conversation_id,
            return_chatlog=self.chatlog is not None,
            return_prompt=self.prompt is not None,
            return_preamble=self.preamble is not None,
            max_tokens=max_tokens,
        )


StreamingText = NamedTuple("StreamingText", [("index", Optional[int]), ("text", str), ("is_finished", bool)])


class StreamingChat(CohereObject):
    def __init__(self, response):
        self.response = response
        self.texts = []
        self.response_id = None
        self.conversation_id = None
        self.preamble = None
        self.prompt = None
        self.chatlog = None
        self.finish_reason = None

    def _make_response_item(self, index, line) -> Any:
        streaming_item = json.loads(line)
        is_finished = streaming_item.get("is_finished")
        text = streaming_item.get("text")

        if not is_finished:
            return StreamingText(text=text, is_finished=is_finished, index=index)

        response = streaming_item.get("response")

        if response is None:
            return None

        self.response_id = response.get("response_id")
        self.conversation_id = response.get("conversation_id")
        self.preamble = response.get("preamble")
        self.prompt = response.get("prompt")
        self.chatlog = response.get("chatlog")
        self.finish_reason = streaming_item.get("finish_reason")
        self.texts = [response.get("text")]
        return None

    def __iter__(self) -> Generator[StreamingText, None, None]:
        if not isinstance(self.response, requests.Response):
            raise ValueError("For AsyncClient, use `async for` to iterate through the `StreamingChat`")

        for index, line in enumerate(self.response.iter_lines()):
            item = self._make_response_item(index, line)
            if item is not None:
                yield item

    async def __aiter__(self) -> Generator[StreamingText, None, None]:
        index = 0
        async for line in self.response.content:
            item = self._make_response_item(index, line)
            index += 1
            if item is not None:
                yield item
