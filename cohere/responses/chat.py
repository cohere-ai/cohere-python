import json
from typing import Any, Dict, Generator, List, Optional

import requests

from cohere.responses.base import CohereObject


class Chat(CohereObject):
    def __init__(
        self,
        response_id: str,
        generation_id: str,
        message: str,
        text: str,
        conversation_id: str,
        meta: Optional[Dict[str, Any]] = None,
        prompt: Optional[str] = None,
        chatlog: Optional[List[Dict[str, str]]] = None,
        preamble: Optional[str] = None,
        token_count: Optional[Dict[str, int]] = None,
        client=None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.response_id = response_id
        self.generation_id = generation_id
        self.message = message
        self.text = text
        self.conversation_id = conversation_id  # optional
        self.prompt = prompt  # optional
        self.chatlog = chatlog  # optional
        self.preamble = preamble  # optional
        self.client = client
        self.token_count = token_count
        self.meta = meta

    @classmethod
    def from_dict(cls, response: Dict[str, Any], message: str, client) -> "Chat":
        return cls(
            id=response["response_id"],
            response_id=response["response_id"],
            generation_id=response["generation_id"],
            message=message,
            conversation_id=response.get("conversation_id"),  # optional
            text=response.get("text"),
            prompt=response.get("prompt"),  # optional
            chatlog=response.get("chatlog"),  # optional
            preamble=response.get("preamble"),  # optional
            client=client,
            token_count=response.get("token_count"),
            meta=response.get("meta"),
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


class StreamResponse(CohereObject):
    def __init__(
        self,
        is_finished: bool,
        index: Optional[int],
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.is_finished = is_finished
        self.index = index


class StreamStart(StreamResponse):
    def __init__(
        self,
        generation_id: str,
        conversation_id: Optional[str],
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.generation_id = generation_id
        self.conversation_id = conversation_id


class StreamTextGeneration(StreamResponse):
    def __init__(
        self,
        text: str,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.text = text


class StreamingChat(CohereObject):
    def __init__(self, response):
        self.response = response
        self.texts = []
        self.response_id = None
        self.conversation_id = None
        self.generation_id = None
        self.preamble = None
        self.prompt = None
        self.chatlog = None
        self.finish_reason = None
        self.token_count = None
        self.meta = None

    def _make_response_item(self, index, line) -> Any:
        streaming_item = json.loads(line)
        event_type = streaming_item.get("event_type")

        if event_type == "stream-start":
            self.conversation_id = streaming_item.get("conversation_id")
            self.generation_id = streaming_item.get("generation_id")
            return StreamStart(
                conversation_id=self.conversation_id, generation_id=self.generation_id, is_finished=False, index=index
            )
        elif event_type == "text-generation":
            text = streaming_item.get("text")
            return StreamTextGeneration(text=text, is_finished=False, index=index)
        elif event_type == "stream-end":
            response = streaming_item.get("response")
            self.finish_reason = streaming_item.get("finish_reason")

            if response is None:
                return None

            self.response_id = response.get("response_id")
            self.conversation_id = response.get("conversation_id")
            self.texts = [response.get("text")]
            self.generation_id = response.get("generation_id")
            self.preamble = response.get("preamble")
            self.prompt = response.get("prompt")
            self.chatlog = response.get("chatlog")
            self.token_count = response.get("token_count")
            self.meta = response.get("meta")
            return None
        return None

    def __iter__(self) -> Generator[StreamResponse, None, None]:
        if not isinstance(self.response, requests.Response):
            raise ValueError("For AsyncClient, use `async for` to iterate through the `StreamingChat`")

        for index, line in enumerate(self.response.iter_lines()):
            item = self._make_response_item(index, line)
            if item is not None:
                yield item

    async def __aiter__(self) -> Generator[StreamResponse, None, None]:
        index = 0
        async for line in self.response.content:
            item = self._make_response_item(index, line)
            index += 1
            if item is not None:
                yield item
