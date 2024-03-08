import json
from enum import Enum
from typing import Any, Dict, Generator, List, Optional, Union

import requests

from cohere.responses.base import CohereObject


class ToolParameterDefinitionsValue(CohereObject, dict):
    def __init__(
        self,
        type: str,
        description: str,
        required: Optional[bool] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.__dict__ = self
        self.type = type
        self.description = description
        if required is not None:
            self.required = required


class Tool(CohereObject, dict):
    def __init__(
        self,
        name: str,
        description: str,
        parameter_definitions: Optional[Dict[str, ToolParameterDefinitionsValue]] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.__dict__ = self
        self.name = name
        self.description = description
        if parameter_definitions is not None:
            self.parameter_definitions = parameter_definitions


class ToolCall(CohereObject, dict):
    def __init__(
        self,
        name: str,
        parameters: Dict[str, Any],
        generation_id: str,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.__dict__ = self
        self.name = name
        self.parameters = parameters
        self.generation_id = generation_id

    @classmethod
    def from_dict(cls, tool_call_res: Dict[str, Any]) -> "ToolCall":
        return cls(
            name=tool_call_res.get("name"),
            parameters=tool_call_res.get("parameters"),
            generation_id=tool_call_res.get("generation_id"),
        )

    @classmethod
    def from_list(cls, tool_calls_res: Optional[List[Dict[str, Any]]]) -> Optional[List["ToolCall"]]:
        if tool_calls_res is None or not isinstance(tool_calls_res, list):
            return None

        return [ToolCall.from_dict(tc) for tc in tool_calls_res]


class ChatRequestToolResultsItem(CohereObject, dict):
    def __init__(
        self,
        call: ToolCall,
        outputs: List[Dict[str, Any]],
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.__dict__ = self
        self.call = call
        self.outputs = outputs


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
        chat_history: Optional[List[Dict[str, Any]]] = None,
        preamble: Optional[str] = None,
        token_count: Optional[Dict[str, int]] = None,
        is_search_required: Optional[bool] = None,
        citations: Optional[List[Dict[str, Any]]] = None,
        documents: Optional[List[Dict[str, Any]]] = None,
        search_results: Optional[List[Dict[str, Any]]] = None,
        search_queries: Optional[List[Dict[str, Any]]] = None,
        finish_reason: Optional[str] = None,
        tool_calls: Optional[List[ToolCall]] = None,
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
        self.chat_history = chat_history  # optional
        self.preamble = preamble  # optional
        self.client = client
        self.token_count = token_count
        self.meta = meta
        self.is_search_required = is_search_required  # optional
        self.citations = citations  # optional
        self.documents = documents  # optional
        self.search_results = search_results  # optional
        self.search_queries = search_queries  # optional
        self.tool_calls = tool_calls  # optional
        self.finish_reason = finish_reason

    @classmethod
    def from_dict(cls, response: Dict[str, Any], message: str, client) -> "Chat":
        return cls(
            id=response["response_id"],
            response_id=response["response_id"],
            generation_id=response.get("generation_id"),  # optional
            message=message,
            conversation_id=response.get("conversation_id"),  # optional
            text=response.get("text"),
            prompt=response.get("prompt"),  # optional
            chat_history=response.get("chat_history"),  # optional
            preamble=response.get("preamble"),  # optional
            client=client,
            token_count=response.get("token_count"),
            meta=response.get("meta"),
            is_search_required=response.get("is_search_required"),  # optional
            citations=response.get("citations"),  # optional
            documents=response.get("documents"),  # optional
            search_results=response.get("search_results"),  # optional
            search_queries=response.get("search_queries"),  # optional
            finish_reason=response.get("finish_reason"),
            tool_calls=ToolCall.from_list(response.get("tool_calls")),  # optional
        )

    def respond(self, response: str, max_tokens: int = None) -> "Chat":
        return self.client.chat(
            message=response,
            conversation_id=self.conversation_id,
            return_chat_history=self.chat_history is not None,
            return_prompt=self.prompt is not None,
            return_preamble=self.preamble is not None,
            max_tokens=max_tokens,
        )


class AsyncChat(Chat):
    async def respond(self, response: str, max_tokens: int = None) -> "AsyncChat":
        return await self.client.chat(
            message=response,
            conversation_id=self.conversation_id,
            return_chat_history=self.chat_history is not None,
            return_prompt=self.prompt is not None,
            return_preamble=self.preamble is not None,
            max_tokens=max_tokens,
        )


class StreamEvent(str, Enum):
    STREAM_START = "stream-start"
    SEARCH_QUERIES_GENERATION = "search-queries-generation"
    SEARCH_RESULTS = "search-results"
    TEXT_GENERATION = "text-generation"
    TOOL_CALLS_GENERATION = "tool-calls-generation"
    CITATION_GENERATION = "citation-generation"
    STREAM_END = "stream-end"


class StreamResponse(CohereObject):
    def __init__(
        self,
        is_finished: bool,
        event_type: Union[StreamEvent, str],
        index: Optional[int],
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.is_finished = is_finished
        self.index = index
        self.event_type = event_type


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


class StreamCitationGeneration(StreamResponse):
    def __init__(
        self,
        citations: Optional[List[Dict[str, Any]]],
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.citations = citations


class StreamQueryGeneration(StreamResponse):
    def __init__(
        self,
        search_queries: Optional[List[Dict[str, Any]]],
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.search_queries = search_queries


class StreamSearchResults(StreamResponse):
    def __init__(
        self,
        search_results: Optional[List[Dict[str, Any]]],
        documents: Optional[List[Dict[str, Any]]],
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.search_results = search_results
        self.documents = documents


class StreamEnd(StreamResponse):
    def __init__(
        self,
        finish_reason: str,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.finish_reason = finish_reason


class ChatToolCallsGenerationEvent(StreamResponse):
    def __init__(
        self,
        tool_calls: Optional[List[ToolCall]],
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.tool_calls = tool_calls


class StreamingChat(CohereObject):
    def __init__(self, response):
        self.response = response
        self.texts = []
        self.response_id = None
        self.conversation_id = None
        self.generation_id = None
        self.preamble = None
        self.prompt = None
        self.chat_history = None
        self.finish_reason = None
        self.token_count = None
        self.meta = None
        self.is_search_required = None
        self.citations = None
        self.documents = None
        self.search_results = None
        self.search_queries = None
        self.tool_calls = None

    def _make_response_item(self, index, line) -> Any:
        streaming_item = json.loads(line)
        event_type = streaming_item.get("event_type")

        if event_type == StreamEvent.STREAM_START:
            self.conversation_id = streaming_item.get("conversation_id")
            self.generation_id = streaming_item.get("generation_id")
            return StreamStart(
                conversation_id=self.conversation_id,
                generation_id=self.generation_id,
                is_finished=False,
                event_type=event_type,
                index=index,
            )
        elif event_type == StreamEvent.SEARCH_QUERIES_GENERATION:
            search_queries = streaming_item.get("search_queries")
            return StreamQueryGeneration(
                search_queries=search_queries, is_finished=False, event_type=event_type, index=index
            )
        elif event_type == StreamEvent.SEARCH_RESULTS:
            search_results = streaming_item.get("search_results")
            documents = streaming_item.get("documents")
            return StreamSearchResults(
                search_results=search_results,
                documents=documents,
                is_finished=False,
                event_type=event_type,
                index=index,
            )
        elif event_type == StreamEvent.TEXT_GENERATION:
            text = streaming_item.get("text")
            return StreamTextGeneration(text=text, is_finished=False, event_type=event_type, index=index)
        elif event_type == StreamEvent.CITATION_GENERATION:
            citations = streaming_item.get("citations")
            return StreamCitationGeneration(citations=citations, is_finished=False, event_type=event_type, index=index)
        elif event_type == StreamEvent.TOOL_CALLS_GENERATION:
            tool_calls = ToolCall.from_list(streaming_item.get("tool_calls"))
            return ChatToolCallsGenerationEvent(
                tool_calls=tool_calls, is_finished=False, event_type=event_type, index=index
            )
        elif event_type == StreamEvent.STREAM_END:
            response = streaming_item.get("response")
            finish_reason = streaming_item.get("finish_reason")
            self.finish_reason = finish_reason

            if response is None:
                return None

            self.response_id = response.get("response_id")
            self.conversation_id = response.get("conversation_id")
            self.texts = [response.get("text")]
            self.generation_id = response.get("generation_id")
            self.preamble = response.get("preamble")
            self.prompt = response.get("prompt")
            self.chat_history = response.get("chat_history")
            self.token_count = response.get("token_count")
            self.meta = response.get("meta")
            self.is_search_required = response.get("is_search_required")  # optional
            self.citations = response.get("citations")  # optional
            self.documents = response.get("documents")  # optional
            self.search_results = response.get("search_results")  # optional
            self.search_queries = response.get("search_queries")  # optional
            self.tool_calls = ToolCall.from_list(response.get("tool_calls"))  # optional
            return StreamEnd(finish_reason=finish_reason, is_finished=True, event_type=event_type, index=index)
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
