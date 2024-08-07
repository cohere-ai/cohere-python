# This file was auto-generated by Fern from our API Definition.

import typing

import pydantic

from ...core.pydantic_utilities import IS_PYDANTIC_V2
from .chat_stream_event_type import ChatStreamEventType
from .chat_tool_call_delta_event_delta import ChatToolCallDeltaEventDelta


class ChatToolCallDeltaEvent(ChatStreamEventType):
    """
    A streamed event delta which signifies a delta in tool call arguments.
    """

    index: typing.Optional[int] = None
    delta: typing.Optional[ChatToolCallDeltaEventDelta] = None

    if IS_PYDANTIC_V2:
        model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(extra="allow", frozen=True)  # type: ignore # Pydantic v2
    else:

        class Config:
            frozen = True
            smart_union = True
            extra = pydantic.Extra.allow
