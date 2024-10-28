# This file was auto-generated by Fern from our API Definition.

from .chat_stream_event_type import ChatStreamEventType
import typing
from ..core.pydantic_utilities import IS_PYDANTIC_V2
import pydantic


class ChatToolCallEndEvent(ChatStreamEventType):
    """
    A streamed event delta which signifies a tool call has finished streaming.
    """

    index: typing.Optional[int] = None

    if IS_PYDANTIC_V2:
        model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(extra="allow")  # type: ignore # Pydantic v2
    else:

        class Config:
            smart_union = True
            extra = pydantic.Extra.allow
