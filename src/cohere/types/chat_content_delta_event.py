# This file was auto-generated by Fern from our API Definition.

from .chat_stream_event_type import ChatStreamEventType
import typing
from .chat_content_delta_event_delta import ChatContentDeltaEventDelta
from .logprob_item import LogprobItem
from ..core.pydantic_utilities import IS_PYDANTIC_V2
import pydantic


class ChatContentDeltaEvent(ChatStreamEventType):
    """
    A streamed delta event which contains a delta of chat text content.
    """

    index: typing.Optional[int] = None
    delta: typing.Optional[ChatContentDeltaEventDelta] = None
    logprobs: typing.Optional[LogprobItem] = None

    if IS_PYDANTIC_V2:
        model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(extra="allow")  # type: ignore # Pydantic v2
    else:

        class Config:
            smart_union = True
            extra = pydantic.Extra.allow
