# This file was auto-generated by Fern from our API Definition.

from .chat_stream_event import ChatStreamEvent
import typing
from .chat_citation import ChatCitation
import pydantic
from ..core.pydantic_utilities import IS_PYDANTIC_V2


class ChatCitationGenerationEvent(ChatStreamEvent):
    citations: typing.List[ChatCitation] = pydantic.Field()
    """
    Citations for the generated reply.
    """

    if IS_PYDANTIC_V2:
        model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(extra="allow", frozen=True)  # type: ignore # Pydantic v2
    else:

        class Config:
            frozen = True
            smart_union = True
            extra = pydantic.Extra.allow
