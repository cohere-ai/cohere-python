# This file was auto-generated by Fern from our API Definition.

from ..v2.types.chat_stream_event_type import ChatStreamEventType
import typing
from .citation_start_event_delta import CitationStartEventDelta
from ..core.pydantic_utilities import IS_PYDANTIC_V2
import pydantic


class CitationStartEvent(ChatStreamEventType):
    """
    A streamed event which signifies a citation has been created.
    """

    index: typing.Optional[int] = None
    delta: typing.Optional[CitationStartEventDelta] = None

    if IS_PYDANTIC_V2:
        model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(extra="allow", frozen=True)  # type: ignore # Pydantic v2
    else:

        class Config:
            frozen = True
            smart_union = True
            extra = pydantic.Extra.allow
