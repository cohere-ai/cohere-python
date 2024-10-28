# This file was auto-generated by Fern from our API Definition.

from ..core.unchecked_base_model import UncheckedBaseModel
import typing
from .chat_content_delta_event_delta_message import ChatContentDeltaEventDeltaMessage
from ..core.pydantic_utilities import IS_PYDANTIC_V2
import pydantic


class ChatContentDeltaEventDelta(UncheckedBaseModel):
    message: typing.Optional[ChatContentDeltaEventDeltaMessage] = None

    if IS_PYDANTIC_V2:
        model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(extra="allow")  # type: ignore # Pydantic v2
    else:

        class Config:
            smart_union = True
            extra = pydantic.Extra.allow
