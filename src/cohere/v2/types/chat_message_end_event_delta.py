# This file was auto-generated by Fern from our API Definition.

import typing

import pydantic

from ...core.pydantic_utilities import IS_PYDANTIC_V2
from ...core.unchecked_base_model import UncheckedBaseModel
from .chat_finish_reason import ChatFinishReason
from .usage import Usage


class ChatMessageEndEventDelta(UncheckedBaseModel):
    finish_reason: typing.Optional[ChatFinishReason] = None
    usage: typing.Optional[Usage] = None

    if IS_PYDANTIC_V2:
        model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(extra="allow", frozen=True)  # type: ignore # Pydantic v2
    else:

        class Config:
            frozen = True
            smart_union = True
            extra = pydantic.Extra.allow
