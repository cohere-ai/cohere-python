# This file was auto-generated by Fern from our API Definition.

from ..core.unchecked_base_model import UncheckedBaseModel
import typing
from .tool_call_v2 import ToolCallV2
import pydantic
from .assistant_message_content import AssistantMessageContent
from .citation import Citation
from ..core.pydantic_utilities import IS_PYDANTIC_V2


class AssistantMessage(UncheckedBaseModel):
    """
    A message from the assistant role can contain text and tool call information.
    """

    tool_calls: typing.Optional[typing.List[ToolCallV2]] = None
    tool_plan: typing.Optional[str] = pydantic.Field(default=None)
    """
    A chain-of-thought style reflection and plan that the model generates when working with Tools.
    """

    content: typing.Optional[AssistantMessageContent] = None
    citations: typing.Optional[typing.List[Citation]] = None

    if IS_PYDANTIC_V2:
        model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(extra="allow", frozen=True)  # type: ignore # Pydantic v2
    else:

        class Config:
            frozen = True
            smart_union = True
            extra = pydantic.Extra.allow
