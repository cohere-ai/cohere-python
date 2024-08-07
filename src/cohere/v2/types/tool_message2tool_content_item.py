# This file was auto-generated by Fern from our API Definition.

from __future__ import annotations

import typing

import pydantic
import typing_extensions

from ...core.pydantic_utilities import IS_PYDANTIC_V2
from ...core.unchecked_base_model import UncheckedBaseModel, UnionMetadata


class ToolMessage2ToolContentItem_ToolResultObject(UncheckedBaseModel):
    output: typing.Dict[str, typing.Any]
    type: typing.Literal["tool_result_object"] = "tool_result_object"

    if IS_PYDANTIC_V2:
        model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(extra="allow", frozen=True)  # type: ignore # Pydantic v2
    else:

        class Config:
            frozen = True
            smart_union = True
            extra = pydantic.Extra.allow


ToolMessage2ToolContentItem = typing_extensions.Annotated[
    ToolMessage2ToolContentItem_ToolResultObject, UnionMetadata(discriminant="type")
]
