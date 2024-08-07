# This file was auto-generated by Fern from our API Definition.

from __future__ import annotations

import typing

import pydantic
import typing_extensions

from ...core.pydantic_utilities import IS_PYDANTIC_V2
from ...core.unchecked_base_model import UncheckedBaseModel, UnionMetadata


class Source_Tool(UncheckedBaseModel):
    """
    A source object containing information about the source of the data cited.
    """

    id: typing.Optional[str] = None
    tool_output: typing.Optional[typing.Dict[str, typing.Any]] = None
    type: typing.Literal["tool"] = "tool"

    if IS_PYDANTIC_V2:
        model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(extra="allow", frozen=True)  # type: ignore # Pydantic v2
    else:

        class Config:
            frozen = True
            smart_union = True
            extra = pydantic.Extra.allow


class Source_Document(UncheckedBaseModel):
    """
    A source object containing information about the source of the data cited.
    """

    id: typing.Optional[str] = None
    document: typing.Optional[typing.Dict[str, typing.Any]] = None
    type: typing.Literal["document"] = "document"

    if IS_PYDANTIC_V2:
        model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(extra="allow", frozen=True)  # type: ignore # Pydantic v2
    else:

        class Config:
            frozen = True
            smart_union = True
            extra = pydantic.Extra.allow


Source = typing_extensions.Annotated[typing.Union[Source_Tool, Source_Document], UnionMetadata(discriminant="type")]
