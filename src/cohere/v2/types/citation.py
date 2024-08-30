# This file was auto-generated by Fern from our API Definition.

import typing

import pydantic

from ...core.pydantic_utilities import IS_PYDANTIC_V2
from ...core.unchecked_base_model import UncheckedBaseModel
from .source import Source


class Citation(UncheckedBaseModel):
    """
    Citation information containing sources and the text cited.
    """

    start: typing.Optional[int] = None
    end: typing.Optional[int] = None
    text: typing.Optional[str] = None
    sources: typing.Optional[typing.List[Source]] = None

    if IS_PYDANTIC_V2:
        model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(extra="allow", frozen=True)  # type: ignore # Pydantic v2
    else:

        class Config:
            frozen = True
            smart_union = True
            extra = pydantic.Extra.allow
