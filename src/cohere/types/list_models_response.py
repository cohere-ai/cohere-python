# This file was auto-generated by Fern from our API Definition.

import typing

import pydantic

from ..core.pydantic_utilities import IS_PYDANTIC_V2
from ..core.unchecked_base_model import UncheckedBaseModel
from .get_model_response import GetModelResponse


class ListModelsResponse(UncheckedBaseModel):
    models: typing.List[GetModelResponse]
    next_page_token: typing.Optional[str] = pydantic.Field(default=None)
    """
    A token to retrieve the next page of results. Provide in the page_token parameter of the next request.
    """

    if IS_PYDANTIC_V2:
        model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(extra="allow", frozen=True)  # type: ignore # Pydantic v2
    else:

        class Config:
            frozen = True
            smart_union = True
            extra = pydantic.Extra.allow
