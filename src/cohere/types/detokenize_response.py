# This file was auto-generated by Fern from our API Definition.

from ..core.unchecked_base_model import UncheckedBaseModel
import pydantic
import typing
from .api_meta import ApiMeta
from ..core.pydantic_utilities import IS_PYDANTIC_V2


class DetokenizeResponse(UncheckedBaseModel):
    text: str = pydantic.Field()
    """
    A string representing the list of tokens.
    """

    meta: typing.Optional[ApiMeta] = None

    if IS_PYDANTIC_V2:
        model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(extra="allow")  # type: ignore # Pydantic v2
    else:

        class Config:
            smart_union = True
            extra = pydantic.Extra.allow
