# This file was auto-generated by Fern from our API Definition.

from ..core.unchecked_base_model import UncheckedBaseModel
import typing
import pydantic
from .api_meta import ApiMeta
from ..core.pydantic_utilities import IS_PYDANTIC_V2


class EmbedFloatsResponse(UncheckedBaseModel):
    id: str
    embeddings: typing.List[typing.List[float]] = pydantic.Field()
    """
    An array of embeddings, where each embedding is an array of floats. The length of the `embeddings` array will be the same as the length of the original `texts` array.
    """

    texts: typing.List[str] = pydantic.Field()
    """
    The text entries for which embeddings were returned.
    """

    meta: typing.Optional[ApiMeta] = None

    if IS_PYDANTIC_V2:
        model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(extra="allow", frozen=True)  # type: ignore # Pydantic v2
    else:

        class Config:
            frozen = True
            smart_union = True
            extra = pydantic.Extra.allow
