# This file was auto-generated by Fern from our API Definition.

import typing

import pydantic

from ..core.pydantic_utilities import IS_PYDANTIC_V2
from ..core.unchecked_base_model import UncheckedBaseModel
from .metrics_embed_data_fields_item import MetricsEmbedDataFieldsItem


class MetricsEmbedData(UncheckedBaseModel):
    fields: typing.Optional[typing.List[MetricsEmbedDataFieldsItem]] = pydantic.Field(default=None)
    """
    the fields in the dataset
    """

    if IS_PYDANTIC_V2:
        model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(extra="allow", frozen=True)  # type: ignore # Pydantic v2
    else:

        class Config:
            frozen = True
            smart_union = True
            extra = pydantic.Extra.allow
