# This file was auto-generated by Fern from our API Definition.

import typing

import pydantic

from ..core.pydantic_utilities import IS_PYDANTIC_V2
from ..core.unchecked_base_model import UncheckedBaseModel
from .finetune_dataset_metrics import FinetuneDatasetMetrics
from .metrics_embed_data import MetricsEmbedData


class Metrics(UncheckedBaseModel):
    finetune_dataset_metrics: typing.Optional[FinetuneDatasetMetrics] = None
    embed_data: typing.Optional[MetricsEmbedData] = None

    if IS_PYDANTIC_V2:
        model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(extra="allow", frozen=True)  # type: ignore # Pydantic v2
    else:

        class Config:
            frozen = True
            smart_union = True
            extra = pydantic.Extra.allow
