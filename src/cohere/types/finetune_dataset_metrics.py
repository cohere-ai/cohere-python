# This file was auto-generated by Fern from our API Definition.

import datetime as dt
import typing

from ..core.datetime_utils import serialize_datetime
from ..core.pydantic_utilities import pydantic_v1


class FinetuneDatasetMetrics(pydantic_v1.BaseModel):
    trainable_token_count: typing.Optional[str] = pydantic_v1.Field(alias="trainableTokenCount", default=None)
    """
    The number of tokens of valid examples that can be used for training.
    """

    total_examples: typing.Optional[str] = pydantic_v1.Field(alias="totalExamples", default=None)
    """
    The overall number of examples.
    """

    train_examples: typing.Optional[str] = pydantic_v1.Field(alias="trainExamples", default=None)
    """
    The number of training examples.
    """

    train_size_bytes: typing.Optional[str] = pydantic_v1.Field(alias="trainSizeBytes", default=None)
    """
    The size in bytes of all training examples.
    """

    eval_examples: typing.Optional[str] = pydantic_v1.Field(alias="evalExamples", default=None)
    """
    Number of evaluation examples.
    """

    eval_size_bytes: typing.Optional[str] = pydantic_v1.Field(alias="evalSizeBytes", default=None)
    """
    The size in bytes of all eval examples.
    """

    def json(self, **kwargs: typing.Any) -> str:
        kwargs_with_defaults: typing.Any = {"by_alias": True, "exclude_unset": True, **kwargs}
        return super().json(**kwargs_with_defaults)

    def dict(self, **kwargs: typing.Any) -> typing.Dict[str, typing.Any]:
        kwargs_with_defaults: typing.Any = {"by_alias": True, "exclude_unset": True, **kwargs}
        return super().dict(**kwargs_with_defaults)

    class Config:
        frozen = True
        smart_union = True
        allow_population_by_field_name = True
        populate_by_name = True
        extra = pydantic_v1.Extra.allow
        json_encoders = {dt.datetime: serialize_datetime}
