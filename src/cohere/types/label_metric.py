# This file was auto-generated by Fern from our API Definition.

import datetime as dt
import typing

from ..core.datetime_utils import serialize_datetime
from ..core.pydantic_utilities import pydantic_v1


class LabelMetric(pydantic_v1.BaseModel):
    total_examples: typing.Optional[str] = pydantic_v1.Field(alias="totalExamples", default=None)
    """
    Total number of examples for this label
    """

    label: typing.Optional[str] = pydantic_v1.Field(default=None)
    """
    value of the label
    """

    samples: typing.Optional[typing.List[str]] = pydantic_v1.Field(default=None)
    """
    samples for this label
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
