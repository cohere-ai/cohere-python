# This file was auto-generated by Fern from our API Definition.

import datetime as dt
import typing

from ....core.datetime_utils import serialize_datetime
from .finetuned_model import FinetunedModel

if pydantic.VERSION.startswith("2."):
    import pydantic.v1 as pydantic  # type: ignore
else:
    import pydantic  # type: ignore


class ListFinetunedModelsResponse(pydantic.BaseModel):
    """
    Response to a request to list fine-tuned models.
    """

    finetuned_models: typing.Optional[typing.List[FinetunedModel]] = pydantic.Field(default=None)
    """
    List of fine-tuned models matching the request.
    """

    next_page_token: typing.Optional[str] = pydantic.Field(default=None)
    """
    Pagination token to retrieve the next page of results. If the value is "",
    it means no further results for the request.
    """

    total_size: typing.Optional[int] = pydantic.Field(default=None)
    """
    Total count of results.
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
        extra = pydantic.Extra.allow
        json_encoders = {dt.datetime: serialize_datetime}
