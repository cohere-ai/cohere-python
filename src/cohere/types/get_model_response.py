# This file was auto-generated by Fern from our API Definition.

import datetime as dt
import typing

from ..core.datetime_utils import serialize_datetime
from .compatible_endpoint import CompatibleEndpoint

try:
    import pydantic.v1 as pydantic  # type: ignore
except ImportError:
    import pydantic  # type: ignore


class GetModelResponse(pydantic.BaseModel):
    """
    Contains information about the model and which API endpoints it can be used with.
    """

    name: typing.Optional[str] = pydantic.Field(default=None)
    """
    Specify this name in the `model` parameter of API requests to use your chosen model.
    """

    endpoints: typing.Optional[typing.List[CompatibleEndpoint]] = pydantic.Field(default=None)
    """
    The API endpoints that the model is compatible with.
    """

    finetuned: typing.Optional[bool] = pydantic.Field(default=None)
    """
    Whether the model has been fine-tuned or not.
    """

    context_length: typing.Optional[float] = pydantic.Field(default=None)
    """
    The maximum number of tokens that the model can process in a single request. Note that not all of these tokens are always available due to special tokens and preambles that Cohere has added by default.
    """

    tokenizer: typing.Optional[str] = pydantic.Field(default=None)
    """
    The name of the tokenizer used for the model.
    """

    tokenizer_url: typing.Optional[str] = pydantic.Field(default=None)
    """
    Public URL to the tokenizer's configuration file.
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
