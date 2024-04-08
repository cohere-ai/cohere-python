# This file was auto-generated by Fern from our API Definition.

import datetime as dt
import typing

from ..core.datetime_utils import serialize_datetime
from ..core.pydantic_utilities import pydantic_v1
from ..core.unchecked_base_model import UncheckedBaseModel


class ChatStreamRequestConnectorsSearchOptions(UncheckedBaseModel):
    """
    (internal) Sets inference and model options for RAG search query and tool use generations. Defaults are used when options are not specified here, meaning that other parameters outside of connectors_search_options are ignored (such as model= or temperature=).
    """

    model: typing.Optional[typing.Any] = None
    temperature: typing.Optional[typing.Any] = None
    max_tokens: typing.Optional[typing.Any] = None
    preamble: typing.Optional[typing.Any] = None
    seed: typing.Optional[float] = pydantic_v1.Field(default=None)
    """
    If specified, the backend will make a best effort to sample tokens deterministically, such that repeated requests with the same seed and parameters should return the same result. However, determinsim cannot be totally guaranteed.
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
        extra = pydantic_v1.Extra.allow
        json_encoders = {dt.datetime: serialize_datetime}
