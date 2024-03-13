# This file was auto-generated by Fern from our API Definition.

import datetime as dt
import typing

from ..core.datetime_utils import serialize_datetime

try:
    import pydantic.v1 as pydantic  # type: ignore
except ImportError:
    import pydantic  # type: ignore


class ChatConnector(pydantic.BaseModel):
    """
    The connector used for fetching documents.
    """

    id: str = pydantic.Field()
    """
    The identifier of the connector.
    """

    user_access_token: typing.Optional[str] = pydantic.Field(default=None)
    """
    When specified, this user access token will be passed to the connector in the Authorization header instead of the Cohere generated one.
    """

    continue_on_failure: typing.Optional[bool] = pydantic.Field(default=None)
    """
    Defaults to `false`.
    
    When `true`, the request will continue if this connector returned an error.
    """

    options: typing.Optional[typing.Dict[str, typing.Any]] = pydantic.Field(default=None)
    """
    Provides the connector with different settings at request time. The key/value pairs of this object are specific to each connector.
    
    For example, the connector `web-search` supports the `site` option, which limits search results to the specified domain.
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