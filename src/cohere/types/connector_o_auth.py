# This file was auto-generated by Fern from our API Definition.

import datetime as dt
import typing

from ..core.datetime_utils import serialize_datetime
from ..core.pydantic_utilities import pydantic_v1


class ConnectorOAuth(pydantic_v1.BaseModel):
    client_id: typing.Optional[str] = pydantic_v1.Field(default=None)
    """
    The OAuth 2.0 client ID. This field is encrypted at rest.
    """

    client_secret: typing.Optional[str] = pydantic_v1.Field(default=None)
    """
    The OAuth 2.0 client Secret. This field is encrypted at rest and never returned in a response.
    """

    authorize_url: str = pydantic_v1.Field()
    """
    The OAuth 2.0 /authorize endpoint to use when users authorize the connector.
    """

    token_url: str = pydantic_v1.Field()
    """
    The OAuth 2.0 /token endpoint to use when users authorize the connector.
    """

    scope: typing.Optional[str] = pydantic_v1.Field(default=None)
    """
    The OAuth scopes to request when users authorize the connector.
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
