# This file was auto-generated by Fern from our API Definition.

import typing

import pydantic

from ..core.pydantic_utilities import IS_PYDANTIC_V2
from ..core.unchecked_base_model import UncheckedBaseModel


class CreateConnectorOAuth(UncheckedBaseModel):
    client_id: typing.Optional[str] = pydantic.Field(default=None)
    """
    The OAuth 2.0 client ID. This fields is encrypted at rest.
    """

    client_secret: typing.Optional[str] = pydantic.Field(default=None)
    """
    The OAuth 2.0 client Secret. This field is encrypted at rest and never returned in a response.
    """

    authorize_url: typing.Optional[str] = pydantic.Field(default=None)
    """
    The OAuth 2.0 /authorize endpoint to use when users authorize the connector.
    """

    token_url: typing.Optional[str] = pydantic.Field(default=None)
    """
    The OAuth 2.0 /token endpoint to use when users authorize the connector.
    """

    scope: typing.Optional[str] = pydantic.Field(default=None)
    """
    The OAuth scopes to request when users authorize the connector.
    """

    if IS_PYDANTIC_V2:
        model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(extra="allow", frozen=True)  # type: ignore # Pydantic v2
    else:

        class Config:
            frozen = True
            smart_union = True
            extra = pydantic.Extra.allow
