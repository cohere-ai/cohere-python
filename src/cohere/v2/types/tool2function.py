# This file was auto-generated by Fern from our API Definition.

from ...core.unchecked_base_model import UncheckedBaseModel
import typing
import pydantic
from ...core.pydantic_utilities import IS_PYDANTIC_V2


class Tool2Function(UncheckedBaseModel):
    """
    The function to be executed.
    """

    name: typing.Optional[str] = pydantic.Field(default=None)
    """
    The name of the function.
    """

    description: typing.Optional[str] = pydantic.Field(default=None)
    """
    The description of the function.
    """

    parameters: typing.Optional[typing.Dict[str, typing.Optional[typing.Any]]] = pydantic.Field(default=None)
    """
    The parameters of the function as a JSON schema.
    """

    if IS_PYDANTIC_V2:
        model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(extra="allow", frozen=True)  # type: ignore # Pydantic v2
    else:

        class Config:
            frozen = True
            smart_union = True
            extra = pydantic.Extra.allow
