# This file was auto-generated by Fern from our API Definition.

from ..core.unchecked_base_model import UncheckedBaseModel
import typing
import pydantic
from ..core.pydantic_utilities import IS_PYDANTIC_V2


class JsonResponseFormatV2(UncheckedBaseModel):
    json_schema: typing.Optional[typing.Dict[str, typing.Optional[typing.Any]]] = pydantic.Field(default=None)
    """
    A [JSON schema](https://json-schema.org/overview/what-is-jsonschema) object that the output will adhere to. There are some restrictions we have on the schema, refer to [our guide](/docs/structured-outputs-json#schema-constraints) for more information.
    Example (required name and age object):
    
    ```json
    {
      "type": "object",
      "properties": {
        "name": { "type": "string" },
        "age": { "type": "integer" }
      },
      "required": ["name", "age"]
    }
    ```
    
    **Note**: This field must not be specified when the `type` is set to `"text"`.
    """

    if IS_PYDANTIC_V2:
        model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(extra="allow")  # type: ignore # Pydantic v2
    else:

        class Config:
            smart_union = True
            extra = pydantic.Extra.allow
