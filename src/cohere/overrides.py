import typing
import uuid

from . import EmbedByTypeResponseEmbeddings
from .core.pydantic_utilities import _get_model_fields, Model, IS_PYDANTIC_V2

from pprint import pprint


def get_fields(obj) -> typing.List[str]:
    return [str(x) for x in _get_model_fields(obj).keys()]


def get_aliases_or_field(obj) -> typing.List[str]:
    return [
        field_info.alias or (field_info and field_info.metadata and field_info.metadata[0] and field_info.metadata[0].alias) or field_name # type: ignore
        for field_name, field_info
        in _get_model_fields(obj).items()
    ]


def get_aliases_and_fields(obj):
    # merge and dedup get_fields(obj), get_aliases_or_field(obj)
    return list(set(get_fields(obj) + get_aliases_or_field(obj)))


def allow_access_to_aliases(self: typing.Type["Model"], name):
    for field_name, field_info in _get_model_fields(self).items():
        alias = field_info.alias or (
                    field_info and field_info.metadata and field_info.metadata[0] and field_info.metadata[0].alias) # type: ignore
        if alias == name or field_name == name:
            return getattr(self, field_name)
    raise AttributeError(
        f"'{type(self).__name__}' object has no attribute '{name}'")


def make_tool_call_v2_id_optional(cls):
    """
    Override ToolCallV2 to make the 'id' field optional with a default UUID.
    This ensures backward compatibility with code that doesn't provide an id.

    We wrap the __init__ method to inject a default id before Pydantic validation runs.
    """
    # Store the original __init__ method
    original_init = cls.__init__

    def patched_init(self, /, **data):
        """Patched __init__ that injects default id if not provided."""
        # Inject default UUID if 'id' is not in the data
        if 'id' not in data:
            data['id'] = str(uuid.uuid4())

        # Call the original __init__ with modified data
        original_init(self, **data)

    # Replace the __init__ method
    cls.__init__ = patched_init

    return cls


def omit_authorization_header_when_api_key_is_empty() -> None:
    """
    Do not send an `Authorization` header when the client is created with an empty API key.

    This allows pointing the client at a proxy or a self-hosted deployment that performs its own
    authentication, e.g. `cohere.Client(api_key="")`.
    """
    from .core.client_wrapper import AsyncClientWrapper, BaseClientWrapper

    if getattr(BaseClientWrapper, "_omits_empty_authorization", False):
        return

    get_headers = BaseClientWrapper.get_headers
    async_get_headers = AsyncClientWrapper.async_get_headers

    def patched_get_headers(self: BaseClientWrapper) -> typing.Dict[str, str]:
        headers = get_headers(self)
        # Inspect the header that get_headers() already built rather than calling _get_token()
        # again: for the callable `api_key` form that would invoke the supplier twice per request,
        # and a supplier whose value changes between the two calls would produce the wrong header.
        if headers.get("Authorization") == "Bearer ":
            headers.pop("Authorization", None)
        return headers

    async def patched_async_get_headers(self: AsyncClientWrapper) -> typing.Dict[str, str]:
        headers = await async_get_headers(self)
        if headers.get("Authorization") == "Bearer ":
            headers.pop("Authorization", None)
        return headers

    BaseClientWrapper.get_headers = patched_get_headers  # type: ignore[method-assign]
    AsyncClientWrapper.async_get_headers = patched_async_get_headers  # type: ignore[method-assign]
    BaseClientWrapper._omits_empty_authorization = True  # type: ignore[attr-defined]


def run_overrides():
    """
        These are overrides to allow us to make changes to generated code without touching the generated files themselves.
        Should be used judiciously!
    """

    # Override to skip the Authorization header entirely when an empty api_key is passed
    omit_authorization_header_when_api_key_is_empty()

    # Override to allow access to aliases in EmbedByTypeResponseEmbeddings eg embeddings.float rather than embeddings.float_
    setattr(EmbedByTypeResponseEmbeddings, "__getattr__", allow_access_to_aliases)

    # Import ToolCallV2 lazily to avoid circular dependency issues
    from . import ToolCallV2

    # Override ToolCallV2 to make id field optional with default UUID
    make_tool_call_v2_id_optional(ToolCallV2)


# Run overrides immediately at module import time to ensure they're applied
# before any code tries to use the modified classes
run_overrides()
