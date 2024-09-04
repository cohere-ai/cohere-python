import typing

from . import EmbedByTypeResponseEmbeddings
from .core.pydantic_utilities import _get_model_fields, Model

from pprint import pprint

def allow_access_to_aliases(self: typing.Type["Model"], name):
    for field_name, field_info in _get_model_fields(self).items():
        alias = field_info.alias or (field_info and field_info.metadata and field_info.metadata[0] and field_info.metadata[0].alias)
        if alias == name:
            return getattr(self, field_name)
    raise AttributeError(
        f"'{type(self).__name__}' object has no attribute '{name}'")


def run_overrides():
    """
        These are overrides to allow us to make changes to generated code without touching the generated files themselves.
        Should be used judiciously!
    """

    # Override to allow access to aliases in EmbedByTypeResponseEmbeddings eg embeddings.float rather than embeddings.float_
    setattr(EmbedByTypeResponseEmbeddings, "__getattr__", allow_access_to_aliases)
