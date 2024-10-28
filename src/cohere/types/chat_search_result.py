# This file was auto-generated by Fern from our API Definition.

from ..core.unchecked_base_model import UncheckedBaseModel
import typing
from .chat_search_query import ChatSearchQuery
from .chat_search_result_connector import ChatSearchResultConnector
import pydantic
from ..core.pydantic_utilities import IS_PYDANTIC_V2


class ChatSearchResult(UncheckedBaseModel):
    search_query: typing.Optional[ChatSearchQuery] = None
    connector: ChatSearchResultConnector = pydantic.Field()
    """
    The connector from which this result comes from.
    """

    document_ids: typing.List[str] = pydantic.Field()
    """
    Identifiers of documents found by this search query.
    """

    error_message: typing.Optional[str] = pydantic.Field(default=None)
    """
    An error message if the search failed.
    """

    continue_on_failure: typing.Optional[bool] = pydantic.Field(default=None)
    """
    Whether a chat request should continue or not if the request to this connector fails.
    """

    if IS_PYDANTIC_V2:
        model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(extra="allow")  # type: ignore # Pydantic v2
    else:

        class Config:
            smart_union = True
            extra = pydantic.Extra.allow
