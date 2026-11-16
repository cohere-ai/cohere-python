import os
import typing
from concurrent.futures import ThreadPoolExecutor

import httpx
from .client import AsyncClient, Client
from .config import embed_stream_batch_size
from .core import RequestOptions
from .environment import ClientEnvironment
from .manually_maintained.streaming_embed import StreamedEmbedding, extract_embeddings_from_response
from .types import EmbedInput, EmbedInputType, EmbeddingType
from .v2.client import OMIT, AsyncRawV2Client, AsyncV2Client, RawV2Client, V2Client
from .v2.types import V2EmbedRequestTruncate


class _CombinedRawClient:
    """Proxy that combines v1 and v2 raw clients.

    V2Client and Client both assign to self._raw_client in __init__,
    causing a collision when combined in ClientV2/AsyncClientV2.
    This proxy delegates to v2 first, falling back to v1 for
    legacy methods like generate_stream.
    """

    def __init__(self, v1_raw_client: typing.Any, v2_raw_client: typing.Any):
        self._v1 = v1_raw_client
        self._v2 = v2_raw_client

    def __getattr__(self, name: str) -> typing.Any:
        try:
            return getattr(self._v2, name)
        except AttributeError:
            return getattr(self._v1, name)


class ClientV2(V2Client, Client):  # type: ignore
    def __init__(
        self,
        api_key: typing.Optional[typing.Union[str,
                                              typing.Callable[[], str]]] = None,
        *,
        base_url: typing.Optional[str] = os.getenv("CO_API_URL"),
        environment: ClientEnvironment = ClientEnvironment.PRODUCTION,
        client_name: typing.Optional[str] = None,
        timeout: typing.Optional[float] = None,
        max_retries: typing.Optional[int] = None,
        httpx_client: typing.Optional[httpx.Client] = None,
        thread_pool_executor: ThreadPoolExecutor = ThreadPoolExecutor(64),
        log_warning_experimental_features: bool = True,
    ):
        Client.__init__(
            self,
            api_key=api_key,
            base_url=base_url,
            environment=environment,
            client_name=client_name,
            timeout=timeout,
            max_retries=max_retries,
            httpx_client=httpx_client,
            thread_pool_executor=thread_pool_executor,
            log_warning_experimental_features=log_warning_experimental_features,
        )
        v1_raw = self._raw_client
        V2Client.__init__(
            self,
            client_wrapper=self._client_wrapper
        )
        self._raw_client = typing.cast(RawV2Client, _CombinedRawClient(v1_raw, self._raw_client))

    def embed_stream(  # type: ignore[override]
        self,
        *,
        model: typing.Optional[str] = OMIT,
        input_type: typing.Optional[EmbedInputType] = OMIT,
        texts: typing.Optional[typing.Sequence[str]] = OMIT,
        inputs: typing.Optional[typing.Sequence[EmbedInput]] = OMIT,
        max_tokens: typing.Optional[int] = OMIT,
        output_dimension: typing.Optional[int] = OMIT,
        embedding_types: typing.Optional[typing.Sequence[EmbeddingType]] = OMIT,
        truncate: typing.Optional[V2EmbedRequestTruncate] = OMIT,
        priority: typing.Optional[int] = OMIT,
        batch_size: int = embed_stream_batch_size,
        request_options: typing.Optional[RequestOptions] = None,
    ) -> typing.Iterator[StreamedEmbedding]:
        """
        Memory-efficient v2 embed that yields embeddings one batch at a time.

        Splits `texts` or `inputs` into requests of `batch_size`, sends them to /v2/embed one after
        another, and yields a StreamedEmbedding per embedding as each response arrives, so large
        datasets can be written to a vector store without holding every embedding in memory.
        Takes the same parameters as `embed()`, except `images`. `model` and `input_type` stay optional,
        as they were when ClientV2 inherited the v1 embed_stream(); when omitted, they are left out of
        the request.

        Args:
            model: Embedding model ID.
            input_type: Input type (search_document, search_query, etc.).
            texts: Texts to embed. Pass either `texts` or `inputs`.
            inputs: Inputs to embed, each mixing text and image content. Pass either `texts` or `inputs`.
            max_tokens: Maximum number of tokens to embed per input.
            output_dimension: Number of dimensions of each output embedding.
            embedding_types: Types of embeddings to return (float, int8, etc.).
            truncate: How to handle inputs longer than the max token length.
            priority: Request priority; lower numbers are handled first.
            batch_size: Entries per API call. Defaults to 96 (API max).
            request_options: Request-specific configuration.

        Yields:
            StreamedEmbedding with index, embedding and embedding_type. `text` is set for `texts`
            and None for `inputs`.
        """
        has_texts = texts is not OMIT and texts is not None
        has_inputs = inputs is not OMIT and inputs is not None
        if has_texts == has_inputs:
            raise ValueError("embed_stream requires exactly one of texts or inputs")
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")

        entries = list(typing.cast(typing.Sequence[typing.Any], texts if has_texts else inputs))

        for batch_start in range(0, len(entries), batch_size):
            batch = entries[batch_start : batch_start + batch_size]

            response = V2Client.embed(
                self,
                model=typing.cast(str, model),
                input_type=typing.cast(EmbedInputType, input_type),
                texts=batch if has_texts else OMIT,
                inputs=OMIT if has_texts else batch,
                max_tokens=max_tokens,
                output_dimension=output_dimension,
                embedding_types=embedding_types,
                truncate=truncate,
                priority=priority,
                request_options=request_options,
            )

            yield from extract_embeddings_from_response(response.dict(), batch if has_texts else [], batch_start)


class AsyncClientV2(AsyncV2Client, AsyncClient):  # type: ignore
    def __init__(
        self,
        api_key: typing.Optional[typing.Union[str,
                                              typing.Callable[[], str]]] = None,
        *,
        base_url: typing.Optional[str] = os.getenv("CO_API_URL"),
        environment: ClientEnvironment = ClientEnvironment.PRODUCTION,
        client_name: typing.Optional[str] = None,
        timeout: typing.Optional[float] = None,
        max_retries: typing.Optional[int] = None,
        httpx_client: typing.Optional[httpx.AsyncClient] = None,
        thread_pool_executor: ThreadPoolExecutor = ThreadPoolExecutor(64),
        log_warning_experimental_features: bool = True,
    ):
        AsyncClient.__init__(
            self,
            api_key=api_key,
            base_url=base_url,
            environment=environment,
            client_name=client_name,
            timeout=timeout,
            max_retries=max_retries,
            httpx_client=httpx_client,
            thread_pool_executor=thread_pool_executor,
            log_warning_experimental_features=log_warning_experimental_features,
        )
        v1_raw = self._raw_client
        AsyncV2Client.__init__(
            self,
            client_wrapper=self._client_wrapper
        )
        self._raw_client = typing.cast(AsyncRawV2Client, _CombinedRawClient(v1_raw, self._raw_client))
