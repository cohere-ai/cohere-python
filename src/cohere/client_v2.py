import os
import typing
from concurrent.futures import ThreadPoolExecutor

import httpx
from .client import AsyncClient, Client
from .environment import ClientEnvironment
from .v2.client import AsyncRawV2Client, AsyncV2Client, RawV2Client, V2Client


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
