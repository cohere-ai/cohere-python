from .client import Client, AsyncClient
from .v2.client import V2Client, AsyncV2Client
import typing
from .environment import ClientEnvironment
import os
import httpx
from concurrent.futures import ThreadPoolExecutor


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
        V2Client.__init__(
            self,
            client_wrapper=self._client_wrapper
        )


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
        AsyncV2Client.__init__(
            self,
            client_wrapper=self._client_wrapper
        )
