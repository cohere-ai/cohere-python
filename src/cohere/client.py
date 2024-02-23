import typing

import httpx

from .environment import BaseCohereEnvironment
from .base_client import BaseCohere, AsyncBaseCohere


class Client(BaseCohere):
    def __init__(
            self,
            api_key: typing.Union[str, typing.Callable[[], str]],
            *,
            base_url: typing.Optional[str] = None,
            environment: BaseCohereEnvironment = BaseCohereEnvironment.PRODUCTION,
            client_name: typing.Optional[str] = None,
            timeout: typing.Optional[float] = 60,
            httpx_client: typing.Optional[httpx.Client] = None,
    ):
        BaseCohere.__init__(
            self,
            base_url=base_url,
            environment=environment,
            client_name=client_name,
            token=api_key,
            timeout=timeout,
            httpx_client=httpx_client,
        )


class AsyncClient(AsyncBaseCohere):
    def __init__(
            self,
            api_key: typing.Union[str, typing.Callable[[], str]],
            *,
            base_url: typing.Optional[str] = None,
            environment: BaseCohereEnvironment = BaseCohereEnvironment.PRODUCTION,
            client_name: typing.Optional[str] = None,
            timeout: typing.Optional[float] = 60,
            httpx_client: typing.Optional[httpx.AsyncClient] = None,
    ):
        AsyncBaseCohere.__init__(
            self,
            base_url=base_url,
            environment=environment,
            client_name=client_name,
            token=api_key,
            timeout=timeout,
            httpx_client=httpx_client,
        )
