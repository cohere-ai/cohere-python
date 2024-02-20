import typing

import httpx

from . import BaseCohereEnvironment
from .base_client import BaseCohere

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
        super(BaseCohere, self).__init__(
            base_url=base_url,
            environment=environment,
            client_name=client_name,
            token=api_key,
            timeout=timeout,
            httpx_client=httpx_client,
        )
