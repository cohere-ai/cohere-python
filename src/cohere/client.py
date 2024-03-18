import typing

import httpx

from .base_client import BaseCohere, AsyncBaseCohere
from .environment import CohereEnvironment


def validate_args(obj: typing.Any, method_name: str, check_fn: typing.Callable[[typing.Any], typing.Any]) -> None:
    method = getattr(obj, method_name)

    def wrapped(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
        check_fn(*args, **kwargs)
        return method(*args, **kwargs)

    setattr(obj, method_name, wrapped)


def throw_if_stream_is_true(*args, **kwargs) -> None:
    if kwargs.get("stream") is True:
        raise ValueError(
            "Since python sdk cohere==5.0.0, you must now use chat_stream(...) instead of chat(stream=True, ...)"
        )


class Client(BaseCohere):
    def __init__(
            self,
            api_key: typing.Union[str, typing.Callable[[], str]],
            *,
            base_url: typing.Optional[str] = None,
            environment: CohereEnvironment = CohereEnvironment.PRODUCTION,
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

        validate_args(self, "chat", throw_if_stream_is_true)


class AsyncClient(AsyncBaseCohere):
    def __init__(
            self,
            api_key: typing.Union[str, typing.Callable[[], str]],
            *,
            base_url: typing.Optional[str] = None,
            environment: CohereEnvironment = CohereEnvironment.PRODUCTION,
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

        validate_args(self, "chat", throw_if_stream_is_true)