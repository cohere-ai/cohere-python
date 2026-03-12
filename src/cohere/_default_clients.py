import typing

import httpx

COHERE_DEFAULT_TIMEOUT = 300

try:
    import httpx_aiohttp
except ImportError:

    class DefaultAioHttpClient(httpx.AsyncClient):  # type: ignore
        def __init__(self, **kwargs: typing.Any) -> None:
            raise RuntimeError(
                "To use the aiohttp client, install the aiohttp extra: "
                "pip install cohere[aiohttp]"
            )

else:

    class DefaultAioHttpClient(httpx_aiohttp.HttpxAiohttpClient):  # type: ignore
        def __init__(self, **kwargs: typing.Any) -> None:
            kwargs.setdefault("timeout", COHERE_DEFAULT_TIMEOUT)
            kwargs.setdefault("follow_redirects", True)
            super().__init__(**kwargs)


class DefaultAsyncHttpxClient(httpx.AsyncClient):
    def __init__(self, **kwargs: typing.Any) -> None:
        kwargs.setdefault("timeout", COHERE_DEFAULT_TIMEOUT)
        kwargs.setdefault("follow_redirects", True)
        super().__init__(**kwargs)
