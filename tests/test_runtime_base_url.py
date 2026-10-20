import typing

import httpx
import pytest

import cohere


@pytest.mark.parametrize("client_class", [cohere.Client, cohere.ClientV2])
@pytest.mark.parametrize("explicit", [False, True])
def test_sync_base_url_is_resolved_at_construction(
    client_class: typing.Any, explicit: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    requests: typing.List[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"models": []})

    for environment_url in ["https://first.example", "https://second.example", None]:
        if environment_url is None:
            monkeypatch.delenv("CO_API_URL", raising=False)
        else:
            monkeypatch.setenv("CO_API_URL", environment_url)
        kwargs = {"base_url": "https://explicit.example"} if explicit else {}
        with client_class(
            api_key="offline-test-key",
            httpx_client=httpx.Client(transport=httpx.MockTransport(respond)),
            **kwargs,
        ) as client:
            assert client.models.list().models == []
        expected_host = "explicit.example" if explicit else httpx.URL(environment_url or "https://api.cohere.com").host
        assert requests[-1].url.host == expected_host
        assert requests[-1].url.path == "/v1/models"


@pytest.mark.parametrize("client_class", [cohere.AsyncClient, cohere.AsyncClientV2])
@pytest.mark.parametrize("explicit", [False, True])
async def test_async_base_url_is_resolved_at_construction(
    client_class: typing.Any, explicit: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    requests: typing.List[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"models": []})

    for environment_url in ["https://first.example", "https://second.example", None]:
        if environment_url is None:
            monkeypatch.delenv("CO_API_URL", raising=False)
        else:
            monkeypatch.setenv("CO_API_URL", environment_url)
        kwargs = {"base_url": "https://explicit.example"} if explicit else {}
        async with client_class(
            api_key="offline-test-key",
            httpx_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
            **kwargs,
        ) as client:
            assert (await client.models.list()).models == []
        expected_host = "explicit.example" if explicit else httpx.URL(environment_url or "https://api.cohere.com").host
        assert requests[-1].url.host == expected_host
        assert requests[-1].url.path == "/v1/models"
