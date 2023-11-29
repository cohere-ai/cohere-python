import pickle

import pytest

from cohere import AsyncClient
from cohere.responses import Connector, ConnectorOAuth


@pytest.mark.asyncio
async def test_async_connector(async_client: AsyncClient):
    created_connector = await async_client.create_connector(
        name="ci-test",
        url="https://ci-test.com/search",
        active=True,
        continue_on_failure=True,
        excludes=["irrelevant_field"],
        oauth={
            "client_secret": "some_secret",
            "client_id": "some_id",
            "authorize_url": "https://someurl.com",
            "token_url": "https://someurl.com",
            "scope": "some_scope",
        },
    )
    assert isinstance(created_connector, Connector)
    assert created_connector.name == "ci-test"
    assert created_connector.url == "https://ci-test.com/search"
    assert created_connector.active is True
    assert created_connector.continue_on_failure is True
    assert created_connector.excludes == ["irrelevant_field"]
    assert isinstance(created_connector.oauth, ConnectorOAuth)
    assert created_connector.oauth.authorize_url == "https://someurl.com"
    assert created_connector.oauth.token_url == "https://someurl.com"
    assert created_connector.oauth.scope == "some_scope"
    connector_id = created_connector.id

    updated_connector = await async_client.update_connector(
        connector_id, name="ci-test2", url="https://ci-test2.com/search", active=False
    )
    get_connector = await async_client.get_connector(connector_id)
    assert pickle.dumps(updated_connector) == pickle.dumps(get_connector)
    assert isinstance(get_connector, Connector)
    assert get_connector.name == "ci-test2"
    assert get_connector.url == "https://ci-test2.com/search"
    assert get_connector.active is False
    assert get_connector.continue_on_failure is True
    assert get_connector.excludes == ["irrelevant_field"]
    assert isinstance(get_connector.oauth, ConnectorOAuth)
    assert get_connector.oauth.authorize_url == "https://someurl.com"
    assert get_connector.oauth.token_url == "https://someurl.com"
    assert get_connector.oauth.scope == "some_scope"

    connectors = await async_client.list_connectors(0, 0)
    found_connector = False
    for connector in connectors:
        if connector.id != connector_id:
            continue
        found_connector = True
    assert found_connector

    await async_client.delete_connector(connector_id)
