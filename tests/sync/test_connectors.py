import pickle
import unittest

from utils import get_api_key

import cohere
from cohere.responses import Connector, ConnectorOAuth


class TestConnectors(unittest.TestCase):
    def test_connector(self):
        co = self.create_co()
        created_connector = co.create_connector(
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
        assert created_connector.name == "ci-test"
        assert created_connector.url == "https://ci-test.com/search"
        assert created_connector.active is True
        assert created_connector.continue_on_failure is True
        assert created_connector.excludes == ["irrelevant_field"]
        assert isinstance(created_connector.oauth, ConnectorOAuth)
        assert created_connector.oauth.authorize_url == "https://someurl.com"
        assert created_connector.oauth.token_url == "https://someurl.com"
        assert created_connector.oauth.scope == "some_scope"
        id = created_connector.id

        updated_connector = co.update_connector(id, name="ci-test2", url="https://ci-test2.com/search", active=False)
        get_connector = co.get_connector(id)
        assert pickle.dumps(updated_connector) == pickle.dumps(get_connector)
        assert get_connector.name == "ci-test2"
        assert get_connector.url == "https://ci-test2.com/search"
        assert get_connector.active is False
        assert get_connector.continue_on_failure is True
        assert get_connector.excludes == ["irrelevant_field"]
        assert isinstance(get_connector.oauth, ConnectorOAuth)
        assert get_connector.oauth.authorize_url == "https://someurl.com"
        assert get_connector.oauth.token_url == "https://someurl.com"
        assert get_connector.oauth.scope == "some_scope"

        connectors = co.list_connectors(0, 0)
        found_connector = False
        for connector in connectors:
            if connector.name != "ci-test":
                continue
            if connector.url != "https://ci-test.com/search":
                continue
            if connector.active is not True:
                continue
            if connector.continue_on_failure is not True:
                continue
            if connector.excludes != ["irrelevant_field"]:
                continue
            if not isinstance(connector.oauth, ConnectorOAuth):
                continue
            if connector.oauth.authorize_url != "https://someurl.com":
                continue
            if connector.oauth.token_url != "https://someurl.com":
                continue
            if connector.oauth.scope != "some_scope":
                continue
            found_connector = True
        assert found_connector

        co.delete_connector(id)

    def create_co(self) -> cohere.Client:
        return cohere.Client(get_api_key(), check_api_key=False, client_name="test")

    def check_result(self, connector: Connector):
        assert connector.id
        assert connector.created_at
        assert connector.connector_type
        assert connector.name
