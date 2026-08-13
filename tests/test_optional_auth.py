import asyncio
import typing
import unittest

import cohere


def _headers(client: typing.Any) -> typing.Dict[str, str]:
    return client._client_wrapper.get_headers()


async def _async_headers(client: typing.Any) -> typing.Dict[str, str]:
    return await client._client_wrapper.async_get_headers()


class TestOptionalAuth(unittest.TestCase):
    def test_empty_api_key_omits_authorization_header(self) -> None:
        self.assertNotIn("Authorization", _headers(cohere.Client(api_key="")))
        self.assertNotIn("Authorization", _headers(cohere.ClientV2(api_key="")))
        self.assertNotIn("Authorization", asyncio.run(_async_headers(cohere.AsyncClient(api_key=""))))
        self.assertNotIn("Authorization", asyncio.run(_async_headers(cohere.AsyncClientV2(api_key=""))))

    def test_api_key_is_sent_when_provided(self) -> None:
        self.assertEqual(_headers(cohere.Client(api_key="n/a"))["Authorization"], "Bearer n/a")
        self.assertEqual(_headers(cohere.ClientV2(api_key="n/a"))["Authorization"], "Bearer n/a")
        self.assertEqual(
            asyncio.run(_async_headers(cohere.AsyncClient(api_key="n/a")))["Authorization"], "Bearer n/a"
        )

    def test_callable_api_key_returning_empty_string_omits_authorization_header(self) -> None:
        self.assertNotIn("Authorization", _headers(cohere.Client(api_key=lambda: "")))
