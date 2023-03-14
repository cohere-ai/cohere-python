import unittest

from utils import get_api_key

import cohere

API_KEY = get_api_key()


class TestClient(unittest.TestCase):
    def test_client_name(self):
        co = cohere.Client(API_KEY, client_name="test")
        co.generate(model="medium", prompt="co:here", max_tokens=1)

    def test_client_404(self):
        co = cohere.Client(API_KEY, client_name="test")
        with self.assertRaises(cohere.CohereError):
            co._request("/test-404")
