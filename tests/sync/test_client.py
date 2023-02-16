import unittest

import cohere

from utils import get_api_key

API_KEY = get_api_key()


class TestClient(unittest.TestCase):

    def test_client_name(self):
        co = cohere.Client(API_KEY, client_name='test')
        co.generate(model='medium', prompt='co:here', max_tokens=1)
