import unittest

from utils import get_api_key

import cohere

API_KEY = get_api_key()
client = cohere.Client(API_KEY)


class TestFinetuneClient(unittest.TestCase):
    def test_list(self):
        self.assertTrue(len(client.list_custom_models()) > 0)

    def test_get(self):
        first = client.list_custom_models()[0]
        by_id = client.get_custom_model(first.id)
        self.assertEqual(first.id, by_id.id)

    def test_get_by_name(self):
        name = "joon-customer-300"
        cm = client.get_custom_model_by_name(name)
        self.assertEqual(name, cm.name)
