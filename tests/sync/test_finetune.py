import unittest

from utils import get_api_key

import cohere

API_KEY = get_api_key()
client = cohere.Client(API_KEY)


class TestFinetuneClient(unittest.TestCase):
    def test_list(self):
        self.assertTrue(len(client.list_finetunes()) > 0)

    def test_get(self):
        first = client.list_finetunes()[0]
        by_id = client.get_finetune(first.id)
        self.assertEqual(first.id, by_id.id)

    def test_get_by_name(self):
        first = client.list_finetunes()[0]
        by_id = client.get_finetune_by_name(first.name)
        self.assertEqual(first.id, by_id.id)
