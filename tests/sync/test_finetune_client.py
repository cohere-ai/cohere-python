import os
import unittest
from unittest import mock

from utils import get_api_key

import cohere

API_KEY = get_api_key()
client = cohere.FinetuneClient(API_KEY)


class TestFinetuneClient(unittest.TestCase):
    def test_list(self):
        self.assertTrue(len(client.list()) > 0)

    def test_get(self):
        first = client.list()[0]
        by_id = client.get(first.id)
        self.assertEqual(first.id, by_id.id)

    def test_get_by_name(self):
        first = client.list()[0]
        by_id = client.get_by_name(first.name)
        self.assertEqual(first.id, by_id.id)
