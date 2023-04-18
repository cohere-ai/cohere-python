import random
import string
import unittest

from utils import get_api_key

import cohere

API_KEY = get_api_key()
co = cohere.Client(API_KEY)


class TestCodebook(unittest.TestCase):
    def test_success(self):
        prediction = co.codebook(model="multilingual-22-12", compression_codebook="32x")
        self.assertEqual(len(prediction.codebook), 96)
        self.assertIsInstance(prediction.codebook[0], list)
        self.assertIsInstance(prediction.codebook[1], list)
        self.assertEqual(len(prediction.codebook[0]), 256)
        self.assertEqual(len(prediction.codebook[1]), 8)
        self.assertTrue(prediction.meta)
        self.assertTrue(prediction.meta["api_version"])
        self.assertTrue(prediction.meta["api_version"]["version"])

    def test_default_model(self):
        prediction = co.codebook(model="multilingual-22-12",)
        self.assertEqual(len(prediction.codebook), 96)
        self.assertIsInstance(prediction.codebook[0], list)
        self.assertIsInstance(prediction.codebook[1], list)

