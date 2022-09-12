import unittest
import cohere
from utils import get_api_key

co = cohere.Client(get_api_key())


class TestDetokenize(unittest.TestCase):
    def test_success(self):
        text = co.detokenize([10104, 12221, 974, 514, 34]).text
        self.assertEqual(text, "detokenize me!")

    def test_empty_tokens(self):
        text = co.detokenize([]).text
        self.assertEqual(text, "")
