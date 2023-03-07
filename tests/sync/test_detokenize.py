import unittest

import cohere

from utils import get_api_key

co = cohere.Client(get_api_key())


class TestDetokenize(unittest.TestCase):

    def test_success(self):
        resp = co.detokenize([10104, 12221, 974, 514, 34])
        text = resp.text
        self.assertEqual(text, "detokenize me!")
        self.assertTrue(resp.meta)
        self.assertTrue(resp.meta["api_version"])
        self.assertTrue(resp.meta["api_version"]["version"])

    def test_success_batched(self):
        _batch_size = 10
        texts = co.batch_detokenize([[10104, 12221, 974, 514, 34]] * _batch_size)
        results = []
        for text in texts:
            results.append(str(text))
        self.assertEqual(results, ["detokenize me!"] * _batch_size)

    def test_empty_tokens(self):
        text = co.detokenize([]).text
        self.assertEqual(text, "")
