import unittest

from utils import get_api_key

import cohere

co = cohere.Client(get_api_key())


class TestDetokenize(unittest.TestCase):
    def test_detokenize_success(self):
        resp = co.detokenize([10104, 12221, 974, 514, 34], model="base")
        text = resp.text
        self.assertEqual(text, "detokenize me!")
        self.assertTrue(resp.meta)
        self.assertTrue(resp.meta["api_version"])
        self.assertTrue(resp.meta["api_version"]["version"])

    def test_detokenize_batched(self):
        _batch_size = 3
        texts = co.batch_detokenize([[10104, 12221, 974, 514, 34]] * _batch_size, model="base")
        results = []
        for text in texts:
            results.append(str(text))
        self.assertEqual(results, ["detokenize me!"] * _batch_size)

    def test_empty_tokens(self):
        text = co.detokenize([]).text
        self.assertEqual(text, "")
