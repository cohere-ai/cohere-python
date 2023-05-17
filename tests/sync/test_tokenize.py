import unittest

from utils import get_api_key

import cohere

co = cohere.Client(get_api_key())


class TestTokenize(unittest.TestCase):
    def test_tokenize_success(self):
        tokens = co.tokenize("tokenize me!")
        self.assertIsInstance(tokens.tokens, list)
        self.assertIsInstance(tokens.token_strings, list)
        self.assertIsInstance(tokens.length, int)
        self.assertEqual(tokens.length, len(tokens.tokens))
        self.assertEqual(tokens.length, len(tokens.token_strings))
        self.assertTrue(tokens.meta)
        self.assertTrue(tokens.meta["api_version"])
        self.assertTrue(tokens.meta["api_version"]["version"])

    def test_batch_tokenize(self):
        tokens = co.batch_tokenize(["tokenize me!", "tokenize me too!"])
        self.assertEqual(len(tokens), 2)
        self.assertIsInstance(tokens[0].tokens, list)
        self.assertIsInstance(tokens[1].tokens, list)

    def test_invalid_text(self):
        with self.assertRaises(cohere.CohereError):
            str(co.tokenize(text=""))
