import unittest

import cohere

from utils import get_api_key

co = cohere.Client(get_api_key())


class TestTokenize(unittest.TestCase):

    def test_success(self):
        tokens = co.tokenize('tokenize me!')
        self.assertIsInstance(tokens.tokens, list)
        self.assertIsInstance(tokens.token_strings, list)
        self.assertIsInstance(tokens.length, int)
        self.assertEqual(tokens.length, len(tokens.tokens))
        self.assertEqual(tokens.length, len(tokens.token_strings))

    def test_invalid_text(self):
        with self.assertRaises(cohere.CohereError):
            str(co.tokenize(text=''))
