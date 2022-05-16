
import os
import unittest
import cohere

API_KEY = os.getenv('CO_API_KEY')
assert type(API_KEY)
co = cohere.Client(str(API_KEY))


class TestTokenize(unittest.TestCase):
    def test_success(self):
        tokens = co.tokenize('medium', 'tokenize me!')
        self.assertIsInstance(tokens.tokens, list)
        self.assertIsInstance(tokens.length, int)
        self.assertEqual(tokens.length, len(tokens))

    def test_invalid_text(self):
        with self.assertRaises(cohere.CohereError):
            co.tokenize(model='medium', text='')


if __name__ == '__main__':
    unittest.main()
