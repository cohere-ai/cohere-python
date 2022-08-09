import unittest

import cohere

from utils import get_api_key

co = cohere.Client(get_api_key())


class TestModerate(unittest.TestCase):

    def test_success(self):
        moderations = co.moderate(inputs=['I Love Cohere!'])
        self.assertEqual(len(moderations), 1)

    def test_invalid_text(self):
        with self.assertRaises(cohere.CohereError):
            co.moderate(inputs=[])
