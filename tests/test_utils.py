import random
import unittest

import cohere.utils as utils


class TestUtils(unittest.TestCase):

    def test_sorts_and_restores_order(self):

        def key(text):
            return len(text)

        texts = ['a' * i for i in range(100)]
        random.shuffle(texts)

        texts_sorted, indices = utils.sort_with_indices(texts, key=key)
        self.assertEqual(texts_sorted, sorted(texts, key=key))
        self.assertEqual(texts, utils.restore_order(texts_sorted, indices))
