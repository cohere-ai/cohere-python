import unittest

from utils import get_api_key

import cohere

API_KEY = get_api_key()
co = cohere.Client(API_KEY)


class TestCodebook(unittest.TestCase):
    def test_success(self):
        compression_codebooks = {
            "PQ32x": (96, 256, 8),
            "PQ48x": (64, 256, 12),
            "PQ64x": (48, 256, 16),
            "PQ96x": (32, 256, 24),
        }
        for cb, (segments, num_centroids, length) in compression_codebooks.items():
            prediction = co.codebook(
                model="multilingual-22-12",
                compression_codebook=cb,
            )
            self.assertIsInstance(prediction.codebook[0], list)
            self.assertIsInstance(prediction.codebook[0][0], list)
            self.assertEqual(len(prediction.codebook), segments)
            self.assertEqual(len(prediction.codebook[0]), num_centroids)
            self.assertEqual(len(prediction.codebook[0][0]), length)
            self.assertTrue(prediction.meta)
            self.assertTrue(prediction.meta["api_version"])
            self.assertTrue(prediction.meta["api_version"]["version"])

    def test_default_model(self):
        prediction = co.codebook(
            model="multilingual-22-12",
        )
        self.assertEqual(len(prediction.codebook), 96)
        self.assertIsInstance(prediction.codebook[0], list)
        self.assertIsInstance(prediction.codebook[0][0], list)
