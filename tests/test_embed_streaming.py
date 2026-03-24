"""Tests for memory-efficient embed_stream functionality.

All embed_stream code lives in manually maintained files (.fernignore protected):
- src/cohere/client.py — Client.embed_stream()
- src/cohere/manually_maintained/streaming_embed.py — StreamedEmbedding, extraction helpers
"""

import unittest

from cohere.manually_maintained.streaming_embed import (
    StreamedEmbedding,
    extract_embeddings_from_response,
)
from cohere.config import embed_stream_batch_size


class TestStreamedEmbedding(unittest.TestCase):
    """Test the StreamedEmbedding dataclass."""

    def test_creation(self):
        emb = StreamedEmbedding(index=0, embedding=[0.1, 0.2], embedding_type="float", text="hello")
        self.assertEqual(emb.index, 0)
        self.assertEqual(emb.embedding, [0.1, 0.2])
        self.assertEqual(emb.embedding_type, "float")
        self.assertEqual(emb.text, "hello")

    def test_text_optional(self):
        emb = StreamedEmbedding(index=0, embedding=[0.1], embedding_type="float")
        self.assertIsNone(emb.text)


class TestExtractEmbeddings(unittest.TestCase):
    """Test extract_embeddings_from_response for V1 and V2 formats."""

    def test_v1_embeddings_floats(self):
        """V1 embeddings_floats response returns flat float embeddings."""
        response = {
            "response_type": "embeddings_floats",
            "embeddings": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        }
        results = list(extract_embeddings_from_response(response, ["hello", "world"]))

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].index, 0)
        self.assertEqual(results[0].embedding, [0.1, 0.2, 0.3])
        self.assertEqual(results[0].embedding_type, "float")
        self.assertEqual(results[0].text, "hello")
        self.assertEqual(results[1].index, 1)
        self.assertEqual(results[1].text, "world")

    def test_v1_embeddings_by_type(self):
        """V1 embeddings_by_type response returns typed embeddings."""
        response = {
            "response_type": "embeddings_by_type",
            "embeddings": {
                "float_": [[0.1, 0.2], [0.3, 0.4]],
                "int8": [[1, 2], [3, 4]],
            },
        }
        results = list(extract_embeddings_from_response(response, ["a", "b"]))

        # 2 texts * 2 types = 4 embeddings
        self.assertEqual(len(results), 4)
        float_results = [r for r in results if r.embedding_type == "float"]
        int8_results = [r for r in results if r.embedding_type == "int8"]
        self.assertEqual(len(float_results), 2)
        self.assertEqual(len(int8_results), 2)

    def test_v2_response_format(self):
        """V2 response (no response_type) returns dict embeddings."""
        response = {
            "embeddings": {
                "float_": [[0.1, 0.2], [0.3, 0.4]],
            },
        }
        results = list(extract_embeddings_from_response(response, ["x", "y"]))

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].embedding_type, "float")
        self.assertEqual(results[0].text, "x")

    def test_global_offset(self):
        """Global offset adjusts indices for batched processing."""
        response = {
            "response_type": "embeddings_floats",
            "embeddings": [[0.1], [0.2]],
        }
        results = list(extract_embeddings_from_response(response, ["c", "d"], global_offset=100))

        self.assertEqual(results[0].index, 100)
        self.assertEqual(results[1].index, 101)

    def test_empty_embeddings(self):
        """Empty response yields nothing."""
        response = {"response_type": "embeddings_floats", "embeddings": []}
        results = list(extract_embeddings_from_response(response, []))
        self.assertEqual(results, [])

    def test_texts_shorter_than_embeddings(self):
        """Text is None when batch_texts runs out."""
        response = {
            "response_type": "embeddings_floats",
            "embeddings": [[0.1], [0.2], [0.3]],
        }
        results = list(extract_embeddings_from_response(response, ["only_one"]))

        self.assertEqual(results[0].text, "only_one")
        self.assertIsNone(results[1].text)
        self.assertIsNone(results[2].text)


class TestBatchSizeConstant(unittest.TestCase):
    """Test that batch_size defaults come from config, not magic numbers."""

    def test_default_batch_size_matches_api_limit(self):
        self.assertEqual(embed_stream_batch_size, 96)


if __name__ == "__main__":
    unittest.main()
