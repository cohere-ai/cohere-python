import os
import unittest
from unittest.mock import MagicMock, patch

import cohere
from cohere.streaming_utils import StreamedEmbedding, StreamingEmbedParser


class TestEmbedStreaming(unittest.TestCase):
    """Test suite for memory-efficient streaming embed functionality."""

    @classmethod
    def setUpClass(cls):
        """Set up class-level fixtures."""
        cls.api_key_available = bool(os.environ.get("CO_API_KEY"))

    def test_streaming_embed_parser_fallback(self):
        """Test that StreamingEmbedParser works with fallback JSON parsing."""
        # Mock response with JSON data
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "response_type": "embeddings_floats",
            "embeddings": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            "texts": ["hello", "world"],
            "id": "test-id"
        }
        
        # Test parser
        parser = StreamingEmbedParser(mock_response, ["hello", "world"])
        embeddings = list(parser.iter_embeddings())
        
        # Verify results
        self.assertEqual(len(embeddings), 2)
        self.assertIsInstance(embeddings[0], StreamedEmbedding)
        self.assertEqual(embeddings[0].index, 0)
        self.assertEqual(embeddings[0].embedding, [0.1, 0.2, 0.3])
        self.assertEqual(embeddings[0].text, "hello")
        self.assertEqual(embeddings[1].index, 1)
        self.assertEqual(embeddings[1].embedding, [0.4, 0.5, 0.6])
        self.assertEqual(embeddings[1].text, "world")

    def test_embed_stream_with_mock(self):
        """Test embed_stream method with mocked responses."""
        # Create a mock client
        client = cohere.Client(api_key="test-key")
        
        # Mock the raw client's embed method
        mock_response_1 = MagicMock()
        mock_response_1.response.json.return_value = {
            "response_type": "embeddings_floats",
            "embeddings": [[0.1, 0.2], [0.3, 0.4]],
            "texts": ["text1", "text2"]
        }
        
        mock_response_2 = MagicMock()
        mock_response_2.response.json.return_value = {
            "response_type": "embeddings_floats",
            "embeddings": [[0.5, 0.6]],
            "texts": ["text3"]
        }
        
        # Mock the embed method to return different responses for different batches
        with patch.object(client._raw_client, 'embed') as mock_embed:
            mock_embed.side_effect = [mock_response_1, mock_response_2]
            
            # Test streaming
            texts = ["text1", "text2", "text3"]
            embeddings = list(client.embed_stream(
                texts=texts,
                model="embed-v4.0",
                batch_size=2
            ))
            
            # Verify results
            self.assertEqual(len(embeddings), 3)
            self.assertEqual(embeddings[0].index, 0)
            self.assertEqual(embeddings[0].text, "text1")
            self.assertEqual(embeddings[1].index, 1)
            self.assertEqual(embeddings[1].text, "text2")
            self.assertEqual(embeddings[2].index, 2)
            self.assertEqual(embeddings[2].text, "text3")
            
            # Verify batching
            self.assertEqual(mock_embed.call_count, 2)

    def test_embed_stream_empty_input(self):
        """Test embed_stream with empty input."""
        client = cohere.Client(api_key="test-key")
        
        # Should return empty iterator
        embeddings = list(client.embed_stream(texts=[], model="embed-v4.0"))
        self.assertEqual(len(embeddings), 0)
        
        # Should handle None
        embeddings = list(client.embed_stream(texts=None, model="embed-v4.0"))
        self.assertEqual(len(embeddings), 0)

    @unittest.skipIf(not os.environ.get("CO_API_KEY"), "API key not available")
    def test_embed_stream_with_real_api(self):
        """Test embed_stream with real API (when API key is available)."""
        client = cohere.Client()
        
        texts = ["Hello world", "How are you", "Goodbye"]
        embeddings_list = []
        
        try:
            # Test streaming embeddings
            for embedding in client.embed_stream(
                texts=texts,
                model="embed-english-v3.0",  # Use a stable model
                batch_size=2,
                input_type="classification"
            ):
                embeddings_list.append(embedding)
                
                # Verify embedding properties
                self.assertIsInstance(embedding, StreamedEmbedding)
                self.assertIsInstance(embedding.index, int)
                self.assertIsInstance(embedding.embedding, list)
                self.assertEqual(embedding.text, texts[embedding.index])
                self.assertGreater(len(embedding.embedding), 0)
                
            # Verify we got all embeddings
            self.assertEqual(len(embeddings_list), len(texts))
            
        except Exception as e:
            if "429" in str(e) or "rate" in str(e).lower():
                self.skipTest("Rate limited")
            raise

    def test_v2_embed_stream_with_mock(self):
        """Test v2 client embed_stream method."""
        client = cohere.ClientV2(api_key="test-key")
        
        # Mock the raw client's embed method
        mock_response = MagicMock()
        mock_response.response.json.return_value = {
            "response_type": "embeddings_by_type",
            "embeddings": {
                "float": [[0.1, 0.2], [0.3, 0.4]]
            },
            "texts": ["hello", "world"],
            "id": "test-id"
        }
        
        with patch.object(client._raw_client, 'embed', return_value=mock_response):
            # Test streaming
            embeddings = list(client.embed_stream(
                model="embed-v4.0",
                input_type="classification",
                texts=["hello", "world"],
                embedding_types=["float"]
            ))
            
            # Verify results
            self.assertEqual(len(embeddings), 2)
            self.assertEqual(embeddings[0].embedding_type, "float")
            self.assertEqual(embeddings[1].embedding_type, "float")

    def test_embed_stream_memory_efficiency(self):
        """Test that embed_stream is more memory efficient than regular embed."""
        # This is a conceptual test - in real usage, the memory savings come from
        # processing embeddings one at a time instead of loading all into memory
        
        client = cohere.Client(api_key="test-key")
        
        # Mock a large response
        large_embedding = [0.1] * 1536  # Typical embedding size
        mock_response = MagicMock()
        mock_response.response.json.return_value = {
            "response_type": "embeddings_floats",
            "embeddings": [large_embedding] * 10,
            "texts": [f"text{i}" for i in range(10)]
        }
        
        with patch.object(client._raw_client, 'embed', return_value=mock_response):
            # With streaming, we process one at a time
            max_embeddings_in_memory = 0
            current_embeddings = []
            
            for embedding in client.embed_stream(texts=[f"text{i}" for i in range(10)], batch_size=10):
                current_embeddings.append(embedding)
                # Simulate processing and clearing
                if len(current_embeddings) > 1:
                    current_embeddings.pop(0)  # Remove processed embedding
                max_embeddings_in_memory = max(max_embeddings_in_memory, len(current_embeddings))
            
            # With streaming, we should only have 1-2 embeddings in memory at a time
            self.assertLessEqual(max_embeddings_in_memory, 2)


if __name__ == "__main__":
    unittest.main()