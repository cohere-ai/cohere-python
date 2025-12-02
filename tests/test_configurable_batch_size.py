"""Tests for configurable batch size in embed method."""

import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

import cohere
from cohere import EmbedResponse
from cohere.base_client import AsyncBaseCohere, BaseCohere


class TestConfigurableBatchSize(unittest.TestCase):
    """Test suite for configurable batch size functionality."""

    def setUp(self):
        """Set up test client."""
        self.api_key = "test-key"
        self.client = cohere.Client(api_key=self.api_key)
    
    def test_custom_batch_size(self):
        """Test that custom batch_size parameter is used correctly."""
        texts = ["text1", "text2", "text3", "text4", "text5"]
        custom_batch_size = 2
        
        # Mock the base embed method
        with patch.object(BaseCohere, 'embed') as mock_embed:
            # Create mock responses
            mock_responses = []
            expected_batches = [
                ["text1", "text2"],
                ["text3", "text4"],
                ["text5"]
            ]
            
            for i, batch in enumerate(expected_batches):
                mock_response = MagicMock(spec=EmbedResponse)
                mock_response.embeddings = [[0.1 * (i + 1)] * 10] * len(batch)
                mock_response.texts = batch
                mock_response.id = f"test-{i}"
                mock_response.response_type = "embeddings_floats"
                mock_response.meta = None  # Add meta attribute
                mock_responses.append(mock_response)
            
            mock_embed.side_effect = mock_responses
            
            # Call embed with custom batch_size
            response = self.client.embed(
                texts=texts,
                model="embed-english-v3.0",
                batch_size=custom_batch_size
            )
            
            # Verify the method was called with correct batch sizes
            self.assertEqual(mock_embed.call_count, 3)
            
            # Verify each call had the correct batch (order may vary due to executor)
            calls = mock_embed.call_args_list
            actual_batches = [call_args[1]['texts'] for call_args in calls]
            # Sort both lists to compare regardless of order
            actual_batches.sort(key=lambda x: x[0])
            expected_batches.sort(key=lambda x: x[0])
            self.assertEqual(actual_batches, expected_batches)
    
    def test_default_batch_size(self):
        """Test that default batch_size is used when not specified."""
        # Create a large list of texts that exceeds default batch size
        texts = [f"text{i}" for i in range(100)]
        
        with patch.object(BaseCohere, 'embed') as mock_embed:
            # Create a mock response
            mock_response = MagicMock(spec=EmbedResponse)
            mock_response.embeddings = [[0.1] * 10] * 96  # Default batch size
            mock_response.texts = texts[:96]
            mock_response.id = "test-1"
            mock_response.response_type = "embeddings_floats"
            mock_response.meta = None
            
            mock_embed.return_value = mock_response
            
            # Call embed without batch_size parameter
            response = self.client.embed(
                texts=texts,
                model="embed-english-v3.0"
            )
            
            # Should use default batch size of 96
            self.assertEqual(mock_embed.call_count, 2)  # 100 texts / 96 batch size = 2 calls
    
    def test_batch_size_edge_cases(self):
        """Test edge cases for batch_size parameter."""
        texts = ["text1", "text2", "text3"]
        
        # Test batch_size = 1
        with patch.object(BaseCohere, 'embed') as mock_embed:
            mock_response = MagicMock(spec=EmbedResponse)
            mock_response.embeddings = [[0.1] * 10]
            mock_response.texts = ["text1"]
            mock_response.id = "test-1"
            mock_response.response_type = "embeddings_floats"
            mock_response.meta = None
            mock_embed.return_value = mock_response
            
            response = self.client.embed(
                texts=texts,
                model="embed-english-v3.0",
                batch_size=1
            )
            
            # Should make 3 calls with batch_size=1
            self.assertEqual(mock_embed.call_count, 3)
        
        # Test batch_size larger than input
        with patch.object(BaseCohere, 'embed') as mock_embed:
            mock_response = MagicMock(spec=EmbedResponse)
            mock_response.embeddings = [[0.1] * 10] * 3
            mock_response.texts = texts
            mock_response.id = "test-1"
            mock_response.response_type = "embeddings_floats"
            mock_response.meta = None
            mock_embed.return_value = mock_response
            
            response = self.client.embed(
                texts=texts,
                model="embed-english-v3.0",
                batch_size=100  # Larger than input
            )
            
            # Should make only 1 call
            self.assertEqual(mock_embed.call_count, 1)
    
    def test_custom_max_workers(self):
        """Test that custom max_workers creates a new ThreadPoolExecutor."""
        texts = ["text1", "text2", "text3", "text4"]
        custom_max_workers = 2
        
        # Track executor usage
        original_executor = self.client._executor
        executors_used = []
        
        def track_executor(*args, **kwargs):
            # Get the executor from the current frame
            import inspect
            frame = inspect.currentframe()
            if frame and frame.f_back and frame.f_back.f_locals:
                executor = frame.f_back.f_locals.get('executor')
                if executor:
                    executors_used.append(executor)
            mock_response = MagicMock(spec=EmbedResponse)
            mock_response.embeddings = [[0.1] * 10]
            mock_response.texts = ["text1"]
            mock_response.id = "test-1" 
            mock_response.response_type = "embeddings_floats"
            mock_response.meta = None
            return mock_response
        
        with patch.object(BaseCohere, 'embed', side_effect=track_executor):
            with patch('cohere.client.ThreadPoolExecutor') as mock_executor_class:
                # Create a mock executor instance
                mock_executor = MagicMock(spec=ThreadPoolExecutor)
                # Create proper mock responses for map
                mock_responses = []
                for i in range(1):  # Only one batch since batch_size defaults to 96
                    mock_resp = MagicMock(spec=EmbedResponse)
                    mock_resp.embeddings = [[0.1] * 10] * 4
                    mock_resp.texts = texts
                    mock_resp.id = "test-1"
                    mock_resp.response_type = "embeddings_floats"
                    mock_resp.meta = None
                    mock_responses.append(mock_resp)
                mock_executor.map.return_value = mock_responses
                mock_executor_class.return_value = mock_executor
                
                response = self.client.embed(
                    texts=texts,
                    model="embed-english-v3.0",
                    max_workers=custom_max_workers
                )
                
                # Verify ThreadPoolExecutor was created with correct max_workers
                mock_executor_class.assert_called_once_with(max_workers=custom_max_workers)
                # Verify shutdown was called
                mock_executor.shutdown.assert_called_once_with(wait=False)
    
    def test_no_batching_ignores_parameters(self):
        """Test that batch_size is ignored when batching=False."""
        texts = ["text1", "text2"]
        
        with patch.object(BaseCohere, 'embed') as mock_embed:
            mock_response = MagicMock(spec=EmbedResponse)
            mock_response.embeddings = [[0.1] * 10] * 2
            mock_response.texts = texts
            mock_response.id = "test-1"
            mock_response.response_type = "embeddings_floats"
            mock_response.meta = None
            mock_embed.return_value = mock_response
            
            response = self.client.embed(
                texts=texts,
                model="embed-english-v3.0",
                batching=False,
                batch_size=1  # Should be ignored
            )
            
            # Should make only 1 call with all texts
            self.assertEqual(mock_embed.call_count, 1)
            call_args = mock_embed.call_args
            _, kwargs = call_args
            self.assertEqual(kwargs['texts'], texts)


class TestAsyncConfigurableBatchSize(unittest.IsolatedAsyncioTestCase):
    """Test suite for async configurable batch size functionality."""

    async def asyncSetUp(self):
        """Set up async test client."""
        self.api_key = "test-key"
        self.client = cohere.AsyncClient(api_key=self.api_key)
    
    async def test_async_custom_batch_size(self):
        """Test that custom batch_size parameter works in async client."""
        texts = ["text1", "text2", "text3", "text4", "text5"]
        custom_batch_size = 2
        
        # Mock the base embed method
        with patch.object(AsyncBaseCohere, 'embed') as mock_embed:
            # Create mock responses
            mock_responses = []
            expected_batches = [
                ["text1", "text2"],
                ["text3", "text4"],
                ["text5"]
            ]
            
            for i, batch in enumerate(expected_batches):
                mock_response = MagicMock(spec=EmbedResponse)
                mock_response.embeddings = [[0.1 * (i + 1)] * 10] * len(batch)
                mock_response.texts = batch
                mock_response.id = f"test-{i}"
                mock_response.response_type = "embeddings_floats"
                mock_response.meta = None  # Add meta attribute
                mock_responses.append(mock_response)
            
            mock_embed.side_effect = mock_responses
            
            # Call embed with custom batch_size
            response = await self.client.embed(
                texts=texts,
                model="embed-english-v3.0",
                batch_size=custom_batch_size
            )
            
            # Verify the method was called with correct batch sizes
            self.assertEqual(mock_embed.call_count, 3)


if __name__ == "__main__":
    unittest.main()