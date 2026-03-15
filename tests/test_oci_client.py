"""Integration tests for OCI Generative AI client.

These tests require:
1. OCI SDK installed: pip install oci
2. OCI credentials configured in ~/.oci/config
3. TEST_OCI environment variable set to run
4. OCI_COMPARTMENT_ID environment variable with valid OCI compartment OCID
5. OCI_REGION environment variable (optional, defaults to us-chicago-1)

Run with:
    TEST_OCI=1 OCI_COMPARTMENT_ID=ocid1.compartment.oc1... pytest tests/test_oci_client.py
"""

import os
import sys
import tempfile
import types
import unittest
from unittest.mock import MagicMock, mock_open, patch

import cohere
from cohere.errors import NotFoundError

if "tokenizers" not in sys.modules:
    tokenizers_stub = types.ModuleType("tokenizers")
    tokenizers_stub.Tokenizer = object
    sys.modules["tokenizers"] = tokenizers_stub

if "fastavro" not in sys.modules:
    fastavro_stub = types.ModuleType("fastavro")
    fastavro_stub.parse_schema = lambda schema: schema
    fastavro_stub.reader = lambda *args, **kwargs: iter(())
    fastavro_stub.writer = lambda *args, **kwargs: None
    sys.modules["fastavro"] = fastavro_stub

if "httpx_sse" not in sys.modules:
    httpx_sse_stub = types.ModuleType("httpx_sse")
    httpx_sse_stub.connect_sse = lambda *args, **kwargs: None
    sys.modules["httpx_sse"] = httpx_sse_stub


@unittest.skipIf(os.getenv("TEST_OCI") is None, "TEST_OCI not set")
class TestOciClient(unittest.TestCase):
    """Test OciClient (v1 API) with OCI Generative AI."""

    def setUp(self):
        """Set up OCI client for each test."""
        compartment_id = os.getenv("OCI_COMPARTMENT_ID")
        if not compartment_id:
            self.skipTest("OCI_COMPARTMENT_ID not set")

        region = os.getenv("OCI_REGION", "us-chicago-1")
        profile = os.getenv("OCI_PROFILE", "DEFAULT")

        self.client = cohere.OciClient(
            oci_region=region,
            oci_compartment_id=compartment_id,
            oci_profile=profile,
        )

    def test_embed(self):
        """Test embedding generation with OCI."""
        response = self.client.embed(
            model="embed-english-v3.0",
            texts=["Hello world", "Cohere on OCI"],
            input_type="search_document",
        )

        self.assertIsNotNone(response)
        self.assertIsNotNone(response.embeddings)
        self.assertEqual(len(response.embeddings), 2)
        # Verify embedding dimensions (1024 for embed-english-v3.0)
        self.assertEqual(len(response.embeddings[0]), 1024)

    def test_embed_with_model_prefix(self):
        """Test embedding with 'cohere.' model prefix."""
        response = self.client.embed(
            model="cohere.embed-english-v3.0",
            texts=["Test with prefix"],
            input_type="search_document",
        )

        self.assertIsNotNone(response)
        self.assertIsNotNone(response.embeddings)
        self.assertEqual(len(response.embeddings), 1)

    @unittest.skip(
        "OCI on-demand models don't support multiple embedding types in a single call. "
        "The embedding_types parameter in OCI accepts a single value, not a list."
    )
    def test_embed_multiple_types(self):
        """Test embedding with multiple embedding types."""
        response = self.client.embed(
            model="embed-english-v3.0",
            texts=["Multi-type test"],
            input_type="search_document",
            embedding_types=["float", "int8"],
        )

        self.assertIsNotNone(response)
        self.assertIsNotNone(response.embeddings)

    def test_chat(self):
        """Test chat with OCI."""
        response = self.client.chat(
            model="command-r-08-2024",
            message="What is 2+2? Answer with just the number.",
        )

        self.assertIsNotNone(response)
        self.assertIsNotNone(response.text)
        self.assertIn("4", response.text)

    def test_chat_with_history(self):
        """Test chat with conversation history."""
        response = self.client.chat(
            model="command-r-08-2024",
            message="What was my previous question?",
            chat_history=[
                {"role": "USER", "message": "What is the capital of France?"},
                {"role": "CHATBOT", "message": "The capital of France is Paris."},
            ],
        )

        self.assertIsNotNone(response)
        self.assertIsNotNone(response.text)

    def test_chat_stream(self):
        """Test streaming chat with OCI."""
        events = []
        for event in self.client.chat_stream(
            model="command-r-08-2024",
            message="Count from 1 to 3.",
        ):
            events.append(event)

        self.assertTrue(len(events) > 0)
        # Verify we received text generation events
        text_events = [e for e in events if hasattr(e, "text") and e.text]
        self.assertTrue(len(text_events) > 0)

    @unittest.skip(
        "OCI TEXT_GENERATION models are finetune base models, not available via on-demand inference. "
        "Only CHAT models (command-r, command-a) support on-demand inference on OCI."
    )
    def test_generate(self):
        """Test text generation with OCI."""
        response = self.client.generate(
            model="command-r-08-2024",
            prompt="Write a haiku about clouds.",
            max_tokens=100,
        )

        self.assertIsNotNone(response)
        self.assertIsNotNone(response.generations)
        self.assertTrue(len(response.generations) > 0)
        self.assertIsNotNone(response.generations[0].text)

    @unittest.skip(
        "OCI TEXT_GENERATION models are finetune base models, not available via on-demand inference. "
        "Only CHAT models (command-r, command-a) support on-demand inference on OCI."
    )
    def test_generate_stream(self):
        """Test streaming text generation with OCI."""
        events = []
        for event in self.client.generate_stream(
            model="command-r-08-2024",
            prompt="Say hello",
            max_tokens=20,
        ):
            events.append(event)

        self.assertTrue(len(events) > 0)

    @unittest.skip(
        "OCI TEXT_RERANK models are base models, not available via on-demand inference. "
        "These models require fine-tuning and deployment before use on OCI."
    )
    def test_rerank(self):
        """Test reranking with OCI."""
        query = "What is the capital of France?"
        documents = [
            "Paris is the capital of France.",
            "London is the capital of England.",
            "Berlin is the capital of Germany.",
        ]

        response = self.client.rerank(
            model="rerank-english-v3.1",
            query=query,
            documents=documents,
            top_n=2,
        )

        self.assertIsNotNone(response)
        self.assertIsNotNone(response.results)
        self.assertEqual(len(response.results), 2)
        # First result should be the Paris document
        self.assertEqual(response.results[0].index, 0)
        self.assertGreater(response.results[0].relevance_score, 0.5)


@unittest.skipIf(os.getenv("TEST_OCI") is None, "TEST_OCI not set")
class TestOciClientV2(unittest.TestCase):
    """Test OciClientV2 (v2 API) with OCI Generative AI."""

    def setUp(self):
        """Set up OCI v2 client for each test."""
        compartment_id = os.getenv("OCI_COMPARTMENT_ID")
        if not compartment_id:
            self.skipTest("OCI_COMPARTMENT_ID not set")

        region = os.getenv("OCI_REGION", "us-chicago-1")
        profile = os.getenv("OCI_PROFILE", "DEFAULT")

        self.client = cohere.OciClientV2(
            oci_region=region,
            oci_compartment_id=compartment_id,
            oci_profile=profile,
        )

    def test_embed_v2(self):
        """Test embedding with v2 client."""
        response = self.client.embed(
            model="embed-english-v3.0",
            texts=["Hello from v2", "Second text"],
            input_type="search_document",
        )

        self.assertIsNotNone(response)
        self.assertIsNotNone(response.embeddings)
        # V2 returns embeddings as a dict with "float" key
        self.assertIsNotNone(response.embeddings.float_)
        self.assertEqual(len(response.embeddings.float_), 2)
        # Verify embedding dimensions (1024 for embed-english-v3.0)
        self.assertEqual(len(response.embeddings.float_[0]), 1024)

    def test_embed_with_model_prefix_v2(self):
        """Test embedding with 'cohere.' model prefix on v2 client."""
        response = self.client.embed(
            model="cohere.embed-english-v3.0",
            texts=["Test with prefix"],
            input_type="search_document",
        )

        self.assertIsNotNone(response)
        self.assertIsNotNone(response.embeddings)
        self.assertIsNotNone(response.embeddings.float_)
        self.assertEqual(len(response.embeddings.float_), 1)

    def test_chat_v2(self):
        """Test chat with v2 client."""
        response = self.client.chat(
            model="command-a-03-2025",
            messages=[{"role": "user", "content": "Say hello"}],
        )

        self.assertIsNotNone(response)
        self.assertIsNotNone(response.message)

    @unittest.skip(
        "Command A Reasoning model (command-a-reasoning-08-2025) may not be available in all regions. "
        "Enable this test when the reasoning model is available in your OCI region."
    )
    def test_chat_v2_with_thinking(self):
        """Test chat with thinking parameter for Command A Reasoning model."""
        from cohere.types import Thinking

        response = self.client.chat(
            model="command-a-reasoning-08-2025",
            messages=[{"role": "user", "content": "What is 15 * 27? Think step by step."}],
            thinking=Thinking(type="enabled", token_budget=5000),
        )

        self.assertIsNotNone(response)
        self.assertIsNotNone(response.message)
        # The response should contain content (may include thinking content)
        self.assertIsNotNone(response.message.content)

    @unittest.skip(
        "Command A Reasoning model (command-a-reasoning-08-2025) may not be available in all regions. "
        "Enable this test when the reasoning model is available in your OCI region."
    )
    def test_chat_stream_v2_with_thinking(self):
        """Test streaming chat with thinking parameter for Command A Reasoning model."""
        from cohere.types import Thinking

        events = []
        for event in self.client.chat_stream(
            model="command-a-reasoning-08-2025",
            messages=[{"role": "user", "content": "What is 15 * 27? Think step by step."}],
            thinking=Thinking(type="enabled", token_budget=5000),
        ):
            events.append(event)

        self.assertTrue(len(events) > 0)
        # Verify we received content-delta events
        content_delta_events = [e for e in events if hasattr(e, "type") and e.type == "content-delta"]
        self.assertTrue(len(content_delta_events) > 0)

    def test_chat_stream_v2(self):
        """Test streaming chat with v2 client."""
        events = []
        for event in self.client.chat_stream(
            model="command-a-03-2025",
            messages=[{"role": "user", "content": "Count from 1 to 3"}],
        ):
            events.append(event)

        self.assertTrue(len(events) > 0)
        # Verify we received content-delta events with text
        content_delta_events = [e for e in events if hasattr(e, "type") and e.type == "content-delta"]
        self.assertTrue(len(content_delta_events) > 0)

        # Verify we can extract text from events
        full_text = ""
        for event in events:
            if (
                hasattr(event, "delta")
                and event.delta
                and hasattr(event.delta, "message")
                and event.delta.message
                and hasattr(event.delta.message, "content")
                and event.delta.message.content
                and hasattr(event.delta.message.content, "text")
                and event.delta.message.content.text is not None
            ):
                full_text += event.delta.message.content.text

        # Should have received some text
        self.assertTrue(len(full_text) > 0)

    @unittest.skip(
        "OCI TEXT_RERANK models are base models, not available via on-demand inference. "
        "These models require fine-tuning and deployment before use on OCI."
    )
    def test_rerank_v2(self):
        """Test reranking with v2 client."""
        response = self.client.rerank(
            model="rerank-english-v3.1",
            query="What is AI?",
            documents=["AI is artificial intelligence.", "AI is not natural."],
            top_n=1,
        )

        self.assertIsNotNone(response)
        self.assertIsNotNone(response.results)


@unittest.skipIf(os.getenv("TEST_OCI") is None, "TEST_OCI not set")
class TestOciClientAuthentication(unittest.TestCase):
    """Test different OCI authentication methods."""

    def test_config_file_auth(self):
        """Test authentication using OCI config file."""
        compartment_id = os.getenv("OCI_COMPARTMENT_ID")
        if not compartment_id:
            self.skipTest("OCI_COMPARTMENT_ID not set")

        # Use API_KEY_AUTH profile (DEFAULT may be session-based)
        profile = os.getenv("OCI_PROFILE", "API_KEY_AUTH")
        client = cohere.OciClient(
            oci_region="us-chicago-1",
            oci_compartment_id=compartment_id,
            oci_profile=profile,
        )

        # Test with a simple embed call
        response = client.embed(
            model="embed-english-v3.0",
            texts=["Auth test"],
            input_type="search_document",
        )

        self.assertIsNotNone(response)
        self.assertIsNotNone(response.embeddings)

    def test_custom_profile_auth(self):
        """Test authentication using custom OCI profile."""
        compartment_id = os.getenv("OCI_COMPARTMENT_ID")
        profile = os.getenv("OCI_PROFILE", "DEFAULT")

        if not compartment_id:
            self.skipTest("OCI_COMPARTMENT_ID not set")

        client = cohere.OciClient(
            oci_profile=profile,
            oci_region="us-chicago-1",
            oci_compartment_id=compartment_id,
        )

        response = client.embed(
            model="embed-english-v3.0",
            texts=["Profile auth test"],
            input_type="search_document",
        )

        self.assertIsNotNone(response)


@unittest.skipIf(os.getenv("TEST_OCI") is None, "TEST_OCI not set")
class TestOciClientErrors(unittest.TestCase):
    """Test error handling in OCI client."""

    def test_missing_compartment_id(self):
        """Test error when compartment ID is missing."""
        with self.assertRaises(TypeError):
            cohere.OciClient(
                oci_region="us-chicago-1",
                # Missing oci_compartment_id
            )

    @unittest.skip("Region is available in config file for current test environment")
    def test_missing_region(self):
        """Test error when region is missing and not in config."""
        # This test assumes no region in config file
        # If config has region, this will pass, so we just check it doesn't crash
        try:
            client = cohere.OciClient(
                oci_compartment_id="ocid1.compartment.oc1...",
            )
            # If this succeeds, region was in config
            self.assertIsNotNone(client)
        except ValueError as e:
            # Expected if no region in config
            self.assertIn("region", str(e).lower())

    def test_invalid_model(self):
        """Test error handling with invalid model."""
        compartment_id = os.getenv("OCI_COMPARTMENT_ID")
        if not compartment_id:
            self.skipTest("OCI_COMPARTMENT_ID not set")

        profile = os.getenv("OCI_PROFILE", "API_KEY_AUTH")
        client = cohere.OciClient(
            oci_region="us-chicago-1",
            oci_compartment_id=compartment_id,
            oci_profile=profile,
        )

        # OCI should return an error for invalid model
        with self.assertRaises(Exception):
            client.embed(
                model="invalid-model-name",
                texts=["Test"],
                input_type="search_document",
            )


@unittest.skipIf(os.getenv("TEST_OCI") is None, "TEST_OCI not set")
class TestOciClientModels(unittest.TestCase):
    """Test different Cohere models on OCI."""

    def setUp(self):
        """Set up OCI client for each test."""
        compartment_id = os.getenv("OCI_COMPARTMENT_ID")
        if not compartment_id:
            self.skipTest("OCI_COMPARTMENT_ID not set")

        region = os.getenv("OCI_REGION", "us-chicago-1")
        profile = os.getenv("OCI_PROFILE", "DEFAULT")

        self.client = cohere.OciClient(
            oci_region=region,
            oci_compartment_id=compartment_id,
            oci_profile=profile,
        )

    def test_embed_english_v3(self):
        """Test embed-english-v3.0 model."""
        response = self.client.embed(
            model="embed-english-v3.0",
            texts=["Test"],
            input_type="search_document",
        )
        self.assertIsNotNone(response.embeddings)
        self.assertEqual(len(response.embeddings[0]), 1024)

    def test_embed_light_v3(self):
        """Test embed-english-light-v3.0 model."""
        try:
            response = self.client.embed(
                model="embed-english-light-v3.0",
                texts=["Test"],
                input_type="search_document",
            )
        except NotFoundError as exc:
            if "embed-english-light-v3.0" in str(exc):
                self.skipTest("embed-english-light-v3.0 is not available in the current OCI region/profile")
            raise
        self.assertIsNotNone(response.embeddings)
        self.assertEqual(len(response.embeddings[0]), 384)

    def test_embed_multilingual_v3(self):
        """Test embed-multilingual-v3.0 model."""
        response = self.client.embed(
            model="embed-multilingual-v3.0",
            texts=["Test"],
            input_type="search_document",
        )
        self.assertIsNotNone(response.embeddings)
        self.assertEqual(len(response.embeddings[0]), 1024)

    def test_command_r_plus(self):
        """Test command-r-plus model for chat."""
        response = self.client.chat(
            model="command-r-08-2024",
            message="Hello",
        )
        self.assertIsNotNone(response.text)

    @unittest.skip(
        "OCI TEXT_RERANK models are base models, not available via on-demand inference. "
        "These models require fine-tuning and deployment before use on OCI."
    )
    def test_rerank_v3(self):
        """Test rerank-english-v3.0 model."""
        response = self.client.rerank(
            model="rerank-english-v3.1",
            query="AI",
            documents=["Artificial Intelligence", "Biology"],
        )
        self.assertIsNotNone(response.results)


class TestOciClientTransformations(unittest.TestCase):
    """Unit tests for OCI request/response transformations (no OCI credentials required)."""

    def test_thinking_parameter_transformation(self):
        """Test that thinking parameter is correctly transformed to OCI format."""
        from cohere.oci_client import transform_request_to_oci

        cohere_body = {
            "model": "command-a-reasoning-08-2025",
            "messages": [{"role": "user", "content": "What is 2+2?"}],
            "thinking": {
                "type": "enabled",
                "token_budget": 10000,
            },
        }

        result = transform_request_to_oci("chat", cohere_body, "compartment-123")

        # Verify thinking parameter is transformed with camelCase for OCI API
        chat_request = result["chatRequest"]
        self.assertIn("thinking", chat_request)
        self.assertEqual(chat_request["thinking"]["type"], "ENABLED")
        self.assertEqual(chat_request["thinking"]["tokenBudget"], 10000)  # camelCase for OCI

    def test_thinking_parameter_disabled(self):
        """Test that disabled thinking is correctly transformed."""
        from cohere.oci_client import transform_request_to_oci

        cohere_body = {
            "model": "command-a-reasoning-08-2025",
            "messages": [{"role": "user", "content": "Hello"}],
            "thinking": {
                "type": "disabled",
            },
        }

        result = transform_request_to_oci("chat", cohere_body, "compartment-123")

        chat_request = result["chatRequest"]
        self.assertIn("thinking", chat_request)
        self.assertEqual(chat_request["thinking"]["type"], "DISABLED")
        self.assertNotIn("token_budget", chat_request["thinking"])

    def test_thinking_response_transformation(self):
        """Test that thinking content in response is correctly transformed."""
        from cohere.oci_client import transform_oci_response_to_cohere

        oci_response = {
            "chatResponse": {
                "id": "test-id",
                "message": {
                    "role": "ASSISTANT",
                    "content": [
                        {"type": "THINKING", "thinking": "Let me think about this..."},
                        {"type": "TEXT", "text": "The answer is 4."},
                    ],
                },
                "finishReason": "COMPLETE",
                "usage": {"inputTokens": 10, "completionTokens": 20},
            }
        }

        result = transform_oci_response_to_cohere("chat", oci_response, is_v2=True)

        # Verify content types are lowercased
        self.assertEqual(result["message"]["content"][0]["type"], "thinking")
        self.assertEqual(result["message"]["content"][1]["type"], "text")

    def test_stream_event_thinking_transformation(self):
        """Test that thinking content in stream events is correctly transformed."""
        from cohere.oci_client import transform_stream_event

        # OCI thinking event
        oci_event = {
            "message": {
                "content": [{"type": "THINKING", "thinking": "Reasoning step..."}]
            }
        }

        result = transform_stream_event("chat", oci_event, is_v2=True)

        self.assertEqual(result[0]["type"], "content-delta")
        self.assertIn("thinking", result[0]["delta"]["message"]["content"])
        self.assertEqual(result[0]["delta"]["message"]["content"]["thinking"], "Reasoning step...")

    def test_stream_event_text_transformation(self):
        """Test that text content in stream events is correctly transformed."""
        from cohere.oci_client import transform_stream_event

        # OCI text event
        oci_event = {
            "message": {
                "content": [{"type": "TEXT", "text": "The answer is..."}]
            }
        }

        result = transform_stream_event("chat", oci_event, is_v2=True)

        self.assertEqual(result[0]["type"], "content-delta")
        self.assertIn("text", result[0]["delta"]["message"]["content"])
        self.assertEqual(result[0]["delta"]["message"]["content"]["text"], "The answer is...")

    def test_thinking_parameter_none(self):
        """Test that thinking=None does not crash (issue: null guard)."""
        from cohere.oci_client import transform_request_to_oci

        cohere_body = {
            "model": "command-a-03-2025",
            "messages": [{"role": "user", "content": "Hello"}],
            "thinking": None,  # Explicitly set to None
        }

        # Should not crash with TypeError
        result = transform_request_to_oci("chat", cohere_body, "compartment-123")

        chat_request = result["chatRequest"]
        # thinking should not be in request when None
        self.assertNotIn("thinking", chat_request)

    def test_v2_response_role_lowercased(self):
        """Test that V2 response message role is lowercased."""
        from cohere.oci_client import transform_oci_response_to_cohere

        oci_response = {
            "chatResponse": {
                "id": "test-id",
                "message": {
                    "role": "ASSISTANT",
                    "content": [{"type": "TEXT", "text": "Hello"}],
                },
                "finishReason": "COMPLETE",
                "usage": {"inputTokens": 10, "completionTokens": 20},
            }
        }

        result = transform_oci_response_to_cohere("chat", oci_response, is_v2=True)

        # Role should be lowercased
        self.assertEqual(result["message"]["role"], "assistant")

    def test_v2_response_finish_reason_uppercase(self):
        """Test that V2 response finish_reason stays uppercase."""
        from cohere.oci_client import transform_oci_response_to_cohere

        oci_response = {
            "chatResponse": {
                "id": "test-id",
                "message": {
                    "role": "ASSISTANT",
                    "content": [{"type": "TEXT", "text": "Hello"}],
                },
                "finishReason": "MAX_TOKENS",
                "usage": {"inputTokens": 10, "completionTokens": 20},
            }
        }

        result = transform_oci_response_to_cohere("chat", oci_response, is_v2=True)

        # V2 finish_reason should stay uppercase
        self.assertEqual(result["finish_reason"], "MAX_TOKENS")

    def test_v2_response_tool_calls_conversion(self):
        """Test that V2 response converts toolCalls to tool_calls."""
        from cohere.oci_client import transform_oci_response_to_cohere

        oci_response = {
            "chatResponse": {
                "id": "test-id",
                "message": {
                    "role": "ASSISTANT",
                    "content": [{"type": "TEXT", "text": "I'll help with that."}],
                    "toolCalls": [
                        {
                            "id": "call_123",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": '{"city": "London"}'},
                        }
                    ],
                },
                "finishReason": "TOOL_CALL",
                "usage": {"inputTokens": 10, "completionTokens": 20},
            }
        }

        result = transform_oci_response_to_cohere("chat", oci_response, is_v2=True)

        # toolCalls should be converted to tool_calls
        self.assertIn("tool_calls", result["message"])
        self.assertNotIn("toolCalls", result["message"])
        self.assertEqual(len(result["message"]["tool_calls"]), 1)
        self.assertEqual(result["message"]["tool_calls"][0]["id"], "call_123")

    def test_normalize_model_for_oci(self):
        """Test model name normalization for OCI."""
        from cohere.oci_client import normalize_model_for_oci

        # Plain model name gets cohere. prefix
        self.assertEqual(normalize_model_for_oci("command-a-03-2025"), "cohere.command-a-03-2025")
        # Already prefixed passes through
        self.assertEqual(normalize_model_for_oci("cohere.embed-english-v3.0"), "cohere.embed-english-v3.0")
        # OCID passes through
        self.assertEqual(
            normalize_model_for_oci("ocid1.generativeaimodel.oc1.us-chicago-1.abc"),
            "ocid1.generativeaimodel.oc1.us-chicago-1.abc",
        )

    def test_transform_embed_request(self):
        """Test embed request transformation to OCI format."""
        from cohere.oci_client import transform_request_to_oci

        body = {
            "model": "embed-english-v3.0",
            "texts": ["hello", "world"],
            "input_type": "search_document",
            "truncate": "end",
            "embedding_types": ["float", "int8"],
        }
        result = transform_request_to_oci("embed", body, "compartment-123")

        self.assertEqual(result["inputs"], ["hello", "world"])
        self.assertEqual(result["inputType"], "SEARCH_DOCUMENT")
        self.assertEqual(result["truncate"], "END")
        self.assertEqual(result["embeddingTypes"], ["FLOAT", "INT8"])
        self.assertEqual(result["compartmentId"], "compartment-123")
        self.assertEqual(result["servingMode"]["modelId"], "cohere.embed-english-v3.0")

    def test_transform_embed_request_with_v2_optional_params(self):
        """Test embed request forwards V2-specific optional params."""
        from cohere.oci_client import transform_request_to_oci

        body = {
            "model": "embed-english-v3.0",
            "inputs": [{"content": [{"type": "text", "text": "hello"}]}],
            "input_type": "classification",
            "max_tokens": 256,
            "output_dimension": 512,
            "priority": 42,
        }
        result = transform_request_to_oci("embed", body, "compartment-123")

        self.assertEqual(result["inputs"], body["inputs"])
        self.assertEqual(result["maxTokens"], 256)
        self.assertEqual(result["outputDimension"], 512)
        self.assertEqual(result["priority"], 42)

    def test_transform_embed_request_rejects_images(self):
        """Test embed request fails clearly for unsupported top-level images."""
        from cohere.oci_client import transform_request_to_oci

        with self.assertRaises(ValueError) as ctx:
            transform_request_to_oci(
                "embed",
                {
                    "model": "embed-english-v3.0",
                    "images": ["data:image/png;base64,abc"],
                    "input_type": "classification",
                },
                "compartment-123",
            )

        self.assertIn("top-level 'images' parameter", str(ctx.exception))

    def test_transform_chat_request_optional_params(self):
        """Test chat request transformation includes optional params."""
        from cohere.oci_client import transform_request_to_oci

        body = {
            "model": "command-a-03-2025",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 100,
            "temperature": 0.7,
            "stop_sequences": ["END"],
            "frequency_penalty": 0.5,
            "strict_tools": True,
            "response_format": {"type": "json_object"},
            "logprobs": True,
            "tool_choice": "REQUIRED",
            "priority": 7,
        }
        result = transform_request_to_oci("chat", body, "compartment-123")

        chat_req = result["chatRequest"]
        self.assertEqual(chat_req["maxTokens"], 100)
        self.assertEqual(chat_req["temperature"], 0.7)
        self.assertEqual(chat_req["stopSequences"], ["END"])
        self.assertEqual(chat_req["frequencyPenalty"], 0.5)
        self.assertTrue(chat_req["strictTools"])
        self.assertEqual(chat_req["responseFormat"], {"type": "json_object"})
        self.assertTrue(chat_req["logprobs"])
        self.assertEqual(chat_req["toolChoice"], "REQUIRED")
        self.assertEqual(chat_req["priority"], 7)

    def test_transform_chat_request_tool_message_fields(self):
        """Test tool message fields are converted to OCI names."""
        from cohere.oci_client import transform_request_to_oci

        body = {
            "model": "command-a-03-2025",
            "messages": [
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Use tool"}],
                    "tool_calls": [{"id": "call_1"}],
                    "tool_plan": "Plan",
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": [{"type": "text", "text": "Result"}],
                },
            ],
        }

        result = transform_request_to_oci("chat", body, "compartment-123")
        assistant_message, tool_message = result["chatRequest"]["messages"]
        self.assertEqual(assistant_message["toolCalls"], [{"id": "call_1"}])
        self.assertEqual(assistant_message["toolPlan"], "Plan")
        self.assertEqual(tool_message["toolCallId"], "call_1")

    def test_get_oci_url_known_endpoints(self):
        """Test URL generation for known endpoints."""
        from cohere.oci_client import get_oci_url

        url = get_oci_url("us-chicago-1", "embed")
        self.assertIn("/actions/embedText", url)

        url = get_oci_url("us-chicago-1", "chat")
        self.assertIn("/actions/chat", url)

        url = get_oci_url("us-chicago-1", "chat_stream")
        self.assertIn("/actions/chat", url)

    def test_get_oci_url_unknown_endpoint_raises(self):
        """Test that unknown endpoints raise ValueError instead of producing bad URLs."""
        from cohere.oci_client import get_oci_url

        with self.assertRaises(ValueError) as ctx:
            get_oci_url("us-chicago-1", "unknown_endpoint")
        self.assertIn("not supported", str(ctx.exception))

    def test_load_oci_config_missing_private_key_raises(self):
        """Test that direct credentials without private key raises clear error."""
        from cohere.oci_client import _load_oci_config

        with patch("cohere.oci_client.lazy_oci", return_value=MagicMock()):
            with self.assertRaises(ValueError) as ctx:
                _load_oci_config(
                    auth_type="api_key",
                    config_path=None,
                    profile=None,
                    user_id="ocid1.user.oc1...",
                    fingerprint="xx:xx:xx",
                    tenancy_id="ocid1.tenancy.oc1...",
                    # No private_key_path or private_key_content
                )
            self.assertIn("oci_private_key_path", str(ctx.exception))

    def test_load_oci_config_ignores_inherited_session_auth(self):
        """Test that named API-key profiles do not inherit DEFAULT session auth fields."""
        from cohere.oci_client import _load_oci_config

        config_text = """
[DEFAULT]
security_token_file=/tmp/default-token

[API_KEY_AUTH]
user=ocid1.user.oc1..test
fingerprint=aa:bb
key_file=/tmp/test.pem
tenancy=ocid1.tenancy.oc1..test
region=us-chicago-1
""".strip()

        with tempfile.NamedTemporaryFile("w", delete=False) as config_file:
            config_file.write(config_text)
            config_path = config_file.name

        try:
            mock_oci = MagicMock()
            mock_oci.config.from_file.return_value = {
                "user": "ocid1.user.oc1..test",
                "fingerprint": "aa:bb",
                "key_file": "/tmp/test.pem",
                "tenancy": "ocid1.tenancy.oc1..test",
                "region": "us-chicago-1",
                "security_token_file": "/tmp/default-token",
            }

            with patch("cohere.oci_client.lazy_oci", return_value=mock_oci):
                config = _load_oci_config(
                    auth_type="api_key",
                    config_path=config_path,
                    profile="API_KEY_AUTH",
                )
        finally:
            os.unlink(config_path)

        self.assertNotIn("security_token_file", config)

    def test_session_auth_prefers_security_token_signer(self):
        """Test session-based auth uses SecurityTokenSigner before API key signer."""
        from cohere.oci_client import map_request_to_oci

        mock_oci = MagicMock()
        mock_security_signer = MagicMock()
        mock_oci.signer.load_private_key_from_file.return_value = "private-key"
        mock_oci.auth.signers.SecurityTokenSigner.return_value = mock_security_signer

        with patch("cohere.oci_client.lazy_oci", return_value=mock_oci), patch(
            "builtins.open", mock_open(read_data="session-token")
        ):
            hook = map_request_to_oci(
                oci_config={
                    "user": "ocid1.user.oc1..example",
                    "fingerprint": "xx:xx",
                    "tenancy": "ocid1.tenancy.oc1..example",
                    "security_token_file": "~/.oci/token",
                    "key_file": "~/.oci/key.pem",
                },
                oci_region="us-chicago-1",
                oci_compartment_id="ocid1.compartment.oc1..example",
            )

            request = MagicMock()
            request.url.path = "/v2/embed"
            request.read.return_value = b'{"model":"embed-english-v3.0","texts":["hello"]}'
            request.method = "POST"
            request.extensions = {}

            hook(request)

        mock_oci.auth.signers.SecurityTokenSigner.assert_called_once_with(
            token="session-token",
            private_key="private-key",
        )
        mock_oci.signer.Signer.assert_not_called()

    def test_embed_response_lowercases_embedding_keys(self):
        """Test embed response uses lowercase keys expected by the SDK model."""
        from cohere.oci_client import transform_oci_response_to_cohere

        result = transform_oci_response_to_cohere(
            "embed",
            {
                "id": "embed-id",
                "embeddings": {"FLOAT": [[0.1, 0.2]], "INT8": [[1, 2]]},
                "usage": {"inputTokens": 3, "completionTokens": 7},
            },
            is_v2=True,
        )

        self.assertIn("float", result["embeddings"])
        self.assertIn("int8", result["embeddings"])
        self.assertNotIn("FLOAT", result["embeddings"])
        self.assertEqual(result["meta"]["tokens"]["output_tokens"], 7)

    def test_normalize_model_for_oci_rejects_empty_model(self):
        """Test model normalization fails clearly for empty model names."""
        from cohere.oci_client import normalize_model_for_oci

        with self.assertRaises(ValueError) as ctx:
            normalize_model_for_oci("")
        self.assertIn("non-empty model", str(ctx.exception))

    def test_transform_rerank_request_uses_v2_max_tokens_per_doc(self):
        """Test rerank request forwards the V2 max_tokens_per_doc parameter."""
        from cohere.oci_client import transform_request_to_oci

        result = transform_request_to_oci(
            "rerank",
            {
                "model": "rerank-english-v3.1",
                "query": "query",
                "documents": ["a", "b"],
                "top_n": 2,
                "max_tokens_per_doc": 512,
                "priority": 3,
            },
            "compartment-123",
        )

        self.assertEqual(result["topN"], 2)
        self.assertEqual(result["maxTokensPerDocument"], 512)
        self.assertEqual(result["priority"], 3)

    def test_stream_wrapper_emits_full_event_lifecycle(self):
        """Test that stream emits message-start, content-start, content-delta, content-end, message-end."""
        import json
        from cohere.oci_client import transform_oci_stream_wrapper

        chunks = [
            b'data: {"message": {"content": [{"type": "TEXT", "text": "Hello"}]}}\n',
            b'data: {"message": {"content": [{"type": "TEXT", "text": " world"}]}, "finishReason": "COMPLETE"}\n',
            b'data: [DONE]\n',
        ]

        events = []
        for raw in transform_oci_stream_wrapper(iter(chunks), "chat", is_v2=True):
            line = raw.decode("utf-8").strip()
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))

        event_types = [e["type"] for e in events]
        self.assertEqual(event_types[0], "message-start")
        self.assertEqual(event_types[1], "content-start")
        self.assertEqual(event_types[2], "content-delta")
        self.assertEqual(event_types[3], "content-delta")
        self.assertEqual(event_types[4], "content-end")
        self.assertEqual(event_types[5], "message-end")

        # Verify message-start has id and role
        self.assertIn("id", events[0])
        self.assertEqual(events[0]["delta"]["message"]["role"], "assistant")

        # Verify content-start has index and type
        self.assertEqual(events[1]["index"], 0)
        self.assertEqual(events[1]["delta"]["message"]["content"]["type"], "text")
        self.assertEqual(events[5]["delta"]["finish_reason"], "COMPLETE")

    def test_stream_wrapper_skips_malformed_json_with_warning(self):
        """Test that malformed JSON in SSE stream is skipped (not silently swallowed)."""
        from cohere.oci_client import transform_oci_stream_wrapper

        chunks = [
            b'data: not-valid-json\n',
            b'data: {"message": {"content": [{"type": "TEXT", "text": "hello"}]}}\n',
            b'data: [DONE]\n',
        ]
        events = list(transform_oci_stream_wrapper(iter(chunks), "chat", is_v2=True))
        # Should get message-start + content-start + content-delta + content-end + message-end.
        self.assertEqual(len(events), 5)

    def test_stream_wrapper_skips_message_end_for_empty_v2_stream(self):
        """Test empty V2 streams do not emit message-end without a preceding message-start."""
        from cohere.oci_client import transform_oci_stream_wrapper

        events = list(transform_oci_stream_wrapper(iter([b"data: [DONE]\n"]), "chat", is_v2=True))

        self.assertEqual(events, [])

    def test_v1_stream_wrapper_preserves_finish_reason_in_stream_end(self):
        """Test that V1 stream-end uses the OCI finish reason from the final event."""
        import json
        from cohere.oci_client import transform_oci_stream_wrapper

        chunks = [
            b'data: {"text": "Hello", "isFinished": false}\n',
            b'data: {"text": " world", "isFinished": true, "finishReason": "MAX_TOKENS"}\n',
            b"data: [DONE]\n",
        ]

        events = [
            json.loads(raw.decode("utf-8"))
            for raw in transform_oci_stream_wrapper(iter(chunks), "chat_stream", is_v2=False)
        ]

        self.assertEqual(events[2]["event_type"], "stream-end")
        self.assertEqual(events[2]["response"]["text"], "Hello world")
        self.assertEqual(events[2]["response"]["finish_reason"], "MAX_TOKENS")

    def test_stream_wrapper_raises_on_transform_error(self):
        """Test that transform errors in stream produce OCI-specific error, not opaque httpx error."""
        from cohere.oci_client import transform_oci_stream_wrapper

        # Event with structure that will cause transform_stream_event to fail
        # (message is None, causing TypeError on "content" in None)
        chunks = [
            b'data: {"message": null}\n',
        ]
        with self.assertRaises(RuntimeError) as ctx:
            list(transform_oci_stream_wrapper(iter(chunks), "chat", is_v2=True))
        self.assertIn("OCI stream event transformation failed", str(ctx.exception))

    def test_stream_event_finish_reason_keeps_final_text(self):
        """Test finish events keep final text before content-end."""
        from cohere.oci_client import transform_stream_event

        events = transform_stream_event(
            "chat",
            {
                "message": {"content": [{"type": "TEXT", "text": " world"}]},
                "finishReason": "COMPLETE",
            },
            is_v2=True,
        )

        self.assertEqual(events[0]["type"], "content-delta")
        self.assertEqual(events[0]["delta"]["message"]["content"]["text"], " world")
        self.assertEqual(events[1]["type"], "content-end")
if __name__ == "__main__":
    unittest.main()
