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
import unittest

import cohere
from cohere.errors import NotFoundError


def get_test_oci_model(env_var: str, default: str) -> str:
    value = os.getenv(env_var)
    if value:
        return value
    return default


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
        self.chat_model = get_test_oci_model("OCI_V2_CHAT_MODEL", "command-a-03-2025")
        self.embed_model = get_test_oci_model("OCI_V2_EMBED_MODEL", "embed-english-v3.0")

    def test_embed_v2(self):
        """Test embedding with v2 client."""
        response = self.client.embed(
            model=self.embed_model,
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
            model=f"cohere.{self.embed_model}",
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
            model=self.chat_model,
            messages=[{"role": "user", "content": "Say hello"}],
        )

        self.assertIsNotNone(response)
        self.assertIsNotNone(response.message)

    def test_chat_stream_v2(self):
        """Test streaming chat with v2 client."""
        events = []
        for event in self.client.chat_stream(
            model=self.chat_model,
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
                hasattr(event, "type")
                and event.type == "content-delta"
                and hasattr(event, "delta")
                and event.delta
                and hasattr(event.delta, "message")
                and event.delta.message
                and hasattr(event.delta.message, "content")
                and event.delta.message.content
                and getattr(event.delta.message.content, "text", None)
            ):
                full_text += event.delta.message.content.text
            elif (
                hasattr(event, "delta")
                and event.delta
                and hasattr(event.delta, "message")
                and event.delta.message
                and hasattr(event.delta.message, "content")
                and event.delta.message.content
                and hasattr(event.delta.message.content, "text")
            ):
                self.assertIsNone(event.delta.message.content.text)

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
        except NotFoundError:
            self.skipTest("embed-english-light-v3.0 is not available in this OCI region/profile")
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

        self.assertEqual(result["type"], "content-delta")
        self.assertIn("thinking", result["delta"]["message"]["content"])
        self.assertEqual(result["delta"]["message"]["content"]["thinking"], "Reasoning step...")

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

        self.assertEqual(result["type"], "content-delta")
        self.assertIn("text", result["delta"]["message"]["content"])
        self.assertEqual(result["delta"]["message"]["content"]["text"], "The answer is...")

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
        from unittest.mock import MagicMock, patch
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
                )
            self.assertIn("oci_private_key_path", str(ctx.exception))

    def test_stream_wrapper_emits_full_event_lifecycle(self):
        """Test that V2 streams emit message-start, content-start, content-delta, content-end, and message-end."""
        import json
        from cohere.oci_client import transform_oci_stream_wrapper

        chunks = [
            b'data: {"message": {"content": [{"type": "TEXT", "text": "Hello"}]}}\n',
            b'data: {"message": {"content": [{"type": "TEXT", "text": " world"}]}, "finishReason": "COMPLETE", "usage": {"inputTokens": 3, "completionTokens": 2}}\n',
            b"data: [DONE]\n",
        ]

        events = []
        for raw in transform_oci_stream_wrapper(iter(chunks), "chat", is_v2=True):
            line = raw.decode("utf-8").strip()
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))

        event_types = [event["type"] for event in events]
        self.assertEqual(event_types[0], "message-start")
        self.assertEqual(event_types[1], "content-start")
        self.assertEqual(event_types[2], "content-delta")
        self.assertEqual(event_types[3], "content-delta")
        self.assertEqual(event_types[4], "content-end")
        self.assertEqual(event_types[5], "message-end")

        self.assertIn("id", events[0])
        self.assertEqual(events[0]["delta"]["message"]["role"], "assistant")
        self.assertEqual(events[1]["index"], 0)
        self.assertEqual(events[1]["delta"]["message"]["content"]["type"], "text")
        self.assertEqual(events[2]["delta"]["message"]["content"]["text"], "Hello")
        self.assertEqual(events[3]["delta"]["message"]["content"]["text"], " world")
        self.assertEqual(events[5]["delta"]["finish_reason"], "COMPLETE")
        self.assertEqual(events[5]["delta"]["usage"]["tokens"]["input_tokens"], 3)
        self.assertEqual(events[5]["delta"]["usage"]["tokens"]["output_tokens"], 2)

    def test_stream_wrapper_emits_reasoning_content_transition(self):
        """Test that V2 reasoning streams emit content block transitions from thinking to text."""
        import json
        from cohere.oci_client import transform_oci_stream_wrapper

        chunks = [
            b'data: {"message": {"content": [{"type": "THINKING", "thinking": "step 1"}]}}\n',
            b'data: {"message": {"content": [{"type": "TEXT", "text": "final answer"}]}, "finishReason": "COMPLETE"}\n',
            b"data: [DONE]\n",
        ]

        events = [
            json.loads(line[6:].decode("utf-8"))
            for line in transform_oci_stream_wrapper(iter(chunks), "chat_stream", is_v2=True)
            if line.startswith(b"data: ")
        ]

        self.assertEqual(
            [event["type"] for event in events],
            [
                "message-start",
                "content-start",
                "content-delta",
                "content-end",
                "content-start",
                "content-delta",
                "content-end",
                "message-end",
            ],
        )
        self.assertEqual(events[1]["delta"]["message"]["content"]["type"], "thinking")
        self.assertEqual(events[2]["index"], 0)
        self.assertEqual(events[2]["delta"]["message"]["content"]["thinking"], "step 1")
        self.assertEqual(events[4]["index"], 1)
        self.assertEqual(events[4]["delta"]["message"]["content"]["type"], "text")
        self.assertEqual(events[5]["index"], 1)
        self.assertEqual(events[5]["delta"]["message"]["content"]["text"], "final answer")

    def test_stream_wrapper_skips_malformed_json_with_warning(self):
        """Test that malformed JSON in SSE streams is skipped with a warning."""
        from cohere.oci_client import transform_oci_stream_wrapper

        chunks = [
            b"data: not-valid-json\n",
            b'data: {"message": {"content": [{"type": "TEXT", "text": "hello"}]}}\n',
            b"data: [DONE]\n",
        ]

        events = list(transform_oci_stream_wrapper(iter(chunks), "chat", is_v2=True))
        self.assertEqual(len(events), 4)

    def test_v1_stream_wrapper_emits_stream_end(self):
        """Test that V1 chat streams end with a stream-end event containing the response payload."""
        import json
        from cohere.oci_client import transform_oci_stream_wrapper

        chunks = [
            b'data: {"text": "Hello", "isFinished": false}\n',
            b'data: {"text": " world", "isFinished": true, "finishReason": "COMPLETE"}\n',
            b"data: [DONE]\n",
        ]

        events = [
            json.loads(raw.decode("utf-8"))
            for raw in transform_oci_stream_wrapper(iter(chunks), "chat_stream", is_v2=False)
        ]

        self.assertEqual(events[0]["event_type"], "text-generation")
        self.assertEqual(events[1]["event_type"], "text-generation")
        self.assertEqual(events[2]["event_type"], "stream-end")
        self.assertEqual(events[2]["finish_reason"], "COMPLETE")
        self.assertEqual(events[2]["response"]["text"], "Hello world")

    def test_session_auth_expands_key_file_path(self):
        """Test that session-based auth expands key_file paths before loading them."""
        from unittest.mock import MagicMock, patch
        from cohere.oci_client import map_request_to_oci

        mock_private_key = object()
        mock_security_token_signer = MagicMock()
        mock_oci = MagicMock()
        mock_oci.signer.load_private_key_from_file.return_value = mock_private_key
        mock_oci.auth.signers.SecurityTokenSigner.return_value = mock_security_token_signer

        oci_config = {
            "security_token_file": "~/.oci/sessions/TEST/token",
            "key_file": "~/.oci/sessions/TEST/oci_api_key.pem",
        }

        with patch("cohere.oci_client.lazy_oci", return_value=mock_oci), patch(
            "builtins.open", create=True
        ) as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = "token"
            with patch("os.path.expanduser", side_effect=lambda p: p.replace("~", "/Users/test")):
                with patch("cohere.oci_client.transform_request_to_oci", return_value={"compartmentId": "c"}):
                    hook = map_request_to_oci(oci_config, "us-chicago-1", "compartment-123", is_v2_client=False)
                    request = MagicMock()
                    request.url.path = "/v1/chat"
                    request.read.return_value = b'{"message":"hi"}'
                    request.method = "POST"
                    request.extensions = {}
                    hook(request)

        mock_oci.signer.load_private_key_from_file.assert_called_with("/Users/test/.oci/sessions/TEST/oci_api_key.pem")

    def test_stream_wrapper_raises_on_transform_error(self):
        """Test that stream transformation errors are re-raised with OCI-specific context."""
        from cohere.oci_client import transform_oci_stream_wrapper

        chunks = [
            b'data: {"message": null}\n',
        ]

        with self.assertRaises(RuntimeError) as ctx:
            list(transform_oci_stream_wrapper(iter(chunks), "chat", is_v2=True))
        self.assertIn("OCI stream event transformation failed", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
