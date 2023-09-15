import unittest
import pytest

from utils import get_api_key

import cohere
from cohere.responses.chat import (
    StreamCitationGeneration,
    StreamEnd,
    StreamQueryGeneration,
    StreamSearchResults,
    StreamStart,
    StreamTextGeneration,
)

API_KEY = get_api_key()
co = cohere.Client(API_KEY)


class TestChat(unittest.TestCase):
    def test_simple_success(self):
        prediction = co.chat("Yo what up?", max_tokens=5)
        self.assertIsInstance(prediction.text, str)
        self.assertTrue(prediction.meta)
        self.assertTrue(prediction.meta["api_version"])
        self.assertTrue(prediction.meta["api_version"]["version"])

    def test_multi_replies(self):
        num_replies = 3
        prediction = co.chat("Yo what up?", max_tokens=5)
        for _ in range(num_replies):
            prediction = prediction.respond("oh that's cool", max_tokens=5)
            self.assertIsInstance(prediction.text, str)

    def test_valid_model(self):
        prediction = co.chat("Yo what up?", model="command", max_tokens=5)
        self.assertIsInstance(prediction.text, str)

    def test_invalid_model(self):
        with self.assertRaises(cohere.CohereError):
            co.chat("Yo what up?", model="NOT_A_VALID_MODEL").text

    def test_return_chatlog(self):
        prediction = co.chat("Yo what up?", return_chatlog=True, max_tokens=5)
        self.assertIsInstance(prediction.text, str)
        self.assertIsNotNone(prediction.chatlog)
        self.assertGreaterEqual(len(prediction.chatlog), len(prediction.text))

    def test_return_chatlog_false(self):
        prediction = co.chat("Yo what up?", return_chatlog=False, max_tokens=5)
        self.assertIsInstance(prediction.text, str)

        assert prediction.chatlog is None

    def test_return_prompt(self):
        prediction = co.chat("Yo what up?", return_prompt=True, max_tokens=5)
        self.assertIsInstance(prediction.text, str)
        self.assertIsNotNone(prediction.prompt)
        self.assertGreaterEqual(len(prediction.prompt), len(prediction.text))

    def test_return_prompt_false(self):
        prediction = co.chat("Yo what up?", return_prompt=False, max_tokens=5)
        self.assertIsInstance(prediction.text, str)
        assert prediction.prompt is None

    def test_preamble_override(self):
        preamble = "You are a dog who mostly barks"
        prediction = co.chat(
            "Yo what up?", preamble_override=preamble, return_prompt=True, return_preamble=True, max_tokens=5
        )
        self.assertIsInstance(prediction.text, str)
        self.assertIn(preamble, prediction.prompt)
        self.assertEqual(preamble, prediction.preamble)

    def test_invalid_preamble_override(self):
        with self.assertRaises(cohere.CohereError) as e:
            co.chat("Yo what up?", preamble_override=123).text
        self.assertIn("invalid type", str(e.exception))

    def test_valid_temperatures(self):
        temperatures = [0.1, 0.9]

        for temperature in temperatures:
            prediction = co.chat("Yo what up?", temperature=temperature, max_tokens=5)
            self.assertIsInstance(prediction.text, str)

    def test_stream(self):
        prediction = co.chat(
            message="Yo what up?",
            max_tokens=5,
            stream=True,
        )

        self.assertIsInstance(prediction, cohere.responses.chat.StreamingChat)
        self.assertIsInstance(prediction.texts, list)
        self.assertEqual(len(prediction.texts), 0)
        self.assertIsNone(prediction.response_id)
        self.assertIsNone(prediction.finish_reason)

        expected_index = 0
        expected_text = ""
        for token in prediction:
            if isinstance(token, cohere.responses.chat.StreamStart):
                self.assertIsNotNone(token.generation_id)
                self.assertFalse(token.is_finished)
            elif isinstance(token, cohere.responses.chat.StreamTextGeneration):
                self.assertIsInstance(token.text, str)
                self.assertGreater(len(token.text), 0)
                expected_text += token.text
                self.assertFalse(token.is_finished)
            self.assertIsInstance(token.index, int)
            self.assertEqual(token.index, expected_index)
            expected_index += 1

        self.assertEqual(prediction.texts, [expected_text])
        self.assertIsNotNone(prediction.response_id)
        self.assertIsNotNone(prediction.finish_reason)

    def test_id(self):
        prediction1 = co.chat("Yo what up?", max_tokens=5)
        self.assertIsNotNone(prediction1.response_id)

        prediction2 = co.chat("hey", max_tokens=5)
        self.assertIsNotNone(prediction2.response_id)

        self.assertNotEqual(prediction1.response_id, prediction2.response_id)

    def test_return_preamble(self):
        prediction = co.chat("Yo what up?", return_preamble=True, return_prompt=True, max_tokens=5)
        self.assertIsInstance(prediction.text, str)
        self.assertIsNotNone(prediction.preamble)
        self.assertIsNotNone(prediction.prompt)
        self.assertIn(prediction.preamble, prediction.prompt)

    def test_return_preamble_false(self):
        prediction = co.chat("Yo what up?", return_preamble=False, max_tokens=5)
        self.assertIsInstance(prediction.text, str)

        assert prediction.preamble is None

    def test_chat_history(self):
        prediction = co.chat(
            "Who are you?",
            chat_history=[
                {"user_name": "User", "message": "Hey!"},
                {"user_name": "Chatbot", "message": "Hey! How can I help you?"},
            ],
            return_prompt=True,
            return_chatlog=True,
            max_tokens=5,
        )
        self.assertIsInstance(prediction.text, str)
        self.assertIsNotNone(prediction.chatlog)
        self.assertIn("User: Hey!", prediction.prompt)
        self.assertIn("Chatbot: Hey! How can I help you?", prediction.prompt)

    def test_invalid_chat_history(self):
        invalid_chat_histories = [
            "invalid",
            ["invalid"],
            [{"user": "invalid", "bot": "invalid"}],
        ]

        for chat_history in invalid_chat_histories:
            with self.assertRaises(cohere.error.CohereError):
                _ = co.chat(
                    message="Hey!",
                    chat_history=chat_history,
                )

    def test_token_count(self):
        prediction = co.chat("Yo what up?", max_tokens=5)
        self.assertLessEqual(prediction.token_count["response_tokens"], 5)
        self.assertEqual(
            prediction.token_count["total_tokens"],
            prediction.token_count["prompt_tokens"] + prediction.token_count["response_tokens"],
        )

    def test_p(self):
        prediction = co.chat("Yo what up?", p=0.9, max_tokens=5)
        self.assertIsInstance(prediction.text, str)

    def test_invalid_p(self):
        with self.assertRaises(cohere.error.CohereError):
            _ = co.chat("Yo what up?", p="invalid", max_tokens=5)

    def test_k(self):
        prediction = co.chat("Yo what up?", k=5, max_tokens=5)
        self.assertIsInstance(prediction.text, str)

    def test_invalid_k(self):
        with self.assertRaises(cohere.error.CohereError):
            _ = co.chat("Yo what up?", k="invalid", max_tokens=5)

    def test_logit_bias(self):
        prediction = co.chat("Yo what up?", logit_bias={42: 10}, max_tokens=5)
        self.assertIsInstance(prediction.text, str)

    def test_invalid_logit_bias(self):
        invalid = [
            "invalid",
            {"hello": "invalid"},
            {-42: 10},
        ]

        for logit_bias in invalid:
            with self.assertRaises(cohere.error.CohereError):
                _ = co.chat("Yo what up?", logit_bias=logit_bias, max_tokens=5)

    def test_search_queries_only_true(self):
        prediction = co.chat(
            "What is the height of Mount Everest? What is the depth of the Mariana Trench? What is the climate like in Nepal?",
            search_queries_only=True,
        )
        self.assertTrue(prediction.is_search_required)
        self.assertIsInstance(prediction.search_queries, list)
        self.assertGreater(len(prediction.search_queries), 0)
        self.assertIsInstance(prediction.search_queries[0]["text"], str)
        self.assertIsInstance(prediction.search_queries[0]["generation_id"], str)

    @pytest.mark.skip(reason="temporarily unblock")
    def test_search_queries_only_false(self):
        prediction = co.chat("hello", search_queries_only=True)
        self.assertFalse(prediction.is_search_required)
        self.assertIsInstance(prediction.search_queries, list)
        self.assertEqual(len(prediction.search_queries), 0)

    def test_with_documents(self):
        prediction = co.chat(
            "How deep in the Mariana Trench",
            temperature=0,
            documents=[
                {
                    "id": "national_geographic_everest",
                    "title": "Height of Mount Everest",
                    "snippet": "The height of Mount Everest is 29,035 feet",
                    "url": "https://education.nationalgeographic.org/resource/mount-everest/",
                },
                {
                    "id": "national_geographic_mariana",
                    "title": "Depth of the Mariana Trench",
                    "snippet": "The depth of the Mariana Trench is 36,070 feet",
                    "url": "https://www.nationalgeographic.org/activity/mariana-trench-deepest-place-earth",
                },
            ],
        )
        self.assertIsInstance(prediction.text, str)
        self.assertIsInstance(prediction.citations, list)
        self.assertGreater(len(prediction.citations), 0)
        self.assertIsInstance(prediction.citations[0]["start"], int)
        self.assertIsInstance(prediction.citations[0]["end"], int)
        self.assertIsInstance(prediction.citations[0]["text"], str)
        self.assertIsInstance(prediction.citations[0]["document_ids"], list)
        self.assertGreater(len(prediction.citations[0]["document_ids"]), 0)
        self.assertIsInstance(prediction.documents, list)
        self.assertGreater(len(prediction.documents), 0)

    def test_with_connectors(self):
        prediction = co.chat("How deep in the Mariana Trench", temperature=0, connectors=[{"id": "web-search"}])
        self.assertIsInstance(prediction.text, str)
        self.assertIsInstance(prediction.citations, list)
        self.assertGreater(len(prediction.citations), 0)
        self.assertIsInstance(prediction.citations[0]["start"], int)
        self.assertIsInstance(prediction.citations[0]["end"], int)
        self.assertIsInstance(prediction.citations[0]["text"], str)
        self.assertIsInstance(prediction.citations[0]["document_ids"], list)
        self.assertGreater(len(prediction.citations[0]["document_ids"]), 0)
        self.assertIsInstance(prediction.documents, list)
        self.assertGreater(len(prediction.documents), 0)
        self.assertIsInstance(prediction.search_results, list)
        self.assertGreater(len(prediction.search_results), 0)

    def test_with_citation_quality(self):
        prediction = co.chat(
            "How deep in the Mariana Trench",
            citation_quality="accurate",
            temperature=0,
            documents=[
                {
                    "id": "national_geographic_mariana",
                    "title": "Depth of the Mariana Trench",
                    "snippet": "The depth of the Mariana Trench is 36,070 feet",
                    "url": "https://www.nationalgeographic.org/activity/mariana-trench-deepest-place-earth",
                },
            ],
        )
        self.assertIsInstance(prediction.text, str)
        self.assertIsInstance(prediction.citations, list)
        self.assertGreater(len(prediction.citations), 0)
        self.assertIsInstance(prediction.citations[0]["start"], int)
        self.assertIsInstance(prediction.citations[0]["end"], int)
        self.assertIsInstance(prediction.citations[0]["text"], str)
        self.assertIsInstance(prediction.citations[0]["document_ids"], list)
        self.assertGreater(len(prediction.citations[0]["document_ids"]), 0)
        self.assertIsInstance(prediction.documents, list)
        self.assertGreater(len(prediction.documents), 0)

    def test_stream_with_connectors(self):
        prediction = co.chat(
            "How deep in the Mariana Trench", temperature=0, stream=True, connectors=[{"id": "web-search"}]
        )

        self.assertIsInstance(prediction, cohere.responses.chat.StreamingChat)
        self.assertIsInstance(prediction.texts, list)
        self.assertEqual(len(prediction.texts), 0)
        self.assertIsNone(prediction.response_id)
        self.assertIsNone(prediction.finish_reason)

        expected_index = 0
        expected_text = ""

        count_stream_start = 0
        count_text_generation = 0
        count_query_generation = 0
        count_citation_generation = 0
        count_search_results = 0
        count_stream_end = 0
        for token in prediction:
            if isinstance(token, StreamStart):
                count_stream_start += 1
                self.assertIsNotNone(token.generation_id)
                self.assertFalse(token.is_finished)
                self.assertEqual(token.event_type, "stream-start")
            elif isinstance(token, StreamQueryGeneration):
                count_query_generation += 1
                self.assertIsNotNone(token.search_queries)
                self.assertEqual(token.event_type, "search-queries-generation")
            elif isinstance(token, StreamSearchResults):
                count_search_results += 1
                self.assertIsNotNone(token.documents)
                self.assertIsNotNone(token.search_results)
                self.assertEqual(token.event_type, "search-results")
            elif isinstance(token, StreamCitationGeneration):
                count_citation_generation += 1
                self.assertIsNotNone(token.citations)
                self.assertEqual(token.event_type, "citation-generation")
            elif isinstance(token, StreamTextGeneration):
                count_text_generation += 1
                self.assertIsInstance(token.text, str)
                self.assertGreater(len(token.text), 0)
                expected_text += token.text
                self.assertFalse(token.is_finished)
                self.assertEqual(token.event_type, "text-generation")
            elif isinstance(token, StreamEnd):
                count_stream_end += 1
                self.assertTrue(token.is_finished)
                self.assertEqual(token.event_type, "stream-end")
                self.assertEqual(token.finish_reason, "COMPLETE")
            self.assertIsInstance(token.index, int)
            self.assertEqual(token.index, expected_index)
            expected_index += 1

        self.assertEqual(count_stream_start, 1)
        self.assertEqual(count_search_results, 1)
        self.assertEqual(count_stream_end, 1)
        self.assertGreaterEqual(count_citation_generation, 1)
        self.assertGreaterEqual(count_query_generation, 1)
        self.assertGreaterEqual(count_text_generation, 1)

        self.assertEqual(prediction.texts, [expected_text])
        self.assertIsNotNone(prediction.response_id)
        self.assertIsNotNone(prediction.finish_reason)
