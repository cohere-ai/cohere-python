import unittest

from utils import get_api_key

import cohere

API_KEY = get_api_key()
co = cohere.Client(API_KEY)


class TestChat(unittest.TestCase):
    def test_simple_success(self):
        prediction = co.chat("Yo what up?")
        self.assertIsInstance(prediction.text, str)
        self.assertIsInstance(prediction.conversation_id, str)
        self.assertTrue(prediction.meta)
        self.assertTrue(prediction.meta["api_version"])
        self.assertTrue(prediction.meta["api_version"]["version"])

    def test_multi_replies(self):
        num_replies = 3
        prediction = co.chat("Yo what up?")
        for _ in range(num_replies):
            prediction = prediction.respond("oh that's cool")
            self.assertIsInstance(prediction.text, str)
            self.assertIsInstance(prediction.conversation_id, str)

    def test_valid_model(self):
        prediction = co.chat("Yo what up?", model="medium")
        self.assertIsInstance(prediction.text, str)
        self.assertIsInstance(prediction.conversation_id, str)

    def test_invalid_model(self):
        with self.assertRaises(cohere.CohereError):
            co.chat("Yo what up?", model="NOT_A_VALID_MODEL").text

    def test_return_chatlog(self):
        prediction = co.chat("Yo what up?", return_chatlog=True)
        self.assertIsInstance(prediction.text, str)
        self.assertIsInstance(prediction.conversation_id, str)
        self.assertIsNotNone(prediction.chatlog)
        self.assertGreaterEqual(len(prediction.chatlog), len(prediction.text))

    def test_return_chatlog_false(self):
        prediction = co.chat("Yo what up?", return_chatlog=False)
        self.assertIsInstance(prediction.text, str)
        self.assertIsInstance(prediction.conversation_id, str)

        assert prediction.chatlog is None

    def test_return_prompt(self):
        prediction = co.chat("Yo what up?", return_prompt=True)
        self.assertIsInstance(prediction.text, str)
        self.assertIsInstance(prediction.conversation_id, str)
        self.assertIsNotNone(prediction.prompt)
        self.assertGreaterEqual(len(prediction.prompt), len(prediction.text))

    def test_return_prompt_false(self):
        prediction = co.chat("Yo what up?", return_prompt=False)
        self.assertIsInstance(prediction.text, str)
        self.assertIsInstance(prediction.conversation_id, str)
        assert prediction.prompt is None

    def test_preamble_override(self):
        preamble = "You are a dog who mostly barks"
        prediction = co.chat("Yo what up?", preamble_override=preamble, return_prompt=True, return_preamble=True)
        self.assertIsInstance(prediction.text, str)
        self.assertIsInstance(prediction.conversation_id, str)
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
            self.assertIsInstance(prediction.conversation_id, str)

    def test_max_tokens(self):
        prediction = co.chat("Yo what up?", max_tokens=10)
        self.assertIsInstance(prediction.text, str)
        self.assertIsInstance(prediction.conversation_id, str)
        tokens = co.tokenize(prediction.text)
        self.assertLessEqual(tokens.length, 10)

    def test_stream(self):
        prediction = co.chat(query="Yo what up?", max_tokens=5, stream=True)

        for token in prediction:
            self.assertIsInstance(token.text, str)
            self.assertGreater(len(token.text), 0)

        self.assertIsInstance(prediction.texts, list)

    def test_id(self):
        prediction1 = co.chat("Yo what up?")
        self.assertIsNotNone(prediction1.response_id)

        prediction2 = co.chat("hey")
        self.assertIsNotNone(prediction2.response_id)

        self.assertNotEqual(prediction1.response_id, prediction2.response_id)

    def test_return_preamble(self):
        prediction = co.chat("Yo what up?", return_preamble=True, return_prompt=True)
        self.assertIsInstance(prediction.text, str)
        self.assertIsInstance(prediction.conversation_id, str)
        self.assertIsNotNone(prediction.preamble)
        self.assertIsNotNone(prediction.prompt)
        self.assertIn(prediction.preamble, prediction.prompt)

    def test_return_preamble_false(self):
        prediction = co.chat("Yo what up?", return_preamble=False)
        self.assertIsInstance(prediction.text, str)
        self.assertIsInstance(prediction.conversation_id, str)

        assert prediction.preamble is None

    def test_chat_history(self):
        prediction = co.chat(
            "Who are you?",
            chat_history=[
                {"speaker": "USER", "text": "Hey!"},
                {"speaker": "CHATBOT", "text": "Hey! How can I help you?"},
            ],
            return_prompt=True,
            return_chatlog=True,
        )
        self.assertIsInstance(prediction.text, str)
        self.assertIsInstance(prediction.conversation_id, str)
        self.assertIsNone(prediction.chatlog)
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
                    query="Hey!",
                    chat_history=chat_history,
                )
