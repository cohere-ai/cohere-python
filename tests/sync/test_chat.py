import unittest

from utils import get_api_key

import cohere

API_KEY = get_api_key()
co = cohere.Client(API_KEY)


class TestChat(unittest.TestCase):
    def test_simple_success(self):
        prediction = co.chat("Yo what up?")
        self.assertIsInstance(prediction.reply, str)
        self.assertIsInstance(prediction.conversation_id, str)
        self.assertTrue(prediction.meta)
        self.assertTrue(prediction.meta["api_version"])
        self.assertTrue(prediction.meta["api_version"]["version"])

    def test_multi_replies(self):
        num_replies = 3
        prediction = co.chat("Yo what up?")
        for _ in range(num_replies):
            prediction = prediction.respond("oh that's cool")
            self.assertIsInstance(prediction.reply, str)
            self.assertIsInstance(prediction.conversation_id, str)

    def test_valid_model(self):
        prediction = co.chat("Yo what up?", model="medium")
        self.assertIsInstance(prediction.reply, str)
        self.assertIsInstance(prediction.conversation_id, str)

    def test_invalid_model(self):
        with self.assertRaises(cohere.CohereError):
            co.chat("Yo what up?", model="NOT_A_VALID_MODEL").reply

    def test_return_chatlog(self):
        prediction = co.chat("Yo what up?", return_chatlog=True)
        self.assertIsInstance(prediction.reply, str)
        self.assertIsInstance(prediction.conversation_id, str)
        self.assertIsNotNone(prediction.chatlog)
        self.assertGreaterEqual(len(prediction.chatlog), len(prediction.reply))

    def test_return_chatlog_false(self):
        prediction = co.chat("Yo what up?", return_chatlog=False)
        self.assertIsInstance(prediction.reply, str)
        self.assertIsInstance(prediction.conversation_id, str)

        assert prediction.chatlog is None

    def testValidChatlogOverride(self):
        query = "What about you?"
        valid_chatlog_overrides = [
            [
                {"Bot": "Hey!"},
                {"User": "I am doing great!"},
                {"Bot": "That is great to hear!"},
            ],
            [],
        ]

        for chatlog_override in valid_chatlog_overrides:
            expected_chatlog = ""
            for message in chatlog_override:
                key, value = next(iter(message.items()))
                expected_chatlog += f"{key}: {value}\n"
            expected_chatlog += "User: " + query

            prediction = co.chat(
                query=query, conversation_id="1234", chatlog_override=chatlog_override, return_chatlog=True
            )

            self.assertIsInstance(prediction.reply, str)
            self.assertIsInstance(prediction.conversation_id, str)
            self.assertIn(expected_chatlog, prediction.chatlog)

    def testInvalidChatlogOverride(self):
        invalid_chatlog_overrides = [
            "invalid",
            ["invalid"],
            [{"user": "invalid", "bot": "invalid"}],
        ]

        for chatlog_override in invalid_chatlog_overrides:
            with self.assertRaises(cohere.error.CohereError):
                _ = co.chat(
                    query="What about you?",
                    conversation_id="1234",
                    chatlog_override=chatlog_override,
                    return_chatlog=True,
                )

    def test_return_prompt(self):
        prediction = co.chat("Yo what up?", return_prompt=True)
        self.assertIsInstance(prediction.reply, str)
        self.assertIsInstance(prediction.conversation_id, str)
        self.assertIsNotNone(prediction.prompt)
        self.assertGreaterEqual(len(prediction.prompt), len(prediction.reply))

    def test_return_prompt_false(self):
        prediction = co.chat("Yo what up?", return_prompt=False)
        self.assertIsInstance(prediction.reply, str)
        self.assertIsInstance(prediction.conversation_id, str)
        assert prediction.prompt is None

    def test_preamble_override(self):
        preamble = "You are a dog who mostly barks"
        prediction = co.chat("Yo what up?", preamble_override=preamble, return_prompt=True)
        self.assertIsInstance(prediction.reply, str)
        self.assertIsInstance(prediction.conversation_id, str)
        self.assertIn(preamble, prediction.prompt)

    def test_invalid_preamble_override(self):
        with self.assertRaises(cohere.CohereError) as e:
            co.chat("Yo what up?", preamble_override=123).reply
        self.assertIn("invalid type", str(e.exception))

    def test_valid_temperatures(self):
        temperatures = [0.1, 0.9]

        for temperature in temperatures:
            prediction = co.chat("Yo what up?", temperature=temperature, max_tokens=5)
            self.assertIsInstance(prediction.reply, str)
            self.assertIsInstance(prediction.conversation_id, str)

    def test_max_tokens(self):
        prediction = co.chat("Yo what up?", max_tokens=10)
        self.assertIsInstance(prediction.reply, str)
        self.assertIsInstance(prediction.conversation_id, str)
        tokens = co.tokenize(prediction.reply)
        self.assertLessEqual(tokens.length, 10)

    def test_stream(self):
        prediction = co.chat(query="Yo what up?", max_tokens=5, stream=True)

        for token in prediction:
            self.assertIsInstance(token.text, str)
            self.assertGreater(len(token.text), 0)

        self.assertIsInstance(prediction.texts, list)

    def test_id(self):
        prediction1 = co.chat("Yo what up?")
        self.assertIsNotNone(prediction1.id)

        prediction2 = co.chat("hey")
        self.assertIsNotNone(prediction2.id)

        self.assertNotEqual(prediction1.id, prediction2.id)

    def test_deprecated_session_id_warning(self):
        with self.assertWarns(DeprecationWarning):
            co.chat("Yo what up?", session_id="1234")

    def test_deprecated_persona_prompt_warning(self):
        with self.assertWarns(DeprecationWarning):
            co.chat("Yo what up?", persona_prompt="This is you")

    def test_deprecated_persona_name_warning(self):
        with self.assertWarns(DeprecationWarning):
            co.chat("Yo what up?", persona_name="You")

    def test_deprecated_user_name_warning(self):
        with self.assertWarns(DeprecationWarning):
            co.chat("Yo what up?", user_name="You")
