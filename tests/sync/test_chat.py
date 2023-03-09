import unittest

from utils import get_api_key,in_ci

import cohere

API_KEY = get_api_key()
co = cohere.Client(API_KEY)


class TestChat(unittest.TestCase):

    def test_simple_success(self):
        prediction = co.chat("Yo what up?")
        self.assertIsInstance(prediction.reply, str)
        self.assertIsInstance(prediction.session_id, str)
        self.assertEqual(prediction.persona_name, "cohere")
        self.assertTrue(prediction.meta)
        self.assertTrue(prediction.meta["api_version"])
        self.assertTrue(prediction.meta["api_version"]["version"])

    def test_multi_replies(self):
        num_replies = 3
        prediction = co.chat("Yo what up?")
        for _ in range(num_replies):
            prediction = prediction.respond("oh that's cool")
            self.assertIsInstance(prediction.reply, str)
            self.assertIsInstance(prediction.session_id, str)
            self.assertEqual(prediction.persona_name, "cohere")

    def test_valid_persona(self):
        prediction = co.chat("Yo what up?", persona_name="Wizard",return_chatlog=True)
        self.assertIsInstance(prediction.reply, str)
        self.assertIsInstance(prediction.session_id, str)
        self.assertEqual(prediction.persona_name, "Wizard")
        self.assertIn('Wizard:', prediction.chatlog) 

    def test_valid_model(self):
        prediction = co.chat("Yo what up?", model="medium")
        self.assertIsInstance(prediction.reply, str)
        self.assertIsInstance(prediction.session_id, str)
        self.assertEqual(prediction.persona_name, "cohere")

    def test_invalid_model(self):
        with self.assertRaises(cohere.CohereError):
            co.chat("Yo what up?", model="NOT_A_VALID_MODEL").reply

    def test_return_chatlog(self):
        prediction = co.chat("Yo what up?", return_chatlog=True)
        self.assertIsInstance(prediction.reply, str)
        self.assertIsInstance(prediction.session_id, str)
        self.assertIsNotNone(prediction.chatlog)
        self.assertGreaterEqual(len(prediction.chatlog), len(prediction.reply))

    def test_return_chatlog_false(self):
        prediction = co.chat("Yo what up?", return_chatlog=False)
        self.assertIsInstance(prediction.reply, str)
        self.assertIsInstance(prediction.session_id, str)

        assert prediction.chatlog is None

    def testValidChatlogOverride(self):
        query = "What about you?"
        valid_chatlog_overrides = [[
            {
                'Bot': 'Hey!'
            },
            {
                'User': 'I am doing great!'
            },
            {
                'Bot': 'That is great to hear!'
            },
        ], []]

        for chatlog_override in valid_chatlog_overrides:
            expected_chatlog = ""
            for message in chatlog_override:
                key, value = next(iter(message.items()))
                expected_chatlog += f"{key}: {value}\n"
            expected_chatlog += "User: " + query

            prediction = co.chat(query=query, session_id="1234", chatlog_override=chatlog_override, return_chatlog=True)

            self.assertIsInstance(prediction.reply, str)
            self.assertIsInstance(prediction.session_id, str)
            self.assertIn(expected_chatlog, prediction.chatlog)

    def testInvalidChatlogOverride(self):
        invalid_chatlog_overrides = [
            "invalid",
            ["invalid"],
            [{
                "user": "invalid",
                "bot": "invalid"
            }],
        ]

        for chatlog_override in invalid_chatlog_overrides:
            with self.assertRaises(cohere.error.CohereError):
                _ = co.chat(query="What about you?",
                            session_id="1234",
                            chatlog_override=chatlog_override,
                            return_chatlog=True)

    def test_return_prompt(self):
        prediction = co.chat("Yo what up?", return_prompt=True)
        self.assertIsInstance(prediction.reply, str)
        self.assertIsInstance(prediction.session_id, str)
        self.assertIsNotNone(prediction.prompt)
        self.assertGreaterEqual(len(prediction.prompt), len(prediction.reply))

    def test_return_prompt_false(self):
        prediction = co.chat("Yo what up?", return_prompt=False)
        self.assertIsInstance(prediction.reply, str)
        self.assertIsInstance(prediction.session_id, str)
        assert prediction.prompt is None

    def test_preamble_override(self):
        preamble = "You are a dog who mostly barks"
        prediction = co.chat("Yo what up?", persona_prompt=preamble, return_prompt=True)
        self.assertIsInstance(prediction.reply, str)
        self.assertIsInstance(prediction.session_id, str)
        self.assertIn(preamble, prediction.prompt)
        print(prediction.prompt)

    def test_invalid_preamble_override(self):
        with self.assertRaises(cohere.CohereError) as e:
            co.chat("Yo what up?", persona_prompt=123).reply
        self.assertIn('invalid type', str(e.exception)) 

    def test_username_override(self):
        username = "CustomUser"
        prediction = co.chat("Yo what up?", user_name=username, return_chatlog=True)
        self.assertIsInstance(prediction.reply, str)
        self.assertIsInstance(prediction.session_id, str)
        chatlog_starts_with_username = prediction.chatlog.strip().startswith(username)
        self.assertTrue(chatlog_starts_with_username)
