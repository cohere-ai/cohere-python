import unittest

from utils import get_api_key

import cohere

API_KEY = get_api_key()
co = cohere.Client(API_KEY)


class TestChat(unittest.TestCase):

    def test_simple_success(self):
        prediction = co.chat("Yo what up?")
        self.assertIsInstance(prediction.reply, str)
        self.assertIsInstance(prediction.session_id, str)

    def test_multi_replies(self):
        num_replies = 3
        prediction = co.chat("Yo what up?")
        for _ in range(num_replies):
            prediction = prediction.respond("oh that's cool")
            self.assertIsInstance(prediction.reply, str)
            self.assertIsInstance(prediction.session_id, str)

    # TODO re-add tests

    # def test_manual_session_id(self):
    #     max_num_tries = 5
    #     prediction = co.chat("Hi my name is Rui")

    #     for _ in range(max_num_tries):
    #         # manually pick the chat back up using the session_id
    #         # check that it still has access to information I gave it
    #         # this is a brittle test, not sure how to improve
    #         prediction = co.chat("Good to meet you. What's my name?", session_id=prediction.session_id)
    #         test = "rui" in prediction.reply.lower()
    #         if test:
    #             break
    #     else:
    #         self.fail()

    def test_valid_model(self):
        prediction = co.chat("Yo what up?", model="medium")
        self.assertIsInstance(prediction.reply, str)
        self.assertIsInstance(prediction.session_id, str)

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

        with self.assertRaises(AttributeError):
            prediction.chatlog

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

        with self.assertRaises(AttributeError):
            prediction.prompt

    def test_username_override(self):
        user_name = "CustomUser"
        prediction = co.chat("Yo what up?", user_name=user_name, return_chatlog=True)
        self.assertIsInstance(prediction.reply, str)
        self.assertIsInstance(prediction.session_id, str)
        chatlog_starts_with_username = prediction.chatlog.strip().startswith(user_name)
        self.assertTrue(chatlog_starts_with_username)

    def test_valid_temperatures(self):
        temperatures = [0.1, 0.5, 0.9]

        for temperature in temperatures:
            prediction = co.chat("Yo what up?", temperature=temperature)
            self.assertIsInstance(prediction.reply, str)
            self.assertIsInstance(prediction.session_id, str)

    def test_max_tokens(self):
        prediction = co.chat("Yo what up?", max_tokens=10)
        self.assertIsInstance(prediction.reply, str)
        self.assertIsInstance(prediction.session_id, str)
        tokens = co.tokenize(prediction.reply)
        self.assertLessEqual(tokens.length, 10)
