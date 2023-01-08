import unittest

import cohere

from utils import get_api_key

API_KEY = get_api_key()
co = cohere.Client(API_KEY)


class TestChat(unittest.TestCase):

    def test_simple_success(self):
        prediction = co.chat("Yo what up?")
        self.assertIsInstance(prediction.reply, str)
        self.assertIsInstance(prediction.session_id, str)
        self.assertEqual(prediction.persona, "cohere")

    def test_multi_replies(self):
        num_replies = 3
        prediction = co.chat("Yo what up?")
        for _ in range(num_replies):
            prediction = prediction.respond("oh that's cool")
            self.assertIsInstance(prediction.reply, str)
            self.assertIsInstance(prediction.session_id, str)
            self.assertEqual(prediction.persona, "cohere")

    def test_invalid_persona(self):
        with self.assertRaises(cohere.CohereError):
            _ = co.chat("Yo what up?", persona="NOT_A_VALID_PERSONA")

    def test_valid_persona(self):
        prediction = co.chat("Yo what up?", persona="wizard")
        self.assertIsInstance(prediction.reply, str)
        self.assertIsInstance(prediction.session_id, str)
        self.assertEqual(prediction.persona, "wizard")

    def test_manual_session_id(self):
        max_num_tries = 5
        prediction = co.chat("Hi my name is Rui")
        print(prediction.reply)

        for _ in range(max_num_tries):
            # manually pick the chat back up using the session_id
            # check that it still has access to information I gave it
            # this is a brittle test, not sure how to improve
            prediction = co.chat("Good to meet you. What's my name?", session_id=prediction.session_id)
            print(prediction.reply)
            test = "rui" in prediction.reply.lower()
            if test:
                break
        else:
            self.fail()
