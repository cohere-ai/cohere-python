import unittest

import cohere
from cohere.feedback import Feedback

from utils import get_api_key

API_KEY = get_api_key()
co = cohere.Client(API_KEY)


class TestFeedback(unittest.TestCase):

    def test_success(self):
        generations = co.generate(model='small', prompt='co:here', max_tokens=1)
        feedback = generations[0].feedback("this is a test")
        self.assertIsInstance(feedback, Feedback)
        self.assertIsNone(prediction.generations[0].token_likelihoods)
        self.assertEqual(generations[0].call_id, feedback.call_id)
