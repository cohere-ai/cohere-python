import unittest

from utils import get_api_key

import cohere
from cohere.responses.feedback import Feedback

API_KEY = get_api_key()
co = cohere.Client(API_KEY)


class TestFeedback(unittest.TestCase):
    def test_from_id(self):
        generations = co.generate(model="medium", prompt="co:here", max_tokens=1)
        feedback = co.feedback(id=generations[0].id, good_response=True, feedback="this is a test")
        self.assertIsInstance(feedback, Feedback)
