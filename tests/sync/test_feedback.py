import unittest

from utils import get_api_key

import cohere
from cohere.responses.feedback import GenerateFeedbackResponse

API_KEY = get_api_key()
co = cohere.Client(API_KEY)


class TestGenerateFeedback(unittest.TestCase):
    def test_from_id(self):
        generations = co.generate(prompt="co:here", max_tokens=1)
        feedback = co.generate_feedback(
            request_id=generations[0].id, desired_response="is the best", good_response=False
        )
        self.assertIsInstance(feedback, GenerateFeedbackResponse)
