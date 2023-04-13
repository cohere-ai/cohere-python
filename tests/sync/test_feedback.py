import unittest

from utils import get_api_key

import cohere
from cohere.responses.feedback import GenerateFeedbackResponse, PreferenceRating

API_KEY = get_api_key()
co = cohere.Client(API_KEY)


class TestGenerateFeedback(unittest.TestCase):

    def test_from_id(self):
        generations = co.generate(prompt="co:here", max_tokens=1)
        feedback = co.generate_feedback(request_id=generations[0].id,
                                        desired_response="is the best",
                                        good_response=False)
        self.assertIsInstance(feedback, GenerateFeedbackResponse)


class TestGeneratePreferenceFeedback(unittest.TestCase):

    def test_from_id(self):
        generations = co.generate(prompt="co:here", max_tokens=1, num_generations=2)
        feedback = co.generate_preference_feedback(prompt="co:here",
                                                   ratings=[
                                                       PreferenceRating(generations[0].id, 0.5, generations[0].text),
                                                       PreferenceRating(generations[1].id, 0.5, generations[1].text)
                                                   ])
        self.assertIsInstance(feedback, GenerateFeedbackResponse)
