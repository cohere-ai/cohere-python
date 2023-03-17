import unittest

from utils import get_api_key

import cohere
from cohere.generate_feedback import GenerateFeedback

API_KEY = get_api_key()
co = cohere.Client(API_KEY)


class TestGenerateFeedback(unittest.TestCase):

    def test_success(self):
        generations = co.generate(model='medium', prompt='co:here', max_tokens=1)
        generations[0].feedback(good_response=True, desired_response="this is a test")

    def test_from_id(self):
        generations = co.generate(model='medium', prompt='co:here', max_tokens=1)
        co.generate_feedback(request_id=generations[0].id, good_response=True, desired_response="this is a test")
