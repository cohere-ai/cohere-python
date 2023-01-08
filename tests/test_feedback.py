import unittest

from utils import get_api_key

import cohere
from cohere.feedback import Feedback

API_KEY = get_api_key()
co = cohere.Client(API_KEY)


class TestFeedback(unittest.TestCase):

    def test_success(self):
        generations = co.generate(model='small', prompt='co:here', max_tokens=1)
        feedback = generations[0].feedback("this is a test", accepted=True)
        self.assertIsInstance(feedback, Feedback)
