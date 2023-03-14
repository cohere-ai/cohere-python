import unittest

from utils import get_api_key

import cohere
from cohere.error import CohereError
from cohere.responses.summarize import SummarizeResponse

API_KEY = get_api_key()
co = cohere.Client(API_KEY)

text = """Leah LaBelle Vladowski was born on September 8, 1986, in Toronto, \
Ontario, and raised in Seattle, Washington. Her parents, Anastasia and Troshan Vladowski, \
are Bulgarian singers and her uncle made rock music in Bulgaria. Anastasia recorded pop music \
and was in a group with Troshan, who was a founding member of Bulgaria's first rock band, \
Srebyrnite grivni.[3] After defecting from Bulgaria during a 1979 tour,[1][3] LaBelle's parents \
emigrated to Canada and later the United States, becoming naturalized citizens in both countries. \
LaBelle grew up listening to music, including jazz and the Beatles, but felt the most connected to R&B"""


class TestFeedback(unittest.TestCase):
    def test_success(self):
        res = co.summarize(text)
        self.assertIsInstance(res, SummarizeResponse)
        self.assertTrue(res.meta)
        self.assertTrue(res.meta["api_version"])
        self.assertTrue(res.meta["api_version"]["version"])

    def error(self):
        res = co.summarize(text, length="potato")
        self.assertIsInstance(res, CohereError)
