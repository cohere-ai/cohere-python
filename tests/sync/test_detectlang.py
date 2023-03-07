import unittest

import cohere

from utils import get_api_key

co = cohere.Client(get_api_key())


class TestDetectLanguage(unittest.TestCase):

    def test_success(self):
        response = co.detect_language(["Hello world!", "Привет Мир!"])
        languages = response.results

        self.assertEqual(languages[0].language_code, "en")
        self.assertEqual(languages[1].language_code, "ru")
        self.assertTrue(response.meta)
        self.assertTrue(response.meta["api_version"])
        self.assertTrue(response.meta["api_version"]["version"])
