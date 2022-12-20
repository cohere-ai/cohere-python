import unittest

import cohere

from utils import get_api_key

co = cohere.Client(get_api_key())


class TestDetectLanguage(unittest.TestCase):

    def test_success(self):
        languages = co.detect_language(["Hello world!", "Привет Мир!"]).results
        self.assertEqual(languages[0].language_code, "en")
        self.assertEqual(languages[1].language_code, "ru")
