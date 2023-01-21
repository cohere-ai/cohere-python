import unittest

import cohere

from utils import get_api_key

API_KEY = get_api_key()
co = cohere.Client(API_KEY)


class TestGenerate(unittest.TestCase):

    def test_success(self):
        prediction = co.generate(model='medium', prompt='co:here', max_tokens=1)
        self.assertIsInstance(prediction.generations[0].text, str)
        self.assertIsNone(prediction.generations[0].token_likelihoods)
        self.assertEqual(prediction.return_likelihoods, None)

    def test_success_batched(self):
        _batch_size = 10
        predictions = co.batch_generate(model='medium', prompts=['co:here'] * _batch_size, max_tokens=1)
        for prediction in predictions:
            self.assertIsInstance(prediction.generations[0].text, str)
            self.assertIsNone(prediction.generations[0].token_likelihoods)
            self.assertEqual(prediction.return_likelihoods, None)

    def test_return_likelihoods_generation(self):
        prediction = co.generate(model='medium', prompt='co:here', max_tokens=1, return_likelihoods='GENERATION')
        self.assertTrue(prediction.generations[0].token_likelihoods)
        self.assertTrue(prediction.generations[0].token_likelihoods[0].token)
        self.assertIsNotNone(prediction.generations[0].likelihood)
        self.assertEqual(prediction.return_likelihoods, 'GENERATION')

    def test_return_likelihoods_all(self):
        prediction = co.generate(model='medium', prompt='hi', max_tokens=1, return_likelihoods='ALL')
        self.assertEqual(len(prediction.generations[0].token_likelihoods), 2)
        self.assertIsNotNone(prediction.generations[0].likelihood)
        self.assertEqual(prediction.return_likelihoods, 'ALL')

    def test_invalid_temp(self):
        with self.assertRaises(cohere.CohereError):
            co.generate(model='large', prompt='hi', max_tokens=1, temperature=-1).generations

    def test_invalid_model(self):
        with self.assertRaises(cohere.CohereError):
            co.generate(model='this-better-not-exist', prompt='co:here', max_tokens=1).generations

    def test_no_version_works(self):
        cohere.Client(API_KEY).generate(model='medium', prompt='co:here', max_tokens=1).generations

    def test_invalid_version_fails(self):
        with self.assertRaises(cohere.CohereError):
            _ = cohere.Client(API_KEY, 'fake').generate(model='medium', prompt='co:here', max_tokens=1).generations

    def test_invalid_key(self):
        with self.assertRaises(cohere.CohereError):
            _ = cohere.Client('invalid')

    def test_preset_success(self):
        prediction = co.generate(preset='SDK-PRESET-TEST-t94jfm')
        self.assertIsInstance(prediction.generations[0].text, str)

    def test_logit_bias(self):
        prediction = co.generate(model='medium', prompt='co:here', logit_bias={11: -5.5}, max_tokens=1)
        self.assertIsInstance(prediction.generations[0].text, str)
        self.assertIsNone(prediction.generations[0].token_likelihoods)
        self.assertEqual(prediction.return_likelihoods, None)

    def test_prompt_vars(self):
        prediction = co.generate(prompt='Hello {{ name }}', prompt_vars={'name': 'Aidan'})
        self.assertIsInstance(prediction.generations[0].text, str)
