import os
import unittest
import cohere

API_KEY = os.getenv('CO_API_KEY')
assert type(API_KEY)
co = cohere.Client(str(API_KEY))


class TestGenerate(unittest.TestCase):
    def test_success(self):
        prediction = co.generate(
            model='small',
            prompt='co:here',
            max_tokens=1)
        self.assertIsInstance(prediction.generations[0].text, str)
        self.assertIsNone(prediction.generations[0].token_likelihoods)
        self.assertEqual(prediction.return_likelihoods, 'NONE')

    def test_return_likelihoods_generation(self):
        prediction = co.generate(
            model='small',
            prompt='co:here',
            max_tokens=1,
            return_likelihoods='GENERATION')
        self.assertTrue(prediction.generations[0].token_likelihoods)
        self.assertTrue(prediction.generations[0].token_likelihoods[0].token)
        self.assertIsNotNone(prediction.generations[0].likelihood)
        self.assertEqual(prediction.return_likelihoods, 'GENERATION')

    def test_return_likelihoods_all(self):
        prediction = co.generate(
            model='small',
            prompt='hi',
            max_tokens=1,
            return_likelihoods='ALL')
        self.assertEqual(len(prediction.generations[0].token_likelihoods), 2)
        self.assertIsNotNone(prediction.generations[0].likelihood)
        self.assertEqual(prediction.return_likelihoods, 'ALL')

    def test_invalid_temp(self):
        with self.assertRaises(cohere.CohereError):
            co.generate(
                model='large',
                prompt='hi',
                max_tokens=1,
                temperature=-1)

    def test_invalid_model(self):
        with self.assertRaises(cohere.CohereError):
            co.generate(
                model='this-better-not-exist',
                prompt='co:here',
                max_tokens=1)

    def test_no_version_works(self):
        cohere.Client(API_KEY).generate(
            model='small',
            prompt='co:here',
            max_tokens=1)

    def test_invalid_version_fails(self):
        with self.assertRaises(cohere.CohereError):
            cohere.Client(API_KEY, 'fake').generate(
                model='small',
                prompt='co:here',
                max_tokens=1)

    def test_invalid_key(self):
        with self.assertRaises(cohere.CohereError):
            _ = cohere.Client('invalid')
