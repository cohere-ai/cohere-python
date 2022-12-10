import unittest

from utils import get_api_key
import cohere

API_KEY = get_api_key()


class TestAsyncGenerate(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        self._co = await cohere.AsyncClient.create(API_KEY)

    async def asyncTearDown(self):
        await self._co.close_connection()

    async def test_success(self):
        prediction = await self._co.generate(
            model='small', prompt='co:here', max_tokens=1)
        self.assertIsInstance(prediction.generations[0].text, str)
        self.assertIsNone(prediction.generations[0].token_likelihoods)
        self.assertEqual(prediction.return_likelihoods, None)

    async def test_success_batched(self):
        _batch_size = 10
        predictions = await self._co.batch_generate(
            model='small', prompts=['co:here'] * _batch_size, max_tokens=1)
        for prediction in predictions:
            self.assertIsInstance(prediction.generations[0].text, str)
            self.assertIsNone(prediction.generations[0].token_likelihoods)
            self.assertEqual(prediction.return_likelihoods, None)

    async def test_return_likelihoods_generation(self):
        prediction = await self._co.generate(
            model='small', prompt='co:here',
            max_tokens=1, return_likelihoods='GENERATION')
        self.assertTrue(prediction.generations[0].token_likelihoods)
        self.assertTrue(prediction.generations[0].token_likelihoods[0].token)
        self.assertIsNotNone(prediction.generations[0].likelihood)
        self.assertEqual(prediction.return_likelihoods, 'GENERATION')

    async def test_return_likelihoods_all(self):
        prediction = await self._co.generate(
            model='small', prompt='hi',
            max_tokens=1, return_likelihoods='ALL')
        self.assertEqual(len(prediction.generations[0].token_likelihoods), 2)
        self.assertIsNotNone(prediction.generations[0].likelihood)
        self.assertEqual(prediction.return_likelihoods, 'ALL')

    async def test_invalid_temp(self):
        with self.assertRaises(cohere.CohereError):
            await self._co.generate(
                model='large', prompt='hi', max_tokens=1, temperature=-1
            )

    async def test_invalid_model(self):
        with self.assertRaises(cohere.CohereError):
            await self._co.generate(
                model='this-better-not-exist', prompt='co:here', max_tokens=1
            )

    async def test_invalid_version_fails(self):
        cl = await cohere.AsyncClient.create(API_KEY, 'fake')
        with self.assertRaises(cohere.CohereError):
            await cl.generate(model='small', prompt='co:here', max_tokens=1)

    async def test_invalid_key(self):
        with self.assertRaises(cohere.CohereError):
            _ = await cohere.AsyncClient.create('invalid')

    async def test_preset_success(self):
        prediction = await self._co.generate(preset='SDK-TESTS-PRESET-cq2r57')
        self.assertIsInstance(prediction.generations[0].text, str)

    async def test_logit_bias(self):
        prediction = await self._co.generate(
            model='small', prompt='co:here', logit_bias={11: -5.5}, max_tokens=1)
        self.assertIsInstance(prediction.generations[0].text, str)
        self.assertIsNone(prediction.generations[0].token_likelihoods)
        self.assertEqual(prediction.return_likelihoods, None)

    async def test_prompt_vars(self):
        prediction = await self._co.generate(
                prompt='Hello {{ name }}', prompt_vars={'name': 'Aidan'})
        self.assertIsInstance(prediction.generations[0].text, str)
