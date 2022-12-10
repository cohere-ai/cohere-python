import unittest

from utils import get_api_key
import cohere


class TestAsyncDetokenize(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        self._co = await cohere.AsyncClient.create(get_api_key())

    async def asyncTearDown(self):
        await self._co.close_connection()


    async def test_success(self):
        text = await self._co.detokenize([10104, 12221, 974, 514, 34]).text
        self.assertEqual(text, "detokenize me!")

    async def test_success_batched(self):
        _batch_size = 10
        texts = await self._co.batch_detokenize(
            [[10104, 12221, 974, 514, 34]] * _batch_size)
        results = []
        for text in texts:
            results.append(str(text))
        self.assertEqual(results, ["detokenize me!"] * _batch_size)

    async def test_empty_tokens(self):
        text = await self._co.detokenize([]).text
        self.assertEqual(text, "")
