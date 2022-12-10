import unittest

from utils import get_api_key
import cohere
from cohere.classify import Example


class TestAsyncClassify(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        self._co = await cohere.AsyncClient.create(get_api_key())

    async def asyncTearDown(self):
        await self._co.close_connection()

    async def test_success(self):
        prediction = await self._co.classify(model='medium',
                inputs=['purple'],
                examples=[
                    Example('apple', 'fruit'),
                    Example('banana', 'fruit'),
                    Example('cherry', 'fruit'),
                    Example('watermelon', 'fruit'),
                    Example('kiwi', 'fruit'),
                    Example('red', 'color'),
                    Example('blue', 'color'),
                    Example('green', 'color'),
                    Example('yellow', 'color'),
                    Example('magenta', 'color')
                    ])
        self.assertIsInstance(prediction.classifications, list)
        self.assertIsInstance(prediction.classifications[0].input, str)
        self.assertIsInstance(prediction.classifications[0].prediction, str)
        self.assertIsInstance(prediction.classifications[0].confidence, (int, float))
        self.assertIsInstance(prediction.classifications[0].labels['color'].confidence, (int, float))
        self.assertEqual(len(prediction.classifications[0].labels), 2)
        self.assertEqual(len(prediction.classifications), 1)
        self.assertEqual(prediction.classifications[0].prediction, 'color')

    async def test_when_empty_then_raise_error(self):
        with self.assertRaises(cohere.CohereError):
            await self._co.classify(model='medium',
                inputs=[],
                examples=[
                    Example('apple', 'fruit'),
                    Example('banana', 'fruit'),
                    Example('cherry', 'fruit'),
                    Example('watermelon', 'fruit'),
                    Example('kiwi', 'fruit'),
                    Example('red', 'color'),
                    Example('blue', 'color'),
                    Example('green', 'color'),
                    Example('yellow', 'color'),
                    Example('magenta', 'color')
                ])

    async def test_success_multi_input(self):
        prediction = await self._co.classify(model='medium',
            inputs=['purple', 'mango'],
            examples=[
            Example('apple', 'fruit'),
            Example('banana', 'fruit'),
            Example('cherry', 'fruit'),
            Example('watermelon', 'fruit'),
            Example('kiwi', 'fruit'),
            Example('red', 'color'),
            Example('blue', 'color'),
            Example('green', 'color'),
            Example('yellow', 'color'),
            Example('magenta', 'color')
        ])
        self.assertEqual(prediction.classifications[0].prediction, 'color')
        self.assertEqual(prediction.classifications[1].prediction, 'fruit')
        self.assertEqual(len(prediction.classifications), 2)

    async def test_success_all_fields(self):
        prediction = await self._co.classify(model='medium',
            inputs=['mango', 'purple'],
            examples=[
            Example('apple', 'fruit'),
            Example('banana', 'fruit'),
            Example('cherry', 'fruit'),
            Example('watermelon', 'fruit'),
            Example('kiwi', 'fruit'),
            Example('red', 'color'),
            Example('blue', 'color'),
            Example('green', 'color'),
            Example('yellow', 'color'),
            Example('magenta', 'color')
        ])
        self.assertEqual(prediction.classifications[0].prediction, 'fruit')
        self.assertEqual(prediction.classifications[1].prediction, 'color')

    async def test_preset_success(self):
        prediction = await self._co.classify(preset='SDK-TESTS-PRESET-rfa6h3')
        self.assertIsInstance(prediction.classifications, list)


if __name__ == '__main__':
    unittest.main()
