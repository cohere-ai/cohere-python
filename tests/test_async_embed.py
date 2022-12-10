import unittest

import random
import string

from utils import get_api_key
import cohere


def random_word():
    return ''.join(random.choice(string.ascii_lowercase) for _ in range(10))


def random_sentence(num_words):
    sentence = ''
    for _ in range(num_words):
        sentence += random_word() + ' '
    return sentence


def random_texts(num_texts, num_words_per_sentence=50):
    arr = []
    for _ in range(num_texts):
        arr.append(random_sentence(num_words_per_sentence))
    return arr


class TestAsyncEmbed(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        self._co = await cohere.AsyncClient.create(get_api_key())

    async def asyncTearDown(self):
        await self._co.close_connection()

    async def test_success(self):
        prediction = await self._co.embed(model='small', texts=['co:here', 'cohere'])
        self.assertEqual(len(prediction.embeddings), 2)
        self.assertIsInstance(prediction.embeddings[0], list)
        self.assertIsInstance(prediction.embeddings[1], list)
        self.assertEqual(len(prediction.embeddings[0]), 1024)
        self.assertEqual(len(prediction.embeddings[1]), 1024)

    async def test_default_model(self):
        prediction = await self._co.embed(texts=['co:here', 'cohere'])
        self.assertEqual(len(prediction.embeddings), 2)
        self.assertIsInstance(prediction.embeddings[0], list)
        self.assertIsInstance(prediction.embeddings[1], list)

    async def test_success_multiple_batches(self):
        prediction = await self._co.embed(
            model='small',
            texts=['co:here', 'cohere', 'embed', 'python', 'golang', 'typescript', 'rust?', 'ai', 'nlp', 'neural'])
        self.assertEqual(len(prediction.embeddings), 10)
        for embed in prediction.embeddings:
            self.assertIsInstance(embed, list)
            self.assertEqual(len(embed), 1024)

    async def test_success_longer_multiple_batches_unaligned_batch(self):
        prediction = await self._co.embed(
            model='small',
            texts=[
                'co:here', 'cohere', 'embed', 'python', 'golang',
                'typescript', 'rust?', 'ai', 'nlp',
                'neural', 'nets'
                ])
        self.assertEqual(len(prediction.embeddings), 11)
        for embed in prediction.embeddings:
            self.assertIsInstance(embed, list)
            self.assertEqual(len(embed), 1024)

    async def test_success_longer_multiple_batches(self):
        prediction = await self._co.embed(
            model='small',
            texts=['co:here', 'cohere', 'embed', 'python', 'golang'] * 200)
        self.assertEqual(len(prediction.embeddings), 200 * 5)
        for embed in prediction.embeddings:
            self.assertIsInstance(embed, list)
            self.assertEqual(len(embed), 1024)

    async def test_success_multiple_batches_in_order(self):
        textAll = []
        predictionsExpected = []

        for _ in range(3):
            text_batch = random_texts(cohere.COHERE_EMBED_BATCH_SIZE)
            prediction = await self._co.embed(model='small', texts=text_batch)
            textAll.extend(text_batch)
            predictionsExpected.extend(prediction)
        predictionsActual = await self._co.embed(model='small', texts=textAll)
        for predictionExpected, predictionActual in zip(predictionsExpected, list(predictionsActual)):
            for elementExpected, elementAcutal in zip(predictionExpected, predictionActual):
                self.assertAlmostEqual(elementExpected, elementAcutal, places=1)

    async def test_invalid_texts(self):
        with self.assertRaises(cohere.CohereError):
            await self._co.embed(model='small', texts=[''])


if __name__ == '__main__':
    unittest.main()
