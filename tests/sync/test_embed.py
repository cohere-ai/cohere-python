import random
import string
import unittest

from utils import get_api_key

import cohere

# API_KEY = get_api_key()

# 1. for oci
API_KEY = "oci"
MODEL = "cohere.embed-english-light-v2.0"
# 2. for cohere
# API_KEY = "TODO"
# MODEL = "small"

co = cohere.Client(API_KEY)


def random_word():
    return "".join(random.choice(string.ascii_lowercase) for _ in range(10))


def random_sentence(num_words):
    sentence = ""

    for _ in range(num_words):
        sentence += random_word() + " "

    return sentence


def random_texts(num_texts, num_words_per_sentence=50):
    arr = []

    for _ in range(num_texts):
        arr.append(random_sentence(num_words_per_sentence))

    return arr


class TestEmbed(unittest.TestCase):
    def test_success(self):
        prediction = co.embed(model=MODEL, texts=["co:here", "cohere"])
        self.assertEqual(len(prediction.embeddings), 2)
        self.assertIsInstance(prediction.embeddings[0], list)
        self.assertIsInstance(prediction.embeddings[1], list)
        self.assertEqual(len(prediction.embeddings[0]), 1024)
        self.assertEqual(len(prediction.embeddings[1]), 1024)
        self.assertTrue(prediction.meta)
        print(type(prediction.meta))
        print(prediction.meta)
        self.assertTrue(prediction.meta["api_version"])
        self.assertTrue(prediction.meta["api_version"]["version"])

    def test_default_model(self):
        prediction = co.embed(texts=["co:here", "cohere"])
        self.assertEqual(len(prediction.embeddings), 2)
        self.assertIsInstance(prediction.embeddings[0], list)
        self.assertIsInstance(prediction.embeddings[1], list)

    def test_success_multiple_batches(self):
        prediction = co.embed(
            model=MODEL,
            texts=["co:here", "cohere", "embed", "python", "golang", "typescript", "rust?", "ai", "nlp", "neural"],
        )
        self.assertEqual(len(prediction.embeddings), 10)
        for embed in prediction.embeddings:
            self.assertIsInstance(embed, list)
            self.assertEqual(len(embed), 1024)

    def test_success_longer_multiple_batches_unaligned_batch(self):
        prediction = co.embed(
            model=MODEL,
            texts=[
                "co:here",
                "cohere",
                "embed",
                "python",
                "golang",
                "typescript",
                "rust?",
                "ai",
                "nlp",
                "neural",
                "nets",
            ],
        )
        self.assertEqual(len(prediction.embeddings), 11)
        for embed in prediction.embeddings:
            self.assertIsInstance(embed, list)
            self.assertEqual(len(embed), 1024)

    def test_success_longer_multiple_batches(self):
        prediction = co.embed(model=MODEL, texts=["co:here", "cohere", "embed", "python", "golang"] * 200)
        self.assertEqual(len(prediction.embeddings), 200 * 5)
        for embed in prediction.embeddings:
            self.assertIsInstance(embed, list)
            self.assertEqual(len(embed), 1024)

    def test_success_multiple_batches_in_order(self):
        textAll = []
        predictionsExpected = []

        for _ in range(3):
            text_batch = random_texts(cohere.COHERE_EMBED_BATCH_SIZE)
            prediction = co.embed(model=MODEL, texts=text_batch)
            textAll.extend(text_batch)
            predictionsExpected.extend(prediction)
        predictionsActual = co.embed(model=MODEL, texts=textAll)
        for predictionExpected, predictionActual in zip(predictionsExpected, list(predictionsActual)):
            for elementExpected, elementAcutal in zip(predictionExpected, predictionActual):
                self.assertAlmostEqual(elementExpected, elementAcutal, places=1)
