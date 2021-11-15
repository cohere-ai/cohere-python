import os
import unittest
import cohere

API_KEY = os.getenv('CO_API_KEY')
co = cohere.Client(API_KEY)

class TestModel(unittest.TestCase):
    def test_invalid_model(self):
        with self.assertRaises(cohere.CohereError):
            co.generate(
                model='this-better-not-exist',
                prompt='co:here',
                max_tokens=1)

    def test_invalid_key(self):
        with self.assertRaises(cohere.CohereError):
            cohere.Client('invalid').generate(
                model='large',
                prompt='co:here',
                max_tokens=1)

class TestGenerate(unittest.TestCase):
    def test_success(self):
        prediction = co.generate(
            model='large',
            prompt='co:here',
            max_tokens=1)
        self.assertIsInstance(prediction.texts, list)
        self.assertIsNone(prediction.token_likelihoods)
        self.assertEqual(prediction.return_likelihoods, 'NONE')

    def test_return_likelihoods_generation(self):
        prediction = co.generate(
            model='large',
            prompt='co:here',
            max_tokens=1, 
            return_likelihoods='GENERATION')
        self.assertEqual(len(prediction.token_likelihoods), 1)
        self.assertEqual(prediction.return_likelihoods, 'GENERATION')

    def test_return_likelihoods_all(self):
        prediction = co.generate(
            model='large',
            prompt='hi',
            max_tokens=1, 
            return_likelihoods='ALL')
        self.assertEqual(len(prediction.token_likelihoods), 2)
        self.assertEqual(prediction.return_likelihoods, 'ALL')
        
    def test_invalid_temp(self):
        with self.assertRaises(cohere.CohereError):
            co.generate(
                model='large',
                prompt='hi',
                max_tokens=1, 
                temperature=-1)

class TestSimilarity(unittest.TestCase):
    def test_success(self):
        prediction = co.similarity(
            model='medium',
           	anchor='cohere ai',
            targets=['co:here', 'cohere'])
        self.assertEqual(len(prediction.similarities), 2)
        self.assertIsInstance(prediction.similarities[0], float)
        self.assertIsInstance(prediction.similarities[1], float)

    def test_invalid_targets(self):
        with self.assertRaises(cohere.CohereError):
            co.similarity(
                model='medium',
                anchor='cohere ai',
                targets=[''])

class TestEmbed(unittest.TestCase):
    def test_success(self):
        prediction = co.embed(
            model='medium',
            texts=['co:here', 'cohere'])
        self.assertEqual(len(prediction.embeddings), 2)
        self.assertIsInstance(prediction.embeddings[0], list)
        self.assertIsInstance(prediction.embeddings[1], list)
        self.assertEqual(len(prediction.embeddings[0]), 768)
        self.assertEqual(len(prediction.embeddings[1]), 768)

    def test_invalid_texts(self):
        with self.assertRaises(cohere.CohereError):
            co.embed(
                model='medium',
                texts=[''])

class TestLikelihood(unittest.TestCase):
    def test_success(self):
        prediction = co.likelihood(
            model='large',
            text='hi')
        self.assertIsInstance(prediction.likelihood, int)
        self.assertEqual(len(prediction.token_likelihoods), 1)
        self.assertIsInstance(prediction.token_likelihoods[0], dict)
        self.assertIsInstance(prediction.token_likelihoods[0]['token'], str)

    def test_invalid_text(self):
        with self.assertRaises(cohere.CohereError):
            co.likelihood(
                model='large',
                text='')


class TestChooseBest(unittest.TestCase):
    def test_success(self):
        prediction = co.choose_best(
            model='large',
            query='Carol picked up a book and walked to the kitchen. She set it down, picked up her glasses and left. This is in the kitchen now: ',
            options=['book', 'glasses', 'dog'],
            mode='APPEND_OPTION')
        self.assertEqual(len(prediction.scores), 3)
        self.assertIsInstance(prediction.scores[0], float)
        self.assertEqual(len(prediction.tokens), 3)
        self.assertIsInstance(prediction.tokens[0], list)
        self.assertIsInstance(prediction.tokens[0][0], str)
        self.assertEqual(len(prediction.token_log_likelihoods), 3)
        self.assertIsInstance(prediction.token_log_likelihoods[0][0], float)

    def test_invalid_text(self):
        with self.assertRaises(cohere.CohereError):
            co.likelihood(
                model='large',
                text='')

if __name__ == '__main__':
    unittest.main()
