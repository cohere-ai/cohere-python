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

class TestEmbed(unittest.TestCase):
    def test_success(self):
        prediction = co.embed(
            model='small',
            texts=['co:here', 'cohere'])
        self.assertEqual(len(prediction.embeddings), 2)
        self.assertIsInstance(prediction.embeddings[0], list)
        self.assertIsInstance(prediction.embeddings[1], list)
        self.assertEqual(len(prediction.embeddings[0]), 1024)
        self.assertEqual(len(prediction.embeddings[1]), 1024)

    def test_invalid_texts(self):
        with self.assertRaises(cohere.CohereError):
            co.embed(
                model='small',
                texts=[''])

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

    def test_empty_options(self):
        with self.assertRaises(cohere.CohereError):
            co.choose_best(
                model='large',
                query='the best book in the world is',
                options=[],
                mode='APPEND_OPTION')

class TestTokenize(unittest.TestCase):
    def test_success(self):
        tokens = co.tokenize('large', 'tokenize me!')
        self.assertIsInstance(tokens.tokens, list)
        self.assertIsInstance(tokens.length, int)
        self.assertEqual(tokens.length, len(tokens))

    def test_invalid_text(self):
        with self.assertRaises(cohere.CohereError):
            co.tokenize(
                model='large',
                text='')

if __name__ == '__main__':
    unittest.main()
