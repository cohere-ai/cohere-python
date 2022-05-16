import os
import unittest
import cohere
from cohere.classify import Example

API_KEY = os.getenv('CO_API_KEY')
assert type(API_KEY)
co = cohere.Client(str(API_KEY))


class TestClassify(unittest.TestCase):
    def test_success(self):
        prediction = co.classify('medium', ['purple'], [
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
        self.assertIsInstance(prediction.classifications[0].confidence[0].confidence, (int, float))
        self.assertIsInstance(prediction.classifications[0].confidence[0].label, str)
        self.assertIsInstance(prediction.classifications[0].confidence[1].confidence, (int, float))
        self.assertIsInstance(prediction.classifications[0].confidence[1].label, str)
        self.assertEqual(len(prediction.classifications), 1)
        self.assertEqual(prediction.classifications[0].prediction, 'color')

    def test_empty_inputs(self):
        with self.assertRaises(cohere.CohereError):
            co.classify(
                'medium', [], [

                    Example('apple', 'fruit'),
                    Example('banana', 'fruit'),
                    Example('cherry', 'fruit'),
                    Example('watermelon', 'fruit'),
                    Example('kiwi', 'fruit'),

                    Example('red', 'color'),
                    Example('blue', 'color'),
                    Example('green', 'color'),
                    Example('yellow', 'color'),
                    Example('magenta', 'color')])

    def test_success_multi_input(self):
        prediction = co.classify('medium', ['purple', 'mango'], [
            Example('apple', 'fruit'),
            Example('banana', 'fruit'),
            Example('cherry', 'fruit'),
            Example('watermelon', 'fruit'),
            Example('kiwi', 'fruit'),

            Example('red', 'color'),
            Example('blue', 'color'),
            Example('green', 'color'),
            Example('yellow', 'color'),
            Example('magenta', 'color')])
        self.assertEqual(prediction.classifications[0].prediction, 'color')
        self.assertEqual(prediction.classifications[1].prediction, 'fruit')
        self.assertEqual(len(prediction.classifications), 2)

    def test_success_all_fields(self):
        prediction = co.classify('medium', ['mango', 'purple'], [
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
        ], 'this is a classifier to determine if a word is a fruit of a color', 'This is a')
        self.assertEqual(prediction.classifications[0].prediction, 'fruit')
        self.assertEqual(prediction.classifications[1].prediction, 'color')
