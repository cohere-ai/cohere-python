import unittest
import cohere
from cohere.classify import Example, LabelPrediction
from utils import get_api_key

co = cohere.Client(get_api_key())


class TestClassify(unittest.TestCase):
    def test_success(self):
        prediction = co.classify(
            model='medium',
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
                Example('magenta', 'color')])
        self.assertIsInstance(prediction.classifications, list)
        self.assertIsInstance(prediction.classifications[0].input, str)
        self.assertIsInstance(prediction.classifications[0].prediction, dict)
        self.assertIsInstance(prediction.classifications[0].labels, dict)
        self.assertIsInstance(prediction.classifications[0].labels['fruit'].confidence, float)
        self.assertIsInstance(prediction.classifications[0].labels['fruit'], LabelPrediction)
        self.assertEqual(len(prediction.classifications), 1)
        self.assertEqual(prediction.classifications[0].prediction_label, 'color')

    def test_empty_inputs(self):
        with self.assertRaises(cohere.CohereError):
            co.classify(
                model='medium',
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
                    Example('magenta', 'color')])

    def test_success_multi_input(self):
        prediction = co.classify(
            model='medium',
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
                Example('magenta', 'color')])
        self.assertEqual(prediction.classifications[0].prediction_label, 'color')
        self.assertEqual(prediction.classifications[1].prediction_label, 'fruit')
        self.assertEqual(len(prediction.classifications), 2)

    def test_success_all_fields(self):
        prediction = co.classify(
            model='medium',
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
            ],
            taskDescription='this is a classifier to determine if a word is a fruit of a color',
            outputIndicator='This is a')
        self.assertEqual(prediction.classifications[0].prediction_label, 'fruit')
        self.assertEqual(prediction.classifications[1].prediction_label, 'color')
