import unittest

from utils import get_api_key

import cohere
from cohere.responses.classify import Example

co = cohere.Client(get_api_key())


class TestClassify(unittest.TestCase):
    def test_success(self):
        prediction = co.classify(
            model="small",
            inputs=["purple"],
            examples=[
                Example("apple", "fruit"),
                Example("banana", "fruit"),
                Example("cherry", "fruit"),
                Example("watermelon", "fruit"),
                Example("kiwi", "fruit"),
                Example("red", "color"),
                Example("blue", "color"),
                Example("green", "color"),
                Example("yellow", "color"),
                Example("magenta", "color"),
            ],
        )
        self.assertIsInstance(prediction.classifications, list)
        self.assertIsInstance(prediction.classifications[0].input, str)
        self.assertIsInstance(prediction.classifications[0].prediction, str)
        self.assertIsInstance(prediction.classifications[0].confidence, (int, float))
        self.assertIsInstance(prediction.classifications[0].labels["color"].confidence, (int, float))
        self.assertTrue(prediction.meta)
        self.assertTrue(prediction.meta["api_version"])
        self.assertTrue(prediction.meta["api_version"]["version"])
        self.assertEqual(len(prediction.classifications[0].labels), 2)
        self.assertEqual(len(prediction.classifications), 1)
        self.assertEqual(prediction.classifications[0].prediction, "color")

    def test_empty_inputs(self):
        with self.assertRaises(cohere.CohereError):
            co.classify(
                model="small",
                inputs=[],
                examples=[
                    Example("apple", "fruit"),
                    Example("banana", "fruit"),
                    Example("cherry", "fruit"),
                    Example("watermelon", "fruit"),
                    Example("kiwi", "fruit"),
                    Example("red", "color"),
                    Example("blue", "color"),
                    Example("green", "color"),
                    Example("yellow", "color"),
                    Example("magenta", "color"),
                ],
            )

    def test_success_multi_input(self):
        prediction = co.classify(
            model="small",
            inputs=["purple", "mango"],
            examples=[
                Example("apple", "fruit"),
                Example("banana", "fruit"),
                Example("cherry", "fruit"),
                Example("watermelon", "fruit"),
                Example("kiwi", "fruit"),
                Example("red", "color"),
                Example("blue", "color"),
                Example("green", "color"),
                Example("yellow", "color"),
                Example("magenta", "color"),
            ],
        )
        self.assertEqual(prediction.classifications[0].prediction, "color")
        self.assertEqual(prediction.classifications[1].prediction, "fruit")
        self.assertEqual(len(prediction.classifications), 2)

    def test_success_all_fields(self):
        prediction = co.classify(
            model="small",
            inputs=["mango", "purple"],
            examples=[
                Example("apple", "fruit"),
                Example("banana", "fruit"),
                Example("cherry", "fruit"),
                Example("watermelon", "fruit"),
                Example("kiwi", "fruit"),
                Example("red", "color"),
                Example("blue", "color"),
                Example("green", "color"),
                Example("yellow", "color"),
                Example("magenta", "color"),
            ],
        )
        self.assertEqual(prediction.classifications[0].prediction, "fruit")
        self.assertEqual(prediction.classifications[1].prediction, "color")

    @pytest.mark.skipif(os.getenv("CO_API_URL"), reason="relies on preset existing in prod")
    def test_preset_success(self):
        prediction = co.classify(preset="SDK-TESTS-PRESET-rfa6h3")
        self.assertIsInstance(prediction.classifications, list)
