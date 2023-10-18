import unittest

from utils import get_api_key

import cohere

API_KEY = get_api_key()
client = cohere.Client(API_KEY)


class TestFinetuneClient(unittest.TestCase):
    def test_list(self):
        self.assertTrue(len(client.list_custom_models()) > 0)

    def test_get(self):
        custom_models = client.list_custom_models()
        for model in custom_models:
            try:
                by_id = client.get_custom_model(model.id)
                self.assertEqual(model.id, by_id.id)
            except cohere.error.CohereAPIError:
                continue
        raise self.failureException("no custom finetunes found")

    def test_metrics(self):
        models = client.list_custom_models(statuses=["PAUSED", "READY"])
        # there should always be a model, but make sure tests don't randomly break
        if models:
            metrics = client.get_custom_model_metrics(models[0].id)
            self.assertNotEquals(metrics, [])
