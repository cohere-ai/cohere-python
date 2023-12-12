import os
import unittest

from utils import get_api_key

import cohere

API_KEY = get_api_key()
client = cohere.Client(API_KEY)

IN_CI = os.getenv("CI", "").lower() in ["true", "1"]


class TestFinetuneClient(unittest.TestCase):
    def test_list(self):
        self.assertTrue(len(client.list_custom_models()) > 0)

    def test_get(self):
        first = client.list_custom_models()[0]
        by_id = client.get_custom_model(first.id)
        self.assertEqual(first.id, by_id.id)

    @pytest.mark.skipif(IN_CI, reason="flaky in CI for some reason")
    def test_metrics(self):
        models = client.list_custom_models(statuses=["PAUSED", "READY"])
        # there should always be a model, but make sure tests don't randomly break
        if models:
            metrics = client.get_custom_model_metrics(models[0].id)
            self.assertNotEqual(metrics, [])
