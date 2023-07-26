import io
import json
import time
import unittest
from typing import Optional

from utils import get_api_key

import cohere
from cohere.responses import Dataset


class TestDataset(unittest.TestCase):
    def test_create_dataset(self):
        co = self.create_co()
        dataset = co.create_dataset(
            name="ci-test",
            data=self.dummy_file(
                [
                    {"text": "this is a text"},
                    {"text": "this is another text"},
                ]
            ),
            dataset_type="embed-input",
        )

        start = time.time()
        while not dataset.has_terminal_status():
            if time.time() - start > 120:  # 120s timeout
                raise TimeoutError()
            time.sleep(5)
            dataset = co.get_dataset(dataset.id)

        self.check_result(dataset, status="validated")

    def test_create_invalid_dataset(self):
        co = self.create_co()
        dataset = co.create_dataset(
            name="ci-test",
            data=self.dummy_file(
                [
                    {"foo": "bar"},
                    {"baz": "foz"},
                ]
            ),
            dataset_type="embed-input",
        )

        start = time.time()
        while not dataset.has_terminal_status():
            if time.time() - start > 120:  # 120s timeout
                raise TimeoutError()
            time.sleep(5)
            dataset = co.get_dataset(dataset.id)

        self.check_result(dataset, status="failed")

    def test_get_dataset(self):
        co = self.create_co()
        datasets = co.list_datasets()
        dataset = co.get_dataset(datasets[0].id)
        self.check_result(dataset)

    def test_list_dataset(self):
        co = self.create_co()
        datasets = co.list_datasets()
        assert len(datasets) > 0
        for dataset in datasets:
            self.check_result(dataset)

    def dummy_file(self, data) -> io.BytesIO:
        final = ""
        for t in data:
            final += json.dumps(t) + "\n"

        binaryData = final.encode()
        vfile = io.BytesIO(binaryData)
        vfile.name = "test.jsonl"
        return vfile

    def create_co(self) -> cohere.Client:
        return cohere.Client(get_api_key(), check_api_key=False, client_name="test")

    def check_result(self, dataset: Dataset, status: Optional[str] = None):
        assert dataset.id
        assert dataset.created_at
        assert dataset.dataset_type
        assert dataset.name

        if status is not None:
            assert dataset.validation_status == status

        if status == "validated":
            assert dataset.urls
            for row in dataset.open():
                assert row

        if status == "failed":
            assert dataset.validation_error
