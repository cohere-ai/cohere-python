import time
import unittest
from typing import Optional

from utils import get_api_key, in_ci

import cohere
from cohere.responses.bulk_embed import BulkEmbedJob

# from cohere.bu import ClusterJobResult
# from cohere.bu import Job

VALID_INPUT_FILE = "gs://cohere-dev-central-2/cluster_tests/all_datasets/reddit_500.jsonl"
# BAD_INPUT_FILE = "./local-file.jsonl"


class TestClient(unittest.TestCase):
    def test_create_job(self):
        co = self.create_co()
        res = co.create_bulk_embed_job(input_file_url=VALID_INPUT_FILE)
        assert res.job_id

    def test_list_jobs(self):
        co = self.create_co()
        jobs = co.list_bulk_embed_jobs()
        assert len(jobs) > 0
        for job in jobs:
            self.check_job_result(job)

    def test_get_job(self):
        co = self.create_co()
        jobs = co.list_bulk_embed_jobs()
        job = co.get_bulk_embed_job(jobs[0].job_id)
        self.check_job_result(job)

    def test_cancel_job(self):
        co = self.create_co()
        jobs = co.list_bulk_embed_jobs()
        co.cacnel_bulk_embed_job(jobs[0].job_id)

    def create_co(self) -> cohere.Client:
        return cohere.Client(get_api_key(), check_api_key=False, client_name="test")

    def check_job_result(self, job: BulkEmbedJob):
        assert job.job_id
        assert job.status
        assert job.created_at
        assert job.input_url
        assert job.model
        assert job.truncate
