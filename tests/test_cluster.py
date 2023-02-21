import time
import unittest

from utils import get_api_key

import cohere


class TestClient(unittest.TestCase):

    def test_cluster_job(self):
        co = self.create_co()
        create_res = co.create_cluster_job(
            "gs://cohere-dev-central-2/cluster_tests/all_datasets/reddit_500.jsonl",
            min_cluster_size=3,
            threshold=0.5,
        )
        job = co.get_cluster_job(create_res.job_id)
        start = time.time()

        while job.status == 'processing':
            if time.time() - start > 60:  # 60s timeout
                raise TimeoutError()
            time.sleep(5)
            job = co.get_cluster_job(create_res.job_id)

        assert job.job_id
        assert job.status == 'complete'
        assert job.output_clusters_url
        assert job.output_outliers_url

    def test_list_jobs(self):
        co = self.create_co()
        jobs = co.get_cluster_jobs()
        assert len(jobs) > 0
        for job in jobs:
            assert job.job_id

    def create_co(self) -> cohere.Client:
        return cohere.Client(get_api_key(), check_api_key=False, client_name='test')
