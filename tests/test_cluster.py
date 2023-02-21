import time
import unittest

from utils import get_api_key

import cohere

INPUT_FILE = "gs://cohere-dev-central-2/cluster_tests/all_datasets/reddit_500.jsonl"


class TestClient(unittest.TestCase):

    def test_cluster_job(self):
        co = cohere.Client(get_api_key(), client_name='test')
        create_res = co.create_cluster_job(
            INPUT_FILE,
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

        assert job.status == 'complete'
        assert job.output_clusters_url is not None
        assert job.output_outliers_url is not None

    def test_wait_succeeds(self):
        co = cohere.Client(get_api_key(), client_name='test')
        create_res = co.create_cluster_job(
            INPUT_FILE,
            min_cluster_size=3,
            threshold=0.5,
        )

        job = co.wait_cluster_job(create_res.job_id, timeout=60, interval=5)
        assert job.status == 'complete'
        assert job.output_clusters_url is not None
        assert job.output_outliers_url is not None

    def test_wait_times_out(self):
        co = cohere.Client(get_api_key(), client_name='test')
        create_res = co.create_cluster_job(
            INPUT_FILE,
            min_cluster_size=3,
            threshold=0.5,
        )

        def wait():
            co.wait_cluster_job(create_res.job_id, timeout=5, interval=2)

        self.assertRaises(TimeoutError, wait)

    def test_handler_wait_succeeds(self):
        co = cohere.Client(get_api_key(), client_name='test')
        create_res = co.create_cluster_job(
            INPUT_FILE,
            min_cluster_size=3,
            threshold=0.5,
        )

        job = create_res.wait(timeout=60, interval=5)
        assert job.status == 'complete'
        assert job.output_clusters_url is not None
        assert job.output_outliers_url is not None

    def test_handler_wait_times_out(self):
        co = cohere.Client(get_api_key(), client_name='test')
        create_res = co.create_cluster_job(
            INPUT_FILE,
            min_cluster_size=3,
            threshold=0.5,
        )

        def wait():
            create_res.wait(timeout=5, interval=2)

        self.assertRaises(TimeoutError, wait)
