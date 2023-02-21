import time
import unittest

from utils import get_api_key, in_ci

import cohere

INPUT_FILE = "gs://cohere-dev-central-2/cluster_tests/all_datasets/reddit_500.jsonl"


class TestClient(unittest.TestCase):

    @unittest.skipIf(in_ci(), "can sometimes fail due to duration variation")
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

    @unittest.skipIf(in_ci(), "can sometimes fail due to duration variation")
    def test_wait_succeeds(self):
        co = cohere.Client(get_api_key(), client_name='test')
        create_res = co.create_cluster_job(
            INPUT_FILE,
            min_cluster_size=3,
            threshold=0.5,
        )

        job = co.wait_for_cluster_job(create_res.job_id, timeout=60, interval=5)
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
            co.wait_for_cluster_job(create_res.job_id, timeout=5, interval=2)

        self.assertRaises(TimeoutError, wait)

    @unittest.skipIf(in_ci(), "can sometimes fail due to duration variation")
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
