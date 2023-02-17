import time
import unittest

from utils import get_api_key

import cohere


class TestClient(unittest.TestCase):

    def test_cluster_job(self):
        co = cohere.Client(get_api_key(), client_name='test')
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

        assert job.status == 'complete'
        assert job.output_clusters_url is not None
