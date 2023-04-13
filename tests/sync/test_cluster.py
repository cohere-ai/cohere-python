import time
import unittest
from typing import Optional

from utils import get_api_key, in_ci

import cohere
from cohere.responses import ClusterJobResult

VALID_INPUT_FILE = "gs://cohere-dev-central-2/cluster_tests/all_datasets/reddit_100.jsonl"
BAD_INPUT_FILE = "./local-file.jsonl"


class TestClient(unittest.TestCase):
    @unittest.skipIf(in_ci(), "can sometimes fail due to duration variation")
    def test_create_cluster_job(self):
        co = self.create_co()
        create_res = co.create_cluster_job(
            VALID_INPUT_FILE,
            min_cluster_size=3,
            threshold=0.5,
        )
        job = co.get_cluster_job(create_res.job_id)
        start = time.time()

        while not job.is_final_state:
            if time.time() - start > 60:  # 60s timeout
                raise TimeoutError()
            time.sleep(5)
            job = co.get_cluster_job(create_res.job_id)

        self.check_job_result(job, status="complete")

    def test_get_cluster_job(self):
        co = self.create_co()
        jobs = co.list_cluster_jobs()
        job = co.get_cluster_job(jobs[0].job_id)
        self.check_job_result(job)  # not finished

    def test_list_cluster_jobs(self):
        co = self.create_co()
        jobs = co.list_cluster_jobs()
        assert len(jobs) > 0
        for job in jobs:
            self.check_job_result(job)

    @unittest.skipIf(in_ci(), "can sometimes fail due to duration variation")
    def test_wait_for_cluster_job_succeeds(self):
        co = self.create_co()
        create_res = co.create_cluster_job(
            VALID_INPUT_FILE,
            min_cluster_size=3,
            threshold=0.5,
        )

        job = co.wait_for_cluster_job(create_res.job_id, timeout=60, interval=5)
        self.check_job_result(job, status="complete")

    def test_wait_for_cluster_job_times_out(self):
        co = self.create_co()
        create_res = co.create_cluster_job(
            VALID_INPUT_FILE,
            min_cluster_size=3,
            threshold=0.5,
        )

        def wait():
            co.wait_for_cluster_job(create_res.job_id, timeout=5, interval=2)

        self.assertRaises(TimeoutError, wait)

    @unittest.skipIf(in_ci(), "can sometimes fail due to duration variation")
    def test_job_wait_method_succeeds(self):
        co = self.create_co()
        create_res = co.create_cluster_job(
            VALID_INPUT_FILE,
            min_cluster_size=3,
            threshold=0.5,
        )

        job = create_res.wait(timeout=60, interval=5)
        self.check_job_result(job, status="complete")

    def test_job_wait_method_times_out(self):
        co = self.create_co()
        create_res = co.create_cluster_job(
            VALID_INPUT_FILE,
            min_cluster_size=3,
            threshold=0.5,
        )

        def wait():
            create_res.wait(timeout=5, interval=2)

        self.assertRaises(TimeoutError, wait)

    @unittest.skipIf(in_ci(), "can sometimes fail due to duration variation")
    def test_job_fails(self):
        co = self.create_co()
        create_res = co.create_cluster_job(
            BAD_INPUT_FILE,
            min_cluster_size=3,
            threshold=0.5,
        )

        job = create_res.wait(timeout=60, interval=5)
        self.check_job_result(job, status="failed")

    def create_co(self) -> cohere.Client:
        return cohere.Client(get_api_key(), check_api_key=False, client_name="test")

    def check_job_result(self, job: ClusterJobResult, status: Optional[str] = None):
        assert job.job_id
        assert job.status
        assert job.meta
        assert job.meta["api_version"]
        assert job.meta["api_version"]["version"]

        if status is not None:
            assert job.status == status

        if status == "complete":
            assert job.output_clusters_url
            assert job.output_outliers_url
            assert job.clusters
