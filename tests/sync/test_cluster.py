import unittest
from typing import Optional

from utils import get_api_key

import cohere
from cohere.responses import ClusterJobResult


class TestClient(unittest.TestCase):
    def test_list_get_cluster_job(self):
        co = self.create_co()
        jobs = co.list_cluster_jobs()
        if len(jobs) == 0:
            self.skipTest("no jobs to test")

        for job in jobs:
            self.check_job_result(job)
        job = co.get_cluster_job(jobs[0].job_id)
        self.check_job_result(job)

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
