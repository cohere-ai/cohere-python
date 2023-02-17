from typing import Any, Optional

from cohere.response import CohereObject


class CreateClusterJobResponse(CohereObject):
    job_id: str

    def __init__(self, job_id: str):
        self.job_id = job_id


class GetClusterJobResponse(CohereObject):
    job_id: str
    status: str
    output_clusters_url: Optional[str]
    output_outliers_url: Optional[str]
    clusters: Any

    def __init__(
        self,
        job_id: str,
        status: str,
        output_clusters_url: Optional[str],
        output_outliers_url: Optional[str],
        clusters: Any,
    ):
        # convert empty string to `None`
        if not output_clusters_url:
            output_clusters_url = None
        if not output_outliers_url:
            output_outliers_url = None

        self.job_id = job_id
        self.status = status
        self.output_clusters_url = output_clusters_url
        self.output_outliers_url = output_outliers_url
        self.clusters = clusters
