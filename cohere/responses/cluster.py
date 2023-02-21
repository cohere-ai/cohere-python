from typing import Optional

from cohere.responses.base import CohereObject


class CreateClusterJobResponse(CohereObject):
    job_id: str

    def __init__(self, job_id: str):
        self.job_id = job_id


class GetClusterJobResponse(CohereObject):
    status: str
    output_clusters_url: Optional[str]
    output_outliers_url: Optional[str]

    def __init__(self, status: str, output_clusters_url: Optional[str], output_outliers_url: Optional[str]):
        # convert empty string to `None`
        if not output_clusters_url:
            output_clusters_url = None
        if not output_outliers_url:
            output_outliers_url = None

        self.status = status
        self.output_clusters_url = output_clusters_url
        self.output_outliers_url = output_outliers_url