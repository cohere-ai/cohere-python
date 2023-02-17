from typing import Any, List, Optional

from cohere.response import CohereObject


class CreateClusterJobResponse(CohereObject):
    job_id: str

    def __init__(self, job_id: str):
        self.job_id = job_id


class Cluster(CohereObject):
    id: str
    keywords: List[str]
    description: str
    size: int
    sample_elements: List[str]

    def __init__(
        self,
        id: str,
        keywords: List[str],
        description: str,
        size: int,
        sample_elements: List[str],
    ) -> None:
        self.id = id
        self.keywords = keywords
        self.description = description
        self.size = size
        self.sample_elements = sample_elements


class GetClusterJobResponse(CohereObject):
    job_id: str
    status: str
    output_clusters_url: Optional[str]
    output_outliers_url: Optional[str]
    clusters: Optional[List[Cluster]]

    def __init__(
        self,
        job_id: str,
        status: str,
        output_clusters_url: Optional[str],
        output_outliers_url: Optional[str],
        clusters: Optional[List[Cluster]],
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


def build_cluster_job_response(response: Any) -> GetClusterJobResponse:
    if response.get('clusters') is None:
        clusters = None
    else:
        clusters = [
            Cluster(
                id=c['id'],
                keywords=c['keywords'],
                description=c['description'],
                size=c['size'],
                sample_elements=c['sample_elements'],
            ) for c in response['clusters']
        ]

    return GetClusterJobResponse(
        job_id=response['job_id'],
        status=response['status'],
        output_clusters_url=response['output_clusters_url'],
        output_outliers_url=response['output_outliers_url'],
        clusters=clusters,
    )
