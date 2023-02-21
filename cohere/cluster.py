from typing import Any, List, Optional, Dict

from cohere.response import CohereObject


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


class ClusterJobResult(CohereObject):
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


class CreateClusterJobResponse(CohereObject):
    job_id: str

    def __init__(self, job_id: str, wait_fn):
        self.job_id = job_id
        self._wait_fn = wait_fn

    def wait(
        self,
        timeout: Optional[float] = None,
        interval: float = 10,
    ) -> ClusterJobResult:
        """Wait for clustering job result.

        Args:
            timeout (Optional[float], optional): Wait timeout in seconds, if None - there is no limit to the wait time.
                Defaults to None.
            interval (float, optional): Wait poll interval in seconds. Defaults to 10.

        Raises:
            TimeoutError: wait timed out

        Returns:
            ClusterJobResult: Clustering job result.
        """

        return self._wait_fn(job_id=self.job_id, timeout=timeout, interval=interval)


def build_cluster_job_response(response: Dict[str, Any]) -> ClusterJobResult:
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

    return ClusterJobResult(
        job_id=response['job_id'],
        status=response['status'],
        output_clusters_url=response['output_clusters_url'],
        output_outliers_url=response['output_outliers_url'],
        clusters=clusters,
    )
