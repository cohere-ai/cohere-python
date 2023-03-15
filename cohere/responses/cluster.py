from typing import Any, Dict, List, Optional

from cohere.responses.base import CohereObject
from cohere.responses.meta_response import Meta


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

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Cluster":
        return cls(
            id=data["id"],
            keywords=data["keywords"],
            description=data["description"],
            size=data["size"],
            sample_elements=data["sample_elements"],
        )


class ClusterJobResult(CohereObject):
    job_id: str
    status: str
    output_clusters_url: Optional[str]
    output_outliers_url: Optional[str]
    clusters: Optional[List[Cluster]]
    error: Optional[str]
    meta: Optional[Meta]

    def __init__(
        self,
        job_id: str,
        status: str,
        output_clusters_url: Optional[str],
        output_outliers_url: Optional[str],
        clusters: Optional[List[Cluster]],
        error: Optional[str],
        meta: Optional[Meta] = None,
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
        self.meta = meta
        self.error = error

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClusterJobResult":
        if data.get("clusters") is None:
            clusters = None
        else:
            clusters = [Cluster.from_dict(c) for c in data["clusters"]]
        return ClusterJobResult(
            job_id=data["job_id"],
            status=data["status"],
            output_clusters_url=data["output_clusters_url"],
            output_outliers_url=data["output_outliers_url"],
            clusters=clusters,
            error=data.get("error"),
            meta=data.get("meta"),
        )


class CreateClusterJobResponse(CohereObject):
    job_id: str
    meta: Optional[Meta]

    def __init__(self, job_id: str, meta: Optional[Meta], wait_fn):
        self.job_id = job_id
        self._wait_fn = wait_fn
        self.meta = meta

    @classmethod
    def from_dict(cls, data: Dict[str, Any], wait_fn) -> "CreateClusterJobResponse":
        return cls(job_id=data["job_id"], wait_fn=wait_fn, meta=data.get("meta"))

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


class AsyncCreateClusterJobResponse(CreateClusterJobResponse):
    async def wait(
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

        return await self._wait_fn(job_id=self.job_id, timeout=timeout, interval=interval)
