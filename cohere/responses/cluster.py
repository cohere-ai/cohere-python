from typing import Any, Dict, List, Optional

from cohere.responses.base import CohereObject
from cohere.utils import JobWithStatus


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


class BaseClusterJobResult(CohereObject, JobWithStatus):
    job_id: str
    status: str
    is_final_state: bool
    output_clusters_url: Optional[str]
    output_outliers_url: Optional[str]
    clusters: Optional[List[Cluster]]
    error: Optional[str]
    meta: Optional[Dict[str, Any]]

    def __init__(
        self,
        job_id: str,
        status: str,
        output_clusters_url: Optional[str],
        output_outliers_url: Optional[str],
        clusters: Optional[List[Cluster]],
        error: Optional[str],
        is_final_state: bool,
        meta: Optional[Dict[str, Any]] = None,
        wait_fn=None,
    ):
        # convert empty string to `None`
        if not output_clusters_url:
            output_clusters_url = None
        if not output_outliers_url:
            output_outliers_url = None

        self.job_id = job_id
        self.status = status
        self.is_final_state = is_final_state
        self.output_clusters_url = output_clusters_url
        self.output_outliers_url = output_outliers_url
        self.clusters = clusters
        self.meta = meta
        self.error = error
        self._wait_fn = wait_fn

    @classmethod
    def from_dict(cls, data: Dict[str, Any], wait_fn) -> "ClusterJobResult":
        if data.get("clusters") is None:
            clusters = None
        else:
            clusters = [Cluster.from_dict(c) for c in data["clusters"]]

        is_final_state = data.get("is_final_state")
        status = data["status"]
        # TODO: remove this. temp for backward compatibility until the `is_final_state` field is added to the API
        if is_final_state is None:
            is_final_state = status in ["complete", "failed"]

        return cls(
            job_id=data["job_id"],
            status=status,
            is_final_state=is_final_state,
            output_clusters_url=data["output_clusters_url"],
            output_outliers_url=data["output_outliers_url"],
            clusters=clusters,
            error=data.get("error"),
            meta=data.get("meta"),
            wait_fn=wait_fn,
        )

    def has_terminal_status(self) -> bool:
        return self.is_final_state


class ClusterJobResult(BaseClusterJobResult):
    def wait(
        self,
        timeout: Optional[float] = None,
        interval: float = 10,
    ) -> "ClusterJobResult":
        """Wait for cluster job completion and updates attributes once finished.

        Args:
            timeout (Optional[float], optional): Wait timeout in seconds, if None - there is no limit to the wait time.
                Defaults to None.
            interval (float, optional): Wait poll interval in seconds. Defaults to 10.

        Raises:
            TimeoutError: wait timed out
        """
        updated_job = self._wait_fn(job_id=self.job_id, timeout=timeout, interval=interval)
        self._update_self(updated_job)
        return updated_job


class AsyncClusterJobResult(BaseClusterJobResult):
    async def wait(
        self,
        timeout: Optional[float] = None,
        interval: float = 10,
    ) -> "ClusterJobResult":
        """Wait for cluster job completion and updates attributes once finished.

        Args:
            timeout (Optional[float], optional): Wait timeout in seconds, if None - there is no limit to the wait time.
                Defaults to None.
            interval (float, optional): Wait poll interval in seconds. Defaults to 10.

        Raises:
            TimeoutError: wait timed out
        """
        updated_job = await self._wait_fn(job_id=self.job_id, timeout=timeout, interval=interval)
        self._update_self(updated_job)
        return updated_job
