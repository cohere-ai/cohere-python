from typing import Optional

from cohere.responses.base import CohereObject


class ClusterJobResult(CohereObject):
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
