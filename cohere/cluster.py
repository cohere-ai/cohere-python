from typing import Optional

from cohere.response import CohereObject


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


class CreateClusterJobResponse(CohereObject):
    job_id: str

    def __init__(self, job_id: str, poll_fn):
        self.job_id = job_id
        self._poll_fn = poll_fn

    def poll(
        self,
        timeout: Optional[float] = None,
        interval: float = 10,
    ) -> GetClusterJobResponse:
        """Poll clustering job results.

        Args:
            timeout (Optional[float], optional): Poll timeout in seconds, if None - there is no limit to the wait time. 
                Defaults to None.
            interval (float, optional): Poll interval in seconds. Defaults to 10.

        Raises:
            TimeoutError: poll timed out

        Returns:
            GetClusterJobResponse: Clustering job results.
        """

        return self._poll_fn(job_id=self.job_id, timeout=timeout, interval=interval)
