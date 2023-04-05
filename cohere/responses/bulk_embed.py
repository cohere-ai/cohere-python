from typing import Any, Dict, List, Optional

from cohere.responses.base import CohereObject
from cohere.utils import JobWithStatus


class BulkEmbedJob(CohereObject, JobWithStatus):
    job_id: str
    status: str
    created_at: str
    input_url: str
    output_urls: Optional[List[str]]
    model: str
    truncate: str
    percent_complete: float
    meta: Optional[Dict[str, Any]]

    def __init__(
        self,
        job_id: str,
        status: str,
        created_at: str,
        input_url: str,
        output_urls: Optional[List[str]],
        model: str,
        truncate: str,
        percent_complete: float,
        meta: Optional[Dict[str, Any]],
    ) -> None:
        self.job_id = job_id
        self.status = status
        self.created_at = created_at
        self.input_url = input_url
        self.output_urls = output_urls
        self.model = model
        self.truncate = truncate
        self.percent_complete = percent_complete
        self.meta = meta

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BulkEmbedJob":
        return cls(
            job_id=data["job_id"],
            status=data["status"],
            created_at=data["created_at"],
            input_url=data["input_url"],
            output_urls=data.get("output_urls"),
            model=data["model"],
            truncate=data["truncate"],
            percent_complete=data["percent_complete"],
            meta=data.get("meta"),
        )

    def has_terminal_status(self) -> bool:
        return self.status in ["complete", "failed", "cancelled"]


class CreateBulkEmbedJobResponse(CohereObject):
    job_id: str
    meta: Optional[Dict[str, Any]]

    def __init__(self, job_id: str, meta: Optional[Dict[str, Any]], wait_fn):
        self.job_id = job_id
        self.meta = meta
        self._wait_fn = wait_fn

    @classmethod
    def from_dict(cls, data: Dict[str, Any], wait_fn) -> "CreateBulkEmbedJobResponse":
        return cls(
            job_id=data["job_id"],
            meta=data.get("meta"),
            wait_fn=wait_fn,
        )

    def wait(
        self,
        timeout: Optional[float] = None,
        interval: float = 10,
    ) -> BulkEmbedJob:
        """Wait for bulk embed job completion.

        Args:
            timeout (Optional[float], optional): Wait timeout in seconds, if None - there is no limit to the wait time.
                Defaults to None.
            interval (float, optional): Wait poll interval in seconds. Defaults to 10.

        Raises:
            TimeoutError: wait timed out

        Returns:
            BulkEmbedJob: Bulk embed job.
        """

        return self._wait_fn(job_id=self.job_id, timeout=timeout, interval=interval)


class AsyncCreateBulkEmbedJobResponse(CreateBulkEmbedJobResponse):
    async def wait(
        self,
        timeout: Optional[float] = None,
        interval: float = 10,
    ) -> BulkEmbedJob:
        """Wait for bulk embed job completion.

        Args:
            timeout (Optional[float], optional): Wait timeout in seconds, if None - there is no limit to the wait time.
                Defaults to None.
            interval (float, optional): Wait poll interval in seconds. Defaults to 10.

        Raises:
            TimeoutError: wait timed out

        Returns:
            BulkEmbedJob: Bulk embed job.
        """

        return await self._wait_fn(job_id=self.job_id, timeout=timeout, interval=interval)
