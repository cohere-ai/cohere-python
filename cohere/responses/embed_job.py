from typing import Any, Awaitable, Callable, Dict, List, Optional, Union

from cohere.responses.base import CohereObject
from cohere.responses.dataset import Dataset
from cohere.utils import JobWithStatus


class BaseEmbedJob(CohereObject, JobWithStatus):
    job_id: str
    status: str
    created_at: str
    input_dataset_id: Optional[str]
    output: Dataset
    model: str
    truncate: str
    percent_complete: float
    _wait_fn: Union[Callable[[], "EmbedJob"], Callable[[], Awaitable["EmbedJob"]]]

    def __init__(
        self,
        job_id: str,
        status: str,
        created_at: str,
        input_dataset_id: Optional[str],
        model: str,
        input_type: str,
        embedding_types: List[str],
        truncate: str,
        percent_complete: float,
        wait_fn,
    ) -> None:
        self.job_id = job_id
        self.status = status
        self.created_at = created_at
        self.input_dataset_id = input_dataset_id
        self.model = model
        self.input_type = input_type
        self.embedding_types = embedding_types
        self.truncate = truncate
        self.percent_complete = percent_complete
        self._wait_fn = wait_fn

    @classmethod
    def from_dict(cls, data: Dict[str, Any], wait_fn) -> "BaseEmbedJob":
        return cls(
            job_id=data["job_id"],
            status=data["status"],
            created_at=data["created_at"],
            input_dataset_id=data.get("input_dataset_id"),
            model=data["model"],
            input_type=data.get("input_type"),
            embedding_types=data.get("embedding_types"),
            truncate=data["truncate"],
            percent_complete=data["percent_complete"],
            wait_fn=wait_fn,
        )

    def has_terminal_status(self) -> bool:
        return self.status in ["complete", "failed", "cancelled"]


class EmbedJob(BaseEmbedJob):
    def wait(
        self,
        timeout: Optional[float] = None,
        interval: float = 10,
    ) -> "EmbedJob":
        """Wait for embed job completion and updates all fields.

        Args:
            timeout (Optional[float], optional): Wait timeout in seconds, if None - there is no limit to the wait time.
                Defaults to None.
            interval (float, optional): Wait poll interval in seconds. Defaults to 10.

        Raises:
            TimeoutError: wait timed out

        Returns:
            EmbedJob: the updated job
        """
        updated_job = self._wait_fn(job_id=self.job_id, timeout=timeout, interval=interval)
        self._update_self(updated_job)
        return updated_job


class AsyncEmbedJob(BaseEmbedJob):
    async def wait(
        self,
        timeout: Optional[float] = None,
        interval: float = 10,
    ) -> "EmbedJob":
        """Wait for embed job completion and updates all fields.

        Args:
            timeout (Optional[float], optional): Wait timeout in seconds, if None - there is no limit to the wait time.
                Defaults to None.
            interval (float, optional): Wait poll interval in seconds. Defaults to 10.

        Raises:
            TimeoutError: wait timed out

        Returns:
            EmbedJob: the updated job
        """
        updated_job = await self._wait_fn(job_id=self.job_id, timeout=timeout, interval=interval)
        self._update_self(updated_job)
        return updated_job
