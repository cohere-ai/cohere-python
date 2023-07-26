import csv
import json
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import requests
from fastavro import reader

from cohere.error import CohereError
from cohere.responses.base import CohereObject
from cohere.utils import JobWithStatus, parse_datetime

supported_formats = ["jsonl", "csv"]


class BaseDataset(CohereObject, JobWithStatus):
    id: str
    name: str
    dataset_type: str
    validation_status: str
    validation_error: Optional[str]
    created_at: datetime
    updated_at: datetime
    urls: List[str]
    size_bytes: int
    _wait_fn: Callable[[], "Dataset"]

    def __init__(
        self,
        id: str,
        name: str,
        dataset_type: str,
        validation_status: str,
        created_at: str,
        updated_at: str,
        validation_error: str = None,
        urls: List[str] = None,
        wait_fn=None,
    ) -> None:
        self.id = id
        self.name = name
        self.dataset_type = dataset_type
        self.validation_status = validation_status
        self.created_at = parse_datetime(created_at)
        self.updated_at = parse_datetime(updated_at)
        self.urls = urls
        self._wait_fn = wait_fn
        self.validation_error = validation_error

    @classmethod
    def from_dict(cls, data: Dict[str, Any], wait_fn) -> "Dataset":
        urls = []
        if data["validation_status"] == "validated":
            urls = [part.get("url") for part in data["dataset_parts"] if part.get("url")]

        return cls(
            id=data["id"],
            name=data["name"],
            dataset_type=data["dataset_type"],
            validation_status=data["validation_status"],
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            urls=urls,
            wait_fn=wait_fn,
            validation_error=data.get("validation_error"),
        )

    def has_terminal_status(self) -> bool:
        return self.validation_status in ["validated", "failed"]

    def await_validation(self, timeout: Optional[float] = None, interval: float = 10) -> "Dataset":
        return self.wait(timeout, interval)

    def open(self):
        if self.validation_status != "validated":
            raise CohereError(message="cannot open non-validated dataset")
        for url in self.urls:
            resp = requests.get(url, stream=True)
            for record in reader(resp.raw):
                yield record

    def save(self, filepath: str, format: str = "jsonl"):
        if format == "jsonl":
            return self.save_jsonl(filepath)
        if format == "csv":
            return self.save_csv(filepath)
        raise CohereError(message=f"unsupported format must be one of : {supported_formats}")

    def save_jsonl(self, filepath: str):
        with open(filepath, "w") as outfile:
            for data in self.open():
                json.dump(data, outfile)
                outfile.write("\n")

    def save_csv(self, filepath: str):
        with open(filepath, "w") as outfile:
            for i, data in enumerate(self.open()):
                if i == 0:
                    writer = csv.DictWriter(outfile, fieldnames=list(data.keys()))
                    writer.writeheader()
                writer.writerow(data)


class Dataset(BaseDataset):
    def wait(
        self,
        timeout: Optional[float] = None,
        interval: float = 10,
    ) -> "Dataset":
        """Wait for dataset validation and updates attributes once finished.

        Args:
            timeout (Optional[float], optional): Wait timeout in seconds, if None - there is no limit to the wait time.
                Defaults to None.
            interval (float, optional): Wait poll interval in seconds. Defaults to 10.

        Raises:
            TimeoutError: wait timed out
        """
        updated_job = self._wait_fn(job_id=self.id, timeout=timeout, interval=interval)
        self._update_self(updated_job)
        return updated_job


class AsyncDataset(BaseDataset):
    async def wait(
        self,
        timeout: Optional[float] = None,
        interval: float = 10,
    ) -> "Dataset":
        """Wait for dataset validation and updates attributes once finished.

        Args:
            timeout (Optional[float], optional): Wait timeout in seconds, if None - there is no limit to the wait time.
                Defaults to None.
            interval (float, optional): Wait poll interval in seconds. Defaults to 10.

        Raises:
            TimeoutError: wait timed out
        """
        updated_job = await self._wait_fn(job_id=self.id, timeout=timeout, interval=interval)
        self._update_self(updated_job)
        return updated_job
