import csv
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import requests
from fastavro import parse_schema, reader, writer

from cohere.error import CohereError
from cohere.responses.base import CohereObject
from cohere.utils import JobWithStatus, parse_datetime

supported_formats = ["jsonl", "csv", "avro"]


class BaseDataset(CohereObject, JobWithStatus):
    id: str
    name: str
    dataset_type: str
    validation_status: str
    validation_error: Optional[str]
    validation_warnings: List[str]
    created_at: datetime
    updated_at: datetime
    download_urls: List[str]
    size_bytes: int
    schema: str
    _wait_fn: Callable[[], "Dataset"]

    def __init__(
        self,
        id: str,
        name: str,
        dataset_type: str,
        validation_status: str,
        created_at: str,
        updated_at: str,
        validation_warnings: List[str],
        validation_error: str = None,
        download_urls: List[str] = None,
        schema: str = None,
        wait_fn=None,
    ) -> None:
        self.id = id
        self.name = name
        self.dataset_type = dataset_type
        self.validation_status = validation_status
        self.created_at = parse_datetime(created_at)
        self.updated_at = parse_datetime(updated_at)
        self.schema = schema
        self.download_urls = download_urls
        self._wait_fn = wait_fn
        self.validation_error = validation_error
        self.validation_warnings = validation_warnings

    def __iter__(self):
        return self.open()

    @classmethod
    def from_dict(cls, data: Dict[str, Any], wait_fn) -> "Dataset":
        download_urls = []
        if data["validation_status"] == "validated":
            download_urls = [part.get("url") for part in data["dataset_parts"] if part.get("url")]

        return cls(
            id=data["id"],
            name=data["name"],
            dataset_type=data["dataset_type"],
            validation_status=data["validation_status"],
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            schema=data.get("schema"),
            download_urls=download_urls,
            wait_fn=wait_fn,
            validation_error=data.get("validation_error"),
            validation_warnings=data.get("validation_warnings", []),
        )

    def has_terminal_status(self) -> bool:
        return self.validation_status in ["validated", "failed"]

    def await_validation(self, timeout: Optional[float] = None, interval: float = 10) -> "Dataset":
        return self.wait(timeout, interval)

    def open(self):
        if self.validation_status != "validated":
            raise CohereError(message="cannot open non-validated dataset")
        for url in self.download_urls:
            resp = requests.get(url, stream=True)
            for record in reader(resp.raw):
                yield record

    def save(self, filepath: str, format: str = "jsonl"):
        if format == "jsonl":
            return self.save_jsonl(filepath)
        if format == "csv":
            return self.save_csv(filepath)
        if format == "avro":
            return self.save_avro(filepath)
        raise CohereError(message=f"unsupported format must be one of : {supported_formats}")

    def save_avro(self, filepath: str):
        schema = parse_schema(json.loads(self.schema))
        with open(filepath, "wb") as outfile:
            writer(outfile, schema, self.open())

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
        updated_job = self._wait_fn(dataset_id=self.id, timeout=timeout, interval=interval)
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
        updated_job = await self._wait_fn(dataset_id=self.id, timeout=timeout, interval=interval)
        self._update_self(updated_job)
        return updated_job


class DatasetUsage(CohereObject):
    user_usage: int
    organization_usage: int

    def __init__(self, user_usage: int, organization_usage: int) -> None:
        self.user_usage = user_usage
        self.organization_usage = organization_usage

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetUsage":
        return cls(user_usage=data.get("user_usage"), organization_usage=data.get("organization_usage"))


@dataclass
class ParseInfo:
    separator: Optional[str] = None
    delimiter: Optional[str] = None

    def get_params(self) -> Dict[str, str]:
        params = {}
        if self.separator:
            params["text_separator"] = self.separator
        if self.delimiter:
            params["csv_delimiter"] = self.delimiter
        return params
