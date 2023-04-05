from pprint import pprint
from typing import Any, Dict, List, Optional

from cohere.responses.base import CohereObject


class CreateBulkEmbedJobResponse(CohereObject):
    job_id: str
    meta: Optional[Dict[str, Any]]

    def __init__(self, job_id: str, meta: Optional[Dict[str, Any]]):
        self.job_id = job_id
        self.meta = meta

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CreateBulkEmbedJobResponse":
        return cls(
            job_id=data["job_id"],
            meta=data.get("meta"),
        )


class BulkEmbedJob(CohereObject):
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
