from datetime import datetime
from typing import Any, Dict, Optional
from typing_extensions import Literal

from cohere.responses.base import CohereObject

FINETUNE_STATUS = Literal[
    "UNKNOWN",
    "CREATED",
    "DATA_PROCESSING",
    "FINETUNING",
    "EXPORTING_MODEL",
    "DEPLOYING_API",
    "READY",
    "FAILED",
    "DELETED",
    "DELETE_FAILED",
    "CANCELLED",
    "TEMPORARILY_OFFLINE",
    "PAUSED",
    "QUEUED",
]

INTERNAL_FINETUNE_TYPE = Literal[
    "GENERATIVE",
    "CONTRASTIVE",
    "CLASSIFICATION",
]
FINETUNE_TYPE = Literal["GENERATIVE", "EMBED", "CLASSIFY"]
FINETUNE_PRODUCT_MAPPING: dict[FINETUNE_TYPE, INTERNAL_FINETUNE_TYPE] = {
    "GENERATIVE": "GENERATIVE",
    "EMBED": "CONTRASTIVE",
    "CLASSIFY": "CLASSIFICATION",
}


class Finetune(CohereObject):
    def __init__(
        self, id: str, name: str, status: FINETUNE_STATUS, created_at: datetime, completed_at: Optional[datetime]
    ) -> None:
        super().__init__()
        self.id = id
        self.name = name
        self.status = status
        self.created_at = created_at
        self.completed_at = completed_at

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Finetune":
        return cls(
            id=data["id"],
            name=data["name"],
            status=data["status"],
            created_at=_parse_date(data["created_at"]),
            completed_at=_parse_date(data["completed_at"]) if "completed_at" in data else None,
        )


def _parse_date(datetime_string: str) -> datetime:
    return datetime.strptime(datetime_string, "%Y-%m-%dT%H:%M:%S.%f%z")
