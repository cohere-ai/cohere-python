from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from cohere.responses.base import CohereObject

CUSTOM_MODEL_STATUS = Literal[
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

INTERNAL_CUSTOM_MODEL_TYPE = Literal[
    "GENERATIVE",
    "CONTRASTIVE",
    "CLASSIFICATION",
]
CUSTOM_MODEL_TYPE = Literal["GENERATIVE", "EMBED", "CLASSIFY"]
CUSTOM_MODEL_PRODUCT_MAPPING: Dict[CUSTOM_MODEL_TYPE, INTERNAL_CUSTOM_MODEL_TYPE] = {
    "GENERATIVE": "GENERATIVE",
    "EMBED": "CONTRASTIVE",
    "CLASSIFY": "CLASSIFICATION",
}
REVERSE_CUSTOM_MODEL_PRODUCT_MAPPING: Dict[INTERNAL_CUSTOM_MODEL_TYPE, CUSTOM_MODEL_TYPE] = {
    v: k for k, v in CUSTOM_MODEL_PRODUCT_MAPPING.items()
}


class CustomModel(CohereObject):
    def __init__(
        self,
        id: str,
        name: str,
        status: CUSTOM_MODEL_STATUS,
        model_type: CUSTOM_MODEL_TYPE,
        created_at: datetime,
        completed_at: Optional[datetime],
        model_id: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.id = id
        self.name = name
        self.status = status
        self.model_type = model_type
        self.created_at = created_at
        self.completed_at = completed_at
        self.model_id = model_id

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CustomModel":
        return cls(
            id=data["id"],
            name=data["name"],
            status=data["status"],
            model_type=REVERSE_CUSTOM_MODEL_PRODUCT_MAPPING[data["settings"]["finetuneType"]],
            created_at=_parse_date(data["created_at"]),
            completed_at=_parse_date(data["completed_at"]) if "completed_at" in data else None,
            model_id=data["model"]["route"] if "model" in data else None,
        )


@dataclass
class ModelMetric(CohereObject):
    created_at: datetime
    step_num: int
    loss: Optional[float] = None
    accuracy: Optional[float] = None
    f1: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelMetric":
        return cls(
            created_at=_parse_date_with_variable_seconds(data["created_at"]),
            step_num=data["step_num"],
            loss=data.get("loss"),
            accuracy=data.get("accuracy"),
            f1=data.get("f1"),
            precision=data.get("precision"),
            recall=data.get("recall"),
        )


def _parse_date(datetime_string: str) -> datetime:
    return datetime.strptime(datetime_string, "%Y-%m-%dT%H:%M:%S.%f%z")


def _parse_date_with_variable_seconds(datetime_string: str) -> datetime:
    # model metrics timestamp sometimes contains nanoseconds, so we truncate
    dt_concat = datetime_string[:26] + datetime_string[-1:]
    return datetime.strptime(dt_concat, "%Y-%m-%dT%H:%M:%S.%f%z")
