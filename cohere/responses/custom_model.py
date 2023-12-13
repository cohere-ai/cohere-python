from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

from cohere.utils import JobWithStatus

try:
    from typing import Literal, TypedDict
except ImportError:
    from typing_extensions import Literal, TypedDict

from cohere.responses.base import CohereObject

CUSTOM_MODEL_STATUS = Literal[
    "UNKNOWN",
    "CREATED",
    "TRAINING",
    "DEPLOYING",
    "READY",
    "FAILED",
    "DELETED",
    "TEMPORARILY_OFFLINE",
    "PAUSED",
    "QUEUED",
]
INTERNAL_CUSTOM_MODEL_STATUS = Literal[
    "UNKNOWN",
    "CREATED",
    "FINETUNING",
    "DEPLOYING_API",
    "READY",
    "FAILED",
    "DELETED",
    "TEMPORARILY_OFFLINE",
    "PAUSED",
    "QUEUED",
]
CUSTOM_MODEL_INTERNAL_STATUS_MAPPING: Dict[CUSTOM_MODEL_STATUS, INTERNAL_CUSTOM_MODEL_STATUS] = {
    "UNKNOWN": "UNKNOWN",
    "CREATED": "CREATED",
    "TRAINING": "FINETUNING",
    "DEPLOYING": "DEPLOYING_API",
    "READY": "READY",
    "FAILED": "FAILED",
    "DELETED": "DELETED",
    "TEMPORARILY_OFFLINE": "TEMPORARILY_OFFLINE",
    "PAUSED": "PAUSED",
    "QUEUED": "QUEUED",
}
REVERSE_CUSTOM_MODEL_INTERNAL_STATUS_MAPPING = {v: k for k, v in CUSTOM_MODEL_INTERNAL_STATUS_MAPPING.items()}

INTERNAL_CUSTOM_MODEL_TYPE = Literal["GENERATIVE", "CLASSIFICATION", "RERANK", "CHAT"]
CUSTOM_MODEL_TYPE = Literal["GENERATIVE", "CLASSIFY", "RERANK", "CHAT"]
CUSTOM_MODEL_PRODUCT_MAPPING: Dict[CUSTOM_MODEL_TYPE, INTERNAL_CUSTOM_MODEL_TYPE] = {
    "GENERATIVE": "GENERATIVE",
    "CLASSIFY": "CLASSIFICATION",
    "RERANK": "RERANK",
    "CHAT": "CHAT",
}
REVERSE_CUSTOM_MODEL_PRODUCT_MAPPING: Dict[INTERNAL_CUSTOM_MODEL_TYPE, CUSTOM_MODEL_TYPE] = {
    v: k for k, v in CUSTOM_MODEL_PRODUCT_MAPPING.items()
}


@dataclass
class HyperParameters:
    early_stopping_patience: int
    early_stopping_threshold: float
    train_batch_size: int
    train_steps: int
    train_epochs: int
    learning_rate: float

    @staticmethod
    def from_response(response: Optional[dict]) -> "HyperParameters":
        return HyperParameters(
            early_stopping_patience=response.get("earlyStoppingPatience"),
            early_stopping_threshold=response.get("earlyStoppingThreshold"),
            train_batch_size=response.get("trainBatchSize"),
            train_steps=response.get("trainSteps"),
            train_epochs=response.get("trainEpochs"),
            learning_rate=response.get("learningRate"),
        )


class HyperParametersInput(TypedDict):
    """
    early_stopping_patience: int (default=6, min=0, max=10)
    early_stopping_threshold: float (default=0.01, min=0, max=0.1)
    train_batch_size: int (default=16, min=2, max=16)
    train_epochs: int (default=1, min=1, max=10)
    learning_rate: float (default=0.01, min=0.000005, max=0.1)
    """

    early_stopping_patience: int
    early_stopping_threshold: float
    train_batch_size: int
    train_epochs: int
    learning_rate: float


@dataclass
class FinetuneBilling:
    train_epochs: int
    num_training_tokens: int
    unit_price: float
    total_cost: float

    @staticmethod
    def from_response(response: Optional[dict]) -> "FinetuneBilling":
        return FinetuneBilling(
            train_epochs=response.get("epochs"),
            num_training_tokens=response.get("numTrainingTokens"),
            unit_price=response.get("unitPrice"),
            total_cost=response.get("totalCost"),
        )


class BaseCustomModel(CohereObject, JobWithStatus):
    def __init__(
        self,
        wait_fn,
        id: str,
        name: str,
        status: CUSTOM_MODEL_STATUS,
        model_type: CUSTOM_MODEL_TYPE,
        created_at: datetime,
        completed_at: Optional[datetime],
        base_model: Optional[str] = None,
        model_id: Optional[str] = None,
        hyperparameters: Optional[HyperParameters] = None,
        dataset_id: Optional[str] = None,
        billing: Optional[FinetuneBilling] = None,
    ) -> None:
        super().__init__()
        self.id = id
        self.name = name
        self.status = status
        self.model_type = model_type
        self.created_at = created_at
        self.completed_at = completed_at
        self.base_model = base_model
        self.model_id = model_id
        self.hyperparameters = hyperparameters
        self.dataset_id = dataset_id
        self._wait_fn = wait_fn
        self.billing = billing

    @classmethod
    def from_dict(cls, data: Dict[str, Any], wait_fn) -> "BaseCustomModel":
        return cls(
            wait_fn=wait_fn,
            id=data["id"],
            name=data["name"],
            status=REVERSE_CUSTOM_MODEL_INTERNAL_STATUS_MAPPING[data["status"]],
            model_type=REVERSE_CUSTOM_MODEL_PRODUCT_MAPPING[data["settings"]["finetuneType"]],
            created_at=_parse_date(data["created_at"]),
            completed_at=_parse_date(data["completed_at"]) if "completed_at" in data else None,
            base_model=data["settings"]["baseModel"],
            model_id=data["model"]["route"] if "model" in data else None,
            hyperparameters=HyperParameters.from_response(data["settings"]["hyperparameters"])
            if data.get("settings").get("hyperparameters") is not None
            else None,
            dataset_id=data["settings"].get("datasetID"),
            billing=FinetuneBilling.from_response(data.get("billing")) if data.get("billing") is not None else None,
        )

    def has_terminal_status(self) -> bool:
        return self.status == "READY"


class CustomModel(BaseCustomModel):
    def wait(
        self,
        timeout: Optional[float] = None,
        interval: float = 60,
    ) -> "CustomModel":
        """Wait for custom model job completion.

        Args:
            timeout (Optional[float], optional): Wait timeout in seconds, if None - there is no limit to the wait time.
                Defaults to None.
            interval (float, optional): Wait poll interval in seconds. Defaults to 60.

        Raises:
            TimeoutError: wait timed out

        Returns:
            CustomModel: custom model.
        """

        return self._wait_fn(custom_model_id=self.id, timeout=timeout, interval=interval)


class AsyncCustomModel(BaseCustomModel):
    async def wait(
        self,
        timeout: Optional[float] = None,
        interval: float = 60,
    ) -> "CustomModel":
        """Wait for custom model job completion.

        Args:
            timeout (Optional[float], optional): Wait timeout in seconds, if None - there is no limit to the wait time.
                Defaults to None.
            interval (float, optional): Wait poll interval in seconds. Defaults to 60.

        Raises:
            TimeoutError: wait timed out

        Returns:
            CustomModel: custom model.
        """

        return await self._wait_fn(custom_model_id=self.id, timeout=timeout, interval=interval)


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
