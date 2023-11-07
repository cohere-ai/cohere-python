from typing import Any, Dict, List, NamedTuple, Optional

from cohere.logging import logger
from cohere.responses.base import CohereObject

LabelPrediction = NamedTuple("LabelPrediction", [("confidence", float)])
Example = NamedTuple("Example", [("text", str), ("label", str)])


class Classification(CohereObject):
    def __init__(
        self,
        input: str,
        predictions: Optional[List[str]],
        confidences: Optional[List[float]],
        prediction: Optional[str],
        confidence: Optional[float],
        labels: Dict[str, LabelPrediction],
        classification_type: str,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.input = input
        self._prediction = prediction  # to be removed
        self._confidence = confidence  # to be removed
        self.predictions = predictions
        self.confidences = confidences
        self.labels = labels
        self.classification_type = classification_type

        if self._prediction is None or self._confidence is None:
            if self._prediction is not None or self._confidence is not None:
                raise ValueError("Cannot have one of `prediction` and `confidence` be None and not the other one")
            if self.predictions is None or self.confidences is None:
                raise ValueError("Cannot have `predictions` or `confidences` be None if `prediction` is None")

    def __repr__(self) -> str:
        if self._prediction is not None:
            return f'Classification<prediction: "{self._prediction}", confidence: {self._confidence}, labels: {self.labels}>'
        else:
            return f'Classification<predictions: "{self.predictions}", confidences: {self.confidences}, labels: {self.labels}>'

    @property
    def prediction(self):
        logger.warning("`prediction` is deprecated and will be removed soon. Please use `predictions` instead.")
        return self._prediction

    @property
    def confidence(self):
        logger.warning("`confidence` is deprecated and will be removed soon. Please use `confidences` instead.")
        return self._confidence

    def is_multilabel(self) -> bool:
        return self.classification_type == "multi-label"


class Classifications(CohereObject):
    def __init__(self, classifications: List[Classification], meta: Optional[Dict[str, Any]] = None) -> None:
        self.classifications = classifications
        self.meta = meta

    def __repr__(self) -> str:
        return self.classifications.__repr__()

    def __str__(self) -> str:
        return self.classifications.__str__()

    def __iter__(self) -> iter:
        return iter(self.classifications)

    def __len__(self) -> int:
        return len(self.classifications)

    def __getitem__(self, index) -> Classification:
        return self.classifications[index]
