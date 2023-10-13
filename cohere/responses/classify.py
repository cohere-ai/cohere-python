from typing import Any, Dict, List, NamedTuple, Optional

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
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.input = input
        self.prediction = prediction  # to be deprecated
        self.confidence = confidence  # to be deprecated
        self.predictions = predictions
        self.confidences = confidences
        self.labels = labels

    def __repr__(self) -> str:
        if self.prediction is not None:
            assert self.confidence is not None
            return (
                f'Classification<prediction: "{self.prediction}", confidence: {self.confidence}, labels: {self.labels}>'
            )
        else:
            assert self.predictions is not None
            assert self.confidences is not None
            return f'Classification<predictions: "{self.predictions}", confidences: {self.confidences}, labels: {self.labels}>'


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
