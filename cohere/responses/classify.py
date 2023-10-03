from typing import Any, Dict, List, NamedTuple, Optional, Union

from cohere.responses.base import CohereObject

LabelPrediction = NamedTuple("LabelPrediction", [("confidence", float)])
Example = NamedTuple("Example", [("text", str), ("label", str)])


class Classification(CohereObject):
    def __init__(
        self, input: str, prediction: Union[str, List[str]], confidence: Union[float, List[float]], labels: Dict[str, LabelPrediction], *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.input = input
        self.prediction = prediction
        self.confidence = confidence
        self.labels = labels

    def __repr__(self) -> str:
        prediction = self.prediction
        confidence = self.confidence
        labels = self.labels
        prediction_str = f'"{prediction}"' if isinstance(prediction, str) else prediction
        return f'Classification<prediction: {prediction_str}, confidence: {confidence}, labels: {labels}>'


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
