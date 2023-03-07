from typing import Dict, List, NamedTuple, Optional

from cohere.responses.base import CohereObject
from cohere.responses.meta_response import Meta

LabelPrediction = NamedTuple("LabelPrediction", [("confidence", float)])
Example = NamedTuple("Example", [("text", str), ("label", str)])


class Classification(CohereObject):

    def __init__(self, input: str, prediction: str, confidence: float, labels: Dict[str, LabelPrediction], meta: Optional[List[Meta]] = None, *args,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.input = input
        self.prediction = prediction
        self.confidence = confidence
        self.labels = labels
        self.meta = meta

    def __repr__(self) -> str:
        prediction = self.prediction
        confidence = self.confidence
        labels = self.labels
        meta = self.meta
        return f"Classification<prediction: \"{prediction}\", confidence: {confidence}, labels: {labels}, meta: {meta}>"


class Classifications(CohereObject):

    def __init__(self, classifications: List[Classification]) -> None:
        self.classifications = classifications

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
