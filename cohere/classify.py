from typing import Dict, List, NamedTuple

from cohere.response import CohereObject

Confidence = NamedTuple("Confidence", [("label", str), ("confidence", float)])
LabelPrediction = NamedTuple("LabelPrediction", [("confidence", Confidence)])
Example = NamedTuple("Example", [("text", str), ("label", str)])


class Classification(CohereObject):

    def __init__(self, input: str, prediction: str, confidence: List[Confidence],
                 labels: Dict[str, LabelPrediction]) -> None:
        self.input = input
        self.prediction = prediction
        self.confidence = confidence
        self.labels = labels

    def __repr__(self) -> str:
        return f"Classification<prediction: \"{self.prediction}\", confidence: {self.confidence}>"


class Classifications(CohereObject):

    def __init__(self, classifications: List[Classification]) -> None:
        self.classifications = classifications
        self.iterator = None

    def __repr__(self) -> str:
        return self.classifications.__repr__()

    def __str__(self) -> str:
        return self.classifications.__str__()

    def __iter__(self) -> iter:
        if self.iterator is None:
            self.iterator = iter(self.classifications)
        return self.iterator

    def __next__(self) -> next:
        return next(self.iterator)

    def __len__(self) -> int:
        return len(self.classifications)

    def __getitem__(self, key) -> Classification:
        return self.classifications[key]
