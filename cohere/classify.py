from cohere.response import CohereObject
from typing import List, Dict


class Confidence(CohereObject):
    def __init__(self, label: str, confidence: float) -> None:
        self.label = label
        self.confidence = confidence


class LabelPrediction(CohereObject):
    def __init__(self, confidence: float) -> None:
        self.confidence = confidence


class Classification(CohereObject):
    def __init__(self, input: str,
                 prediction: str, confidence: List[Confidence], labels: Dict[str, LabelPrediction]) -> None:
        self.input = input
        self.prediction = prediction
        self.confidence = confidence
        self.labels = labels


class Classifications(CohereObject):
    def __init__(self, classifications: List[Classification]) -> None:
        self.classifications = classifications
        self.iterator = iter(classifications)

    def __iter__(self) -> iter:
        return self.iterator

    def __next__(self) -> next:
        return next(self.iterator)

    def __len__(self) -> int:
        return len(self.classifications)


class Example(CohereObject):
    def __init__(self, text: str, label: str) -> None:
        self.text = text
        self.label = label
