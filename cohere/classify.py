from cohere.response import CohereObject
from typing import List


class LabelPrediction(CohereObject):
    def __init__(self, confidence: float) -> None:
        self.confidence = confidence


class Classification(CohereObject):
    def __init__(self, input: str,
                 prediction: dict[str, float], labels: dict[str, LabelPrediction]) -> None:
        self.input = input
        self.prediction = prediction
        if len(prediction) == 1 :
            self.prediction_label = list(prediction.keys())[0]
            self.prediction_confidence = list(prediction.values())[0]
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
