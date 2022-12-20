from typing import Dict, List, NamedTuple

from cohere.response import CohereObject

LabelPrediction = NamedTuple("LabelPrediction", [("confidence", float)])
Example = NamedTuple("Example", [("text", str), ("label", str)])


class Classification(CohereObject):

    def __init__(self, input: str, prediction: str, confidence: float, labels: Dict[str, LabelPrediction]) -> None:
        self.input = input
        self.prediction = prediction
        self.confidence = confidence
        self.labels = labels

    def __repr__(self) -> str:
        prediction = self.prediction
        confidence = self.confidence
        labels = self.labels
        return f"Classification<prediction: \"{prediction}\", confidence: {confidence}, labels: {labels}>"


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
