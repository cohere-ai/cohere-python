from cohere.response import CohereObject
from typing import List

class LabelProbability(CohereObject): 
    def __init__(self, label: str, probability: float) -> None:
        self.label = label
        self.probability = probability

class Classification(CohereObject):
    def __init__(self, text: str, prediction: str, labelProbability: LabelProbability) -> None:
        self.text = text
        self.prediction = prediction
        self.labelProbability = labelProbability

class Classifications(CohereObject):
    def __init__(self, classifications: List[Classification]) -> None:
        self.classifications = classifications
        self.iterator = iter(classifications)

    def __iter__(self) -> iter:
        return self.iterator

    def __next__(self) -> next:
        return next(self.iterator)

class ClassifyExample(CohereObject):
    def __init__(self, text: str, label: str) -> None:
        self.text = text
        self.label = label

