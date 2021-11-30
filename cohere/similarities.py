from cohere.response import CohereObject
from typing import List

class Similarities(CohereObject):
    def __init__(self, similarities: List[float]) -> None:
        self.similarities = similarities
        self.iterator = iter(similarities)

    def __iter__(self) -> iter:
        return self.iterator

    def __next__(self) -> next:
        return next(self.iterator)
