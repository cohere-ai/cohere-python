from typing import Iterator, List

from cohere.response import CohereObject


class Embeddings(CohereObject):

    def __init__(self, embeddings: List[List[float]]) -> None:
        self.embeddings = embeddings

    def __iter__(self) -> Iterator:
        return iter(self.embeddings)
