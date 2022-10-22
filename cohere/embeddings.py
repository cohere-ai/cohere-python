from typing import Iterator, List

from cohere.response import CohereObject


class Embedding(CohereObject):

    def __init__(self, embedding: List[float]) -> None:
        self.embedding = embedding

    def __iter__(self) -> Iterator:
        return iter(self.embedding)

    def __len__(self) -> int:
        return len(self.embedding)


class Embeddings(CohereObject):

    def __init__(self, embeddings: List[Embedding]) -> None:
        self.embeddings = embeddings

    def __iter__(self) -> Iterator:
        return iter(self.embeddings)

    def __len__(self) -> int:
        return len(self.embeddings)
