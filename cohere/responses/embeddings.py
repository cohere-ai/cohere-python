from typing import Iterator, List, Optional

from cohere.responses.base import CohereObject
from cohere.responses.meta_response import Meta


class Embedding(CohereObject):
    def __init__(self, embedding: List[float]) -> None:
        self.embedding = embedding

    def __iter__(self) -> Iterator:
        return iter(self.embedding)

    def __len__(self) -> int:
        return len(self.embedding)


class Embeddings(CohereObject):
    def __init__(self, embeddings: List[Embedding], meta: Optional[Meta] = None) -> None:
        self.embeddings = embeddings
        self.meta = meta

    def __iter__(self) -> Iterator:
        return iter(self.embeddings)

    def __len__(self) -> int:
        return len(self.embeddings)
