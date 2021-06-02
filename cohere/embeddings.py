from typing import List

class Embeddings:
    def __init__(self, embeddings: List[List[float]]) -> None:
        self.embeddings = embeddings

    def __str__(self) -> str:
        return str(self.embeddings)
