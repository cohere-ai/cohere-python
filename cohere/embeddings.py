from cohere.response import CohereResponse
from typing import List

class Embeddings(CohereResponse):
    def __init__(self, embeddings: List[List[float]]) -> None:
        self.embeddings = embeddings
        self.iterator = iter(embeddings)
    
    def __iter__(self) -> iter:
        return self.iterator

    def __next__(self) -> next:
        return next(self.iterator)
