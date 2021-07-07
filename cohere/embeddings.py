from typing import List

class Embeddings:
    def __init__(self, embeddings: List[List[float]]) -> None:
        self.embeddings = embeddings
        self.iterator = iter(embeddings) 

    def __str__(self) -> str:
        return str(self.embeddings)
    
    def __iter__(self) -> iter:
        return self.iterator

    def __next__(self) -> next:
        return next(self.iterator)
