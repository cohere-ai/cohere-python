from typing import List

class BestChoices:
    def __init__(self, scores: List[float], mode: str) -> None:
        self.scores = scores
        self.mode = mode
        self.iterator = iter(likelihoods)
    
    def __str__(self) -> str:
        return str(self.likelihoods)

    def __iter__(self) -> iter:
        return self.iterator

    def __next__(self) -> next:
        return next(self.iterator)
