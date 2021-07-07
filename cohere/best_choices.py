from typing import List

class BestChoices:
    def __init__(self, likelihoods: List[float], mode: str) -> None:
        self.likelihoods = likelihoods
        self.mode = mode
        self.iterator = likelihoods
    
    def __str__(self) -> str:
        return str(self.likelihoods)

    def __iter__(self) -> iter:
        return self.iterator

    def __next__(self) -> next:
        return next(self.iterator)
