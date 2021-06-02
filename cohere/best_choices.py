from typing import List

class BestChoices:
    def __init__(self, likelihoods: List[float], mode: str) -> None:
        self.likelihoods = likelihoods
        self.mode = mode
    
    def __str__(self) -> str:
        return str(self.likelihoods)
