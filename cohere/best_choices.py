from typing import List

class BestChoices:
    def __init__(self, likelihoods: List[float], tokens: List[List[str]], token_log_likelihoods: List[List[float]], mode: str) -> None:
        self.likelihoods = likelihoods
        self.tokens = tokens
        self.token_log_likelihoods = token_log_likelihoods
        self.mode = mode
        self.iterator = iter(likelihoods)
    
    def __str__(self) -> str:
        return str(self.likelihoods)

    def __iter__(self) -> iter:
        return self.iterator

    def __next__(self) -> next:
        return next(self.iterator)
