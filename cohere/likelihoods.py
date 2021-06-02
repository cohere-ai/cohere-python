from typing import List, Dict

class Likelihoods:
    def __init__(self, likelihood: float, token_likelihoods: List[Dict]) -> None:
        self.likelihood = likelihood
        self.token_likelihoods = token_likelihoods

    def __str__(self) -> str:
        return str(self.likelihood) + "\n" + str(self.token_likelihoods)
