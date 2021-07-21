from typing import List, Dict

class Likelihoods:
    def __init__(self, likelihood: float, token_likelihoods: List[Dict]) -> None:
        self.likelihood = likelihood
        self.token_likelihoods = token_likelihoods

    def __str__(self) -> str:
        contents = ""
        contents += f"\tlikelihood: {self.likelihood}\n"
        contents += f"\ttoken_likelihoods: {self.token_likelihoods}\n"

        output = f"cohere.Likelihoods {{\n{contents}}}"

        return output
