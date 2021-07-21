from typing import List

class BestChoices:
    def __init__(self, scores: List[float], tokens: List[List[str]], token_log_likelihoods: List[List[float]], mode: str) -> None:
        self.scores = scores
        self.tokens = tokens
        self.token_log_likelihoods = token_log_likelihoods
        self.mode = mode
        self.iterator = iter(scores)
    
    def __str__(self) -> str:
        contents = ""
        contents += f"\tscores: {self.scores}\n"
        contents += f"\ttokens: {self.tokens}\n"
        contents += f"\ttoken_log_likelihoods: {self.token_log_likelihoods}\n"

        output = f"cohere.BestChoices {{\n{contents}}}"

        return output

    def __iter__(self) -> iter:
        return self.iterator

    def __next__(self) -> next:
        return next(self.iterator)
