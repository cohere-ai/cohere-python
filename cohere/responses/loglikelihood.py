from typing import Dict, List, NamedTuple

TokenLogLikelihood = NamedTuple("TokenLogLikelihood", [("encoded", int), ("decoded", str), ("log_likelihood", float)])


class LogLikelihoods:
    @staticmethod
    def token_list_from_dict(token_list: List[Dict]):
        return [TokenLogLikelihood(**token) for token in token_list]

    def __init__(self, prompt_tokens: List[TokenLogLikelihood], completion_tokens: List[TokenLogLikelihood]):
        self.prompt_tokens = self.token_list_from_dict(prompt_tokens)
        self.completion_tokens = self.token_list_from_dict(completion_tokens)
