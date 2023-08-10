from typing import Dict, List, NamedTuple, Optional

TokenLogLikelihood = NamedTuple("TokenLogLikelihood", [("encoded", int), ("decoded", str), ("log_likelihood", float)])


class LogLikelihoods:
    @staticmethod
    def token_list_from_dict(token_list: Optional[List[Dict]]):
        if token_list is None:
            return None
        return [TokenLogLikelihood(**token) for token in token_list]

    def __init__(self, prompt_tokens: List[TokenLogLikelihood], completion_tokens: List[TokenLogLikelihood]):
        self.prompt_tokens = self.token_list_from_dict(prompt_tokens)
        self.completion_tokens = self.token_list_from_dict(completion_tokens)
