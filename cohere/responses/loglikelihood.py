from typing import Dict, List, NamedTuple, Optional

from cohere.responses.base import CohereObject, _df_html, _escape_html

TokenLogLikelihood = NamedTuple("TokenLogLikelihood", [("encoded", int), ("decoded", str), ("log_likelihood", float)])


class LogLikelihoods(CohereObject):
    @staticmethod
    def token_list_from_dict(token_list: Optional[List[Dict]]):
        if token_list is None:
            return None
        return [TokenLogLikelihood(**token) for token in token_list]

    def __init__(self, prompt_tokens: List[TokenLogLikelihood], completion_tokens: List[TokenLogLikelihood]):
        self.prompt_tokens = self.token_list_from_dict(prompt_tokens)
        self.completion_tokens = self.token_list_from_dict(completion_tokens)

    def visualize(self, **kwargs):
        import pandas as pd

        dfs = []
        for lbl, tokens in [("prompt_tokens", self.prompt_tokens), ("completion_tokens", self.completion_tokens)]:
            if tokens is not None:
                dfs.append(
                    pd.DataFrame.from_dict(
                        {
                            lbl + ".decoded": [_escape_html(t.decoded) for t in tokens],
                            lbl + ".encoded": [t.encoded for t in tokens],
                            lbl + ".log_likelihood": [t.log_likelihood for t in tokens],
                        },
                        orient="index",
                    )
                )
        return _df_html(pd.concat(dfs, axis=0).fillna(""), style={"font-size": "90%"})
