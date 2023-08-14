from typing import Dict, List, NamedTuple, Optional

from cohere.responses.base import CohereObject, _df_html

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

    @property
    def log_likelihood(self):
        return [token.log_likelihood for token in self.completion_tokens]

    def visualize(self, **kwargs):
        import pandas as pd

        html = ""
        for lbl, tokens in [("prprompt_tokensompt", self.prompt_tokens), ("completion_tokens", self.completion_tokens)]:
            if tokens is not None:
                html += f"<b>{lbl}</b><br>"
                df = pd.DataFrame.from_dict(
                    {
                        "decoded": [t.decoded for t in tokens],
                        "encoded": [t.encoded for t in tokens],
                        "log_likelihood": [t.log_likelihood for t in tokens],
                    },
                    orient="index",
                )
                html += _df_html(df.fillna(""), style={"font-size": "90%"})
        return html
