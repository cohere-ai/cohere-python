from concurrent.futures import Future
from typing import Any, Dict, Iterator, List, NamedTuple, Optional, Union

from cohere.responses.base import CohereObject, _df_html
import html
from collections import UserList
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Union


TokenLikelihood = NamedTuple("TokenLikelihood", [("token", str), ("likelihood", float)])

TOKEN_COLORS = [
    (-2, "#FFECE2"),
    (-4, "#FFD6BC"),
    (-6, "#FFC59A"),
    (-8, "#FFB471"),
    (-10, "#FFA745"),
    (-12, "#FE9F00"),
    (-1e9, "#E18C00"),
]


class Generation(CohereObject, str):

    def __new__(cls, text: str, *_, **__):
        return str.__new__(cls, text)

    def __init__(self, text: str, likelihood: float, token_likelihoods: List[TokenLikelihood], prompt: str=None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.prompt=prompt
        self.text = text
        self.likelihood = likelihood
        self.token_likelihoods = token_likelihoods

    @classmethod
    def from_response(cls, response, prompt=None, **kwargs):
        token_likelihoods = response.get("token_likelihoods")
        if token_likelihoods:
            token_likelihoods = [TokenLikelihood(d["token"], d.get("likelihood")) for d in token_likelihoods]
        return cls(
            text=response.get("text"),
            likelihood=response.get("likelihood"),
            token_likelihoods=token_likelihoods,
            prompt=prompt, 
            id=response.get('id'),
            **kwargs        
        )
    # nice jupyter output
    def visualize_token_likelihoods(self, ignore_first_n=0, midpoint=-3, value_range=8, display=True):  # very WIP
        if self.token_likelihoods is None:
            return None

        def color_token(i, t: TokenLikelihood):
            if t.likelihood is None or i < ignore_first_n:
                col = "#EDEDED"
            else:
                col = next(c for thr, c in TOKEN_COLORS if t.likelihood >= thr)  # first hit
            return f"<span style='background-color:{col}'>{t.token}</span>"

        html = "".join(color_token(i, t) for i, t in enumerate(self.token_likelihoods))
        if display:
            from IPython.display import HTML

            return HTML(html)  # show in jupyter by default, but allow to be used as helper
        return html

    def _visualize_helper(self):
        return dict(prompt=self.prompt,text=self.text,likelihood=self.likelihood, token_likelihoods= self.visualize_token_likelihoods(display=False))

    def visualize(self, **kwargs) -> str: # TODO: this was Generations([self]) but that no longer works
        import pandas as pd
        with pd.option_context("display.max_colwidth", 250):
            return _df_html(pd.DataFrame([self._visualize_helper()]), **kwargs)


class Generations(CohereObject):

    def __init__(self,
                 return_likelihoods: str,
                 response: Optional[Dict[str, Any]] = None,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.return_likelihoods = return_likelihoods
        self.generations = self._generations(response)

    def _generations(self, response: Dict[str, Any]) -> List[Generation]:
        generations: List[Generation] = []
        for gen in response['generations']:
            likelihood = None
            token_likelihoods = None
            if self.return_likelihoods in ['GENERATION', 'ALL']:
                likelihood = gen['likelihood']
            if 'token_likelihoods' in gen.keys():
                token_likelihoods = []
                for likelihoods in gen['token_likelihoods']:
                    token_likelihood = likelihoods['likelihood'] if 'likelihood' in likelihoods.keys() else None
                    token_likelihoods.append(TokenLikelihood(likelihoods['token'], token_likelihood))
            generations.append(Generation(gen['text'], likelihood, token_likelihoods, prompt= response.get("prompt"), id=gen["id"], client=self.client))

        return generations

    def __str__(self) -> str:
        return str(self.generations)

    def __iter__(self) -> Iterator:
        return iter(self.generations)

    def __getitem__(self, index) -> Generation:
        return self.generations[index]

    # nice jupyter output
    def visualize(self, **kwargs) -> str:
        import pandas as pd

        with pd.option_context("display.max_colwidth", 250):
            return _df_html(pd.DataFrame([g._visualize_helper() for g in self]), **kwargs)

    # TODO: keep?
    @property
    def texts(self) -> str:
        """Returns the generated texts in order of highest likelihood (if we know this) or any order otherwise"""
        return [g.text for g in self.generations]

    @property
    def text(self) -> str:
        """Returns the generated text with the highest likelihood (if we know this), or the first otherwise"""
        return self.texts[0]

    @property
    def prompt(self) -> str:
        """Returns the prompt used as input"""
        return self[0].prompt  # should all be the same

