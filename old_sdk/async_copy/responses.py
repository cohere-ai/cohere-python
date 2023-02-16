import html
from collections import UserList
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Union

import numpy as np


def _escape_html(text):
    return html.escape(str(text), quote=False)


def _df_html(
    df, style: Optional[Dict] = None, drop_all_na=True, dont_escape=("token_likelihoods",), **kwargs
):  # keep html in some columns
    formatters = {c: str if c in dont_escape else _escape_html for c in df.columns}
    if drop_all_na:  # do not show likelihood etc if all missing
        df = df.dropna(axis=1, how="all")
    if style:
        df = df.style.set_properties(**style)
    kwargs = dict(escape=False, formatters=formatters, **kwargs)
    return df.to_html(**kwargs)


class CohereObject:
    def _repr_html_(self):  # rich html output for Jupyter
        try:
            return self.visualize()
        except (ImportError, AttributeError, NotImplementedError):  # no pandas or no visualize method
            return None  # ipython will use repr()


@dataclass
class Tokens(CohereObject):
    tokens: List[int]
    token_strings: List[str]

    def __len__(self) -> int:
        return len(self.tokens)

    def visualize(self, **kwargs):
        import pandas as pd

        df = pd.DataFrame.from_dict({"token_strings": self.token_strings, "tokens": self.tokens}, orient="index")
        return _df_html(df.fillna(""), style={"font-size": "90%"})


class BatchedTokens(UserList, CohereObject):
    """Acts a list of Tokens object"""

    # nice jupyter output
    def visualize(self, **kwargs) -> str:
        import pandas as pd

        df = pd.DataFrame.from_dict(
            {f"[{i}].{f}": getattr(t, f) for i, t in enumerate(self) for f in ["token_strings", "tokens"]},
            orient="index",
        )
        return _df_html(df.fillna(""), style={"font-size": "90%"})


# Inheriting from np.ndarray directly is not great, so we wrap it like UserList
class CohereNPArray(CohereObject):
    def __init__(self, data):
        self.data = np.array(data)

    def numpy(self) -> np.ndarray:
        return self.data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}: shape {self.data.shape}"

    def __len__(self):
        return len(self.data)


class Embedding(CohereNPArray):
    """Embedding has size (4096)"""

    def norm(self, ord=2) -> float:
        return np.linalg.norm(self.data, ord=ord)


class Embeddings(CohereNPArray):
    """Embeddings has size (n x 4096)"""

    def norm(self, ord=2) -> np.ndarray:
        return np.linalg.norm(self.data, axis=1, ord=ord)

    def closest(self, item: Embedding, n=1, distance="cosine") -> Union[int, np.ndarray]:
        """Returns the index (or indices if n>1) closest to the given item"""
        assert distance == "cosine", "only cosine distance is currently supported"
        cos_sim = np.dot(self.numpy(), item.numpy()) / self.norm()
        if n == 1:
            return np.argmax(cos_sim)
        else:
            return np.argpartition(-cos_sim, n)[:n]

    def __getitem__(self, key) -> Union[np.array, Embedding, "Embeddings"]:
        result = self.data[key]
        return Embedding(result) if isinstance(key, int) else Embeddings(result)


@dataclass
class TokenLikelihood:
    token: str
    likelihood: Optional[float]


TOKEN_COLORS = [
    (-2, "#FFECE2"),
    (-4, "#FFD6BC"),
    (-6, "#FFC59A"),
    (-8, "#FFB471"),
    (-10, "#FFA745"),
    (-12, "#FE9F00"),
    (-1e9, "#E18C00"),
]


@dataclass
class Generation(CohereObject):
    """A generation contains the generated text along with possibly likelihood and token likelihoods
    A few convenience functions are included to treat this object as its .text attribute directly."""

    prompt: str
    text: str
    likelihood: Optional[float]
    token_likelihoods: Optional[List[TokenLikelihood]]

    @classmethod
    def from_response(cls, response, prompt=None):
        token_likelihoods = response.get("token_likelihoods")
        if token_likelihoods:
            token_likelihoods = [TokenLikelihood(d["token"], d.get("likelihood")) for d in token_likelihoods]
        return cls(
            prompt=prompt,
            text=response.get("text"),
            likelihood=response.get("likelihood"),
            token_likelihoods=token_likelihoods,
        )

    # UserString doesn't work nicely, but we give a few methods at least
    def __getitem__(self, ix):
        return self.text[ix]

    def strip(self):
        return self.text.strip()

    def split(self, *args):
        return self.text.split(*args)

    def __str__(self):
        return self.text

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
        return {**asdict(self), "token_likelihoods": self.visualize_token_likelihoods(display=False)}

    def visualize(self, **kwargs) -> str:
        return Generations([self]).visualize(index=False, **kwargs)


class Generations(UserList, CohereObject):
    """A generations object acts as a list of generations. They are stored in order of likelihood when possible.
    As there is usually only one item, you can get the properties of the first item directly, most notably .text"""

    @classmethod
    def from_response(cls, response: Union[Dict, Exception]) -> Union["Generations", Generation]:
        if isinstance(response, Exception):
            return response
        gens = [Generation.from_response(gen, response.get("prompt")) for gen in response["generations"]]
        if len(gens) > 1 and gens[0].likelihood is not None:  # sort in order of likelihood
            gens.sort(key=lambda g: g.likelihood, reverse=True)
        return cls(gens)

    @property
    def generations(self):  # backward compatibility
        return self.data

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

    def __str__(self) -> str:
        return str(self[0])

    # nice jupyter output
    def visualize(self, **kwargs) -> str:
        import pandas as pd

        with pd.option_context("display.max_colwidth", 250):
            return _df_html(pd.DataFrame([g._visualize_helper() for g in self]), **kwargs)


class BatchedGenerations(UserList, CohereObject):
    """Acts a list of Generations object"""

    # nice jupyter output
    def visualize(self, **kwargs) -> str:
        import pandas as pd

        records = [
            {"prompt #": i, "generation #": gi, **gen._visualize_helper()}
            for i, items in enumerate(self)
            for gi, gen in enumerate(items if isinstance(items, Generations) else [items])  # items=Exception possible
        ]
        with pd.option_context("display.max_colwidth", 250):
            df = pd.DataFrame(records)
            if df["generation #"].nunique() == 1:
                df = df.drop("generation #", axis=1).set_index(["prompt #"], drop=True)
            else:
                df = df.set_index(["prompt #", "generation #"], drop=True)
            return _df_html(df, **kwargs)
