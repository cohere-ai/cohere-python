from typing import Any, Dict, List, Optional

from cohere.responses.base import CohereObject, _df_html


class Tokens(CohereObject):
    def __init__(
        self,
        tokens: Optional[List[int]] = None,
        token_strings: Optional[List[str]] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        assert tokens is not None
        assert token_strings is not None
        self.tokens = tokens
        self.token_strings = token_strings
        self.meta = meta

    @property
    def length(self):
        return len(self)

    def __len__(self) -> int:
        return len(self.tokens)

    def visualize(self, **kwargs):
        import pandas as pd

        df = pd.DataFrame.from_dict({"token_strings": self.token_strings, "tokens": self.tokens}, orient="index")
        return _df_html(df.fillna(""), style={"font-size": "90%"})


class Detokenization(CohereObject):
    def __init__(self, text: Optional[str] = None, meta: Optional[Dict[str, Any]] = None) -> None:
        assert text is not None
        self.text = text
        self.meta = meta

    def __str__(self) -> str:
        return self.text

    def __eq__(self, __o: object) -> bool:
        return self.text == __o.text
