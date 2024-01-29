from typing import Dict, Optional
from .. import TokenizeResponse, Generation
import html

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
        
def tokenize_response_visualize(self, **kwargs):
    import pandas as pd

    df = pd.DataFrame.from_dict({"token_strings": self.token_strings, "tokens": self.tokens}, orient="index")
    return _df_html(df.fillna(""), style={"font-size": "90%"})


setattr(TokenizeResponse, 'visualize', tokenize_response_visualize)

def generation_visualize(self, **kwargs) -> str:
    import pandas as pd

    with pd.option_context("display.max_colwidth", 250):
        return _df_html(pd.DataFrame([g._visualize_helper() for g in self]), **kwargs)

setattr(Generation, 'visualize', generation_visualize)