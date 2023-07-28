import html
from concurrent.futures import Future
from typing import Any, Callable, Dict, Iterator, Optional
from xmlrpc.client import Boolean


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


class AsyncAttribute:
    """An attribute of an object that is lazily fetched.

    `async_request` is a Future object that is expected to resolve to an object that will be consumed by `getter`.
    `getter` is a function that recieves the result of `async_request` and processes it into the desired attribute.

    `getter` is only called once and its result is cached.
    """

    def __init__(self, async_request: Future, getter: Callable[..., Any]) -> None:
        self._request = async_request
        self._getter = getter
        self._resolved = False

    def __len__(self):
        return len(self.resolve())

    def __iter_(self) -> Iterator:
        return iter(self.resolve())

    def __repr__(self):
        return repr(self.resolve())

    def __str__(self):
        return str(self.resolve())

    def is_resolved(self) -> Boolean:
        return self._request.done()

    def resolve(self) -> Any:
        if "_result" in self.__dict__:
            return self._result

        self._result = self._getter(self._request.result())
        return self._result


class CohereObject:
    def __init__(self, id: str = None) -> None:
        self.id = id

    def __getattribute__(self, name: str) -> Any:
        attr = super().__getattribute__(name)
        if isinstance(attr, AsyncAttribute):
            return attr.resolve()
        else:
            return attr

    def __repr__(self) -> str:
        contents = ""
        exclude_list = ["iterator", "_wait_fn"]

        for k in self.__dict__.keys():
            if k not in exclude_list:
                contents += f"\t{k}: {self.__dict__[k]}\n"

        output = f"cohere.{type(self).__name__} {{\n{contents}}}"
        return output

    def _repr_html_(self):  # rich html output for Jupyter
        try:
            return self.visualize()
        except (ImportError, AttributeError, NotImplementedError):  # no pandas or no visualize method
            return None  # ipython will use repr()
