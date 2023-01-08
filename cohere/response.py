from concurrent.futures import Future
from typing import Any, Callable, Iterator
from xmlrpc.client import Boolean

from cohere.feedback import Feedback


class AsyncAttribute():
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


class CohereObject():

    def __init__(self, client=None, id: str = None) -> None:
        self.client = client
        self.id = id

    def __getattribute__(self, name: str) -> Any:
        attr = super().__getattribute__(name)
        if isinstance(attr, AsyncAttribute):
            return attr.resolve()
        else:
            return attr

    def __repr__(self) -> str:
        contents = ''
        exclude_list = ['iterator']

        for k in self.__dict__.keys():
            if k not in exclude_list:
                contents += f'\t{k}: {self.__dict__[k]}\n'

        output = f'cohere.{type(self).__name__} {{\n{contents}}}'
        return output

    def feedback(self, feedback: str, accepted: bool) -> Feedback:
        return self.client.feedback(id=self.id, feedback=feedback, accepted=accepted)
