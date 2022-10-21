from concurrent.futures import Future
from subprocess import call
from typing import Any, Callable, Iterator
from xmlrpc.client import Boolean

from cohere.feedback import Feedback


class AsyncAttribute():

    def __init__(self, async_request: Future, getter: Callable[..., Any]) -> None:
        self._request = async_request
        self._getter = getter

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
        return self._getter(self._request.result())


class CohereObject():

    def __init__(self, client=None, call_id: int = None) -> None:
        self.client = client
        self.call_id = call_id

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

    def feedback(self, feedback: str) -> Feedback:
        return self.client.feedback(call_id=self.call_id, feedback=feedback)
