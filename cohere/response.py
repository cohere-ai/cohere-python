from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, Dict
from xmlrpc.client import Boolean


class AsyncAttribute():
    def __init__(self, async_request: Future, getter: Callable[Any, Any]) -> None:
        self._request = async_request
        self._getter = getter

    def is_resolved() -> Boolean:
        return self._request.done()

    def resolve() -> Any:
        return self._getter(self._request.result())

    def value() -> Any:
        return self.resolve()



class CohereObject():

    def __getattribute__(self, name: str) -> Any:
        attr = super().__getattribute__(name)
        if isinstance(attr, AsyncAttribute):
            if attr.is_resolved():
                return attr.value()
            
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
