from types import SimpleNamespace
from typing import Any, cast

import cohere.utils
from cohere import Dataset
from cohere.utils import dataset_generator


class TrackingResponse:
    def __init__(self) -> None:
        self.raw = object()
        self.closed = False

    def close(self) -> None:
        self.closed = True

    def __enter__(self) -> "TrackingResponse":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


def test_dataset_generator_closes_response_after_exhaustion(monkeypatch: Any) -> None:
    response = TrackingResponse()
    monkeypatch.setattr(cohere.utils.requests, "get", lambda *_args, **_kwargs: response)
    monkeypatch.setattr(cohere.utils, "reader", lambda raw: iter([{"id": 1}]))
    dataset = cast(Dataset, SimpleNamespace(dataset_parts=[SimpleNamespace(url="https://example.test/part.avro")]))

    assert list(dataset_generator(dataset)) == [{"id": 1}]
    assert response.closed


def test_dataset_generator_closes_response_when_consumer_stops(monkeypatch: Any) -> None:
    response = TrackingResponse()
    monkeypatch.setattr(cohere.utils.requests, "get", lambda *_args, **_kwargs: response)
    monkeypatch.setattr(cohere.utils, "reader", lambda raw: iter([{"id": 1}, {"id": 2}]))
    dataset = cast(Dataset, SimpleNamespace(dataset_parts=[SimpleNamespace(url="https://example.test/part.avro")]))
    records = dataset_generator(dataset)

    assert next(records) == {"id": 1}
    records.close()

    assert response.closed
