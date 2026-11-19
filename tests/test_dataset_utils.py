from http import HTTPStatus
from io import BytesIO
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock

import pytest
import requests

import cohere.utils
from cohere import Dataset
from cohere.utils import dataset_generator


class TrackingResponse(requests.Response):
    def __init__(self, status_code: int = 200) -> None:
        super().__init__()
        self.status_code = status_code
        self.reason = HTTPStatus(status_code).phrase
        self.url = "https://example.test/part.avro"
        self.raw = BytesIO(b"not an Avro response")
        self.closed = False

    def close(self) -> None:
        super().close()
        self.closed = True


def test_dataset_generator_closes_response_after_exhaustion(monkeypatch: Any) -> None:
    response = TrackingResponse()
    monkeypatch.setattr(cohere.utils.requests, "get", lambda *_args, **_kwargs: response)
    monkeypatch.setattr(cohere.utils, "reader", lambda raw: iter([{"id": 1}]))
    dataset = cast(Dataset, SimpleNamespace(dataset_parts=[SimpleNamespace(url="https://example.test/part.avro")]))

    assert list(dataset_generator(dataset)) == [{"id": 1}]
    assert response.closed


@pytest.mark.parametrize("status_code", [403, 429, 503])
def test_dataset_generator_preserves_http_errors(monkeypatch: Any, status_code: int) -> None:
    response = TrackingResponse(status_code)
    avro_reader = Mock(side_effect=ValueError("not an Avro response"))
    monkeypatch.setattr(cohere.utils.requests, "get", lambda *_args, **_kwargs: response)
    monkeypatch.setattr(cohere.utils, "reader", avro_reader)
    dataset = cast(Dataset, SimpleNamespace(dataset_parts=[SimpleNamespace(url=response.url)]))

    with pytest.raises(requests.HTTPError) as exc_info:
        list(dataset_generator(dataset))

    assert exc_info.value.response is response
    assert exc_info.value.response.status_code == status_code
    avro_reader.assert_not_called()
    assert response.closed
    assert response.raw.closed


def test_dataset_generator_preserves_decode_errors_and_closes_response(monkeypatch: Any) -> None:
    response = TrackingResponse()
    decode_error = ValueError("invalid Avro body")
    monkeypatch.setattr(cohere.utils.requests, "get", lambda *_args, **_kwargs: response)
    monkeypatch.setattr(cohere.utils, "reader", Mock(side_effect=decode_error))
    dataset = cast(Dataset, SimpleNamespace(dataset_parts=[SimpleNamespace(url=response.url)]))

    with pytest.raises(ValueError) as exc_info:
        list(dataset_generator(dataset))

    assert exc_info.value is decode_error
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
