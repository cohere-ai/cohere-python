"""Tests for ClientV2.embed_stream(), which batches requests to /v2/embed.

Requests go through a mock HTTP transport, so these tests check the request bodies the client
actually sends.
"""

import json
import typing
import unittest

import httpx

import cohere
from cohere import EmbedInput, TextEmbedContent


class _RecordingEmbedServer:
    """Answers /v2/embed with one embedding per entry holding the entry's integer value."""

    def __init__(self) -> None:
        self.requests: typing.List[typing.Dict[str, typing.Any]] = []

    def handle(self, request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v2/embed", request.url.path
        body = json.loads(request.content)
        self.requests.append(body)
        if "texts" in body:
            values = [int(text) for text in body["texts"]]
        else:
            values = [int(entry["content"][0]["text"]) for entry in body["inputs"]]
        embeddings = {
            embedding_type: [[value] for value in values]
            for embedding_type in body.get("embedding_types", ["float"])
        }
        return httpx.Response(
            200,
            json={
                "response_type": "embeddings_by_type",
                "id": f"req-{len(self.requests)}",
                "embeddings": embeddings,
                "texts": body.get("texts"),
                "meta": {"billed_units": {"input_tokens": len(values)}},
            },
        )


def _client(server: _RecordingEmbedServer) -> cohere.ClientV2:
    return cohere.ClientV2(
        api_key="test",
        base_url="https://api.test",
        httpx_client=httpx.Client(transport=httpx.MockTransport(server.handle)),
    )


TEXTS = [str(i) for i in range(250)]


class TestClientV2EmbedStream(unittest.TestCase):

    def test_batches_texts_to_v2_with_v2_params(self) -> None:
        server = _RecordingEmbedServer()

        results = list(
            _client(server).embed_stream(
                model="embed-v4.0",
                input_type="search_document",
                texts=TEXTS,
                embedding_types=["float"],
                output_dimension=256,
                max_tokens=512,
                truncate="END",
                priority=1,
            )
        )

        self.assertEqual([len(r["texts"]) for r in server.requests], [96, 96, 58])
        for request in server.requests:
            self.assertEqual(request["model"], "embed-v4.0")
            self.assertEqual(request["input_type"], "search_document")
            self.assertEqual(request["embedding_types"], ["float"])
            self.assertEqual(request["output_dimension"], 256)
            self.assertEqual(request["max_tokens"], 512)
            self.assertEqual(request["truncate"], "END")
            self.assertEqual(request["priority"], 1)
        self.assertEqual([r.index for r in results], list(range(250)))
        self.assertEqual([r.text for r in results], TEXTS)
        self.assertEqual([r.embedding for r in results], [[float(i)] for i in range(250)])
        self.assertTrue(all(r.embedding_type == "float" for r in results))

    def test_batches_inputs(self) -> None:
        server = _RecordingEmbedServer()
        inputs = [EmbedInput(content=[TextEmbedContent(text=text)]) for text in TEXTS[:100]]

        results = list(_client(server).embed_stream(model="embed-v4.0", input_type="search_document", inputs=inputs))

        self.assertEqual([len(r["inputs"]) for r in server.requests], [96, 4])
        self.assertTrue(all("texts" not in r for r in server.requests))
        self.assertEqual([r.index for r in results], list(range(100)))
        self.assertTrue(all(r.text is None for r in results))

    def test_yields_every_embedding_type(self) -> None:
        server = _RecordingEmbedServer()

        results = list(
            _client(server).embed_stream(
                model="embed-v4.0",
                input_type="search_document",
                texts=TEXTS[:100],
                embedding_types=["float", "int8"],
            )
        )

        for embedding_type in ("float", "int8"):
            typed = [r for r in results if r.embedding_type == embedding_type]
            self.assertEqual([r.index for r in typed], list(range(100)))
            self.assertEqual([r.text for r in typed], TEXTS[:100])

    def test_custom_batch_size(self) -> None:
        server = _RecordingEmbedServer()

        list(_client(server).embed_stream(model="embed-v4.0", input_type="search_document", texts=TEXTS[:25], batch_size=10))

        self.assertEqual([len(r["texts"]) for r in server.requests], [10, 10, 5])

    def test_sends_next_request_only_when_consumed(self) -> None:
        server = _RecordingEmbedServer()

        stream = _client(server).embed_stream(model="embed-v4.0", input_type="search_document", texts=TEXTS)
        first = next(stream)

        self.assertEqual(first.index, 0)
        self.assertEqual(len(server.requests), 1)

    def test_model_and_input_type_stay_optional(self) -> None:
        server = _RecordingEmbedServer()

        results = list(_client(server).embed_stream(texts=TEXTS[:5]))

        self.assertEqual(len(results), 5)
        self.assertNotIn("model", server.requests[0])
        self.assertNotIn("input_type", server.requests[0])

    def test_empty_texts_sends_nothing(self) -> None:
        server = _RecordingEmbedServer()

        results = list(_client(server).embed_stream(model="embed-v4.0", input_type="search_document", texts=[]))

        self.assertEqual(results, [])
        self.assertEqual(server.requests, [])

    def test_rejects_invalid_arguments(self) -> None:
        client = _client(_RecordingEmbedServer())
        inputs = [EmbedInput(content=[TextEmbedContent(text="0")])]

        with self.assertRaises(ValueError):
            list(client.embed_stream(model="embed-v4.0", input_type="search_document"))
        with self.assertRaises(ValueError):
            list(client.embed_stream(model="embed-v4.0", input_type="search_document", texts=["0"], inputs=inputs))
        with self.assertRaises(ValueError):
            list(client.embed_stream(model="embed-v4.0", input_type="search_document", texts=["0"], batch_size=0))


if __name__ == "__main__":
    unittest.main()
