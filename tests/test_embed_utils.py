import os
import unittest

import cohere

from cohere import EmbedResponse_EmbeddingsByType, EmbedByTypeResponseEmbeddings, ApiMeta, ApiMetaBilledUnits
from cohere.utils import merge_embed_responses

from src.cohere import ApiMetaApiVersion, EmbedResponse_EmbeddingsFloats

ebt_1 = EmbedResponse_EmbeddingsByType(
    response_type="embeddings_by_type",
    id="1",
    embeddings=EmbedByTypeResponseEmbeddings(
        float_=[[0, 1, 2], [3, 4, 5]],
        int8=[[0, 1, 2], [3, 4, 5]],
        uint8=[[0, 1, 2], [3, 4, 5]],
        binary=[[0, 1, 2], [3, 4, 5]],
        ubinary=[[0, 1, 2], [3, 4, 5]],
    ),
    texts=["hello", "goodbye"],
    meta=ApiMeta(
        api_version=ApiMetaApiVersion(version="1"),
        billed_units=ApiMetaBilledUnits(
            input_tokens=1,
            output_tokens=1,
            search_units=1,
            classifications=1
        ),
        warnings=["test_warning_1"]
    )
)

ebt_2 = EmbedResponse_EmbeddingsByType(
    response_type="embeddings_by_type",
    id="2",
    embeddings=EmbedByTypeResponseEmbeddings(
        float_=[[7, 8, 9], [10, 11, 12]],
        int8=[[7, 8, 9], [10, 11, 12]],
        uint8=[[7, 8, 9], [10, 11, 12]],
        binary=[[7, 8, 9], [10, 11, 12]],
        ubinary=[[7, 8, 9], [10, 11, 12]],
    ),
    texts=["bye", "seeya"],
    meta=ApiMeta(
        api_version=ApiMetaApiVersion(version="1"),
        billed_units=ApiMetaBilledUnits(
            input_tokens=2,
            output_tokens=2,
            search_units=2,
            classifications=2
        ),
        warnings=["test_warning_1", "test_warning_2"]
    )
)

ebf_1 = EmbedResponse_EmbeddingsFloats(
    response_type="embeddings_floats",
    id="1",
    texts=["hello", "goodbye"],
    embeddings=[[0, 1, 2], [3, 4, 5]],
    meta=ApiMeta(
        api_version=ApiMetaApiVersion(version="1"),
        billed_units=ApiMetaBilledUnits(
            input_tokens=1,
            output_tokens=1,
            search_units=1,
            classifications=1
        ),
        warnings=["test_warning_1"]
    )
)

ebf_2 = EmbedResponse_EmbeddingsFloats(
    response_type="embeddings_floats",
    id="2",
    texts=["bye", "seeya"],
    embeddings=[[7, 8, 9], [10, 11, 12]],
    meta=ApiMeta(
        api_version=ApiMetaApiVersion(version="1"),
        billed_units=ApiMetaBilledUnits(
            input_tokens=2,
            output_tokens=2,
            search_units=2,
            classifications=2
        ),
        warnings=["test_warning_1", "test_warning_2"]
    )
)


class TestClient(unittest.TestCase):

    def test_merge_embeddings_by_type(self) -> None:
        resp = merge_embed_responses([
            ebt_1,
            ebt_2
        ])

        self.assertEqual(set(resp.meta.warnings), {"test_warning_1", "test_warning_2"})
        self.assertEqual(resp, EmbedResponse_EmbeddingsByType(
            response_type="embeddings_by_type",
            id="1, 2",
            embeddings=EmbedByTypeResponseEmbeddings(
                float_=[[0, 1, 2], [3, 4, 5], [7, 8, 9], [10, 11, 12]],
                int8=[[0, 1, 2], [3, 4, 5], [7, 8, 9], [10, 11, 12]],
                uint8=[[0, 1, 2], [3, 4, 5], [7, 8, 9], [10, 11, 12]],
                binary=[[0, 1, 2], [3, 4, 5], [7, 8, 9], [10, 11, 12]],
                ubinary=[[0, 1, 2], [3, 4, 5], [7, 8, 9], [10, 11, 12]],
            ),
            texts=["hello", "goodbye", "bye", "seeya"],
            meta=ApiMeta(
                api_version=ApiMetaApiVersion(version="1"),
                billed_units=ApiMetaBilledUnits(
                    input_tokens=3,
                    output_tokens=3,
                    search_units=3,
                    classifications=3
                ),
                warnings=resp.meta.warnings  # order ignored
            )
        ))

    def test_merge_embeddings_floats(self) -> None:
        resp = merge_embed_responses([
            ebf_1,
            ebf_2
        ])

        self.assertEqual(set(resp.meta.warnings), {"test_warning_1", "test_warning_2"})
        self.assertEqual(resp, EmbedResponse_EmbeddingsFloats(
            response_type="embeddings_floats",
            id="1, 2",
            texts=["hello", "goodbye", "bye", "seeya"],
            embeddings=[[0, 1, 2], [3, 4, 5], [7, 8, 9], [10, 11, 12]],
            meta=ApiMeta(
                api_version=ApiMetaApiVersion(version="1"),
                billed_units=ApiMetaBilledUnits(
                    input_tokens=3,
                    output_tokens=3,
                    search_units=3,
                    classifications=3
                ),
                warnings=resp.meta.warnings  # order ignored
            )
        ))
