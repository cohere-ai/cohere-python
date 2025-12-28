import unittest

from cohere import EmbeddingsByTypeEmbedResponse, EmbedByTypeResponseEmbeddings, ApiMeta, ApiMetaBilledUnits, \
    ApiMetaApiVersion, EmbeddingsFloatsEmbedResponse
from cohere.utils import merge_embed_responses

ebt_1 = EmbeddingsByTypeEmbedResponse(
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

ebt_2 = EmbeddingsByTypeEmbedResponse(
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

ebt_partial_1 = EmbeddingsByTypeEmbedResponse(
    response_type="embeddings_by_type",
    id="1",
    embeddings=EmbedByTypeResponseEmbeddings(
        float_=[[0, 1, 2], [3, 4, 5]],
        int8=[[0, 1, 2], [3, 4, 5]],
        binary=[[5, 6, 7], [8, 9, 10]],
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

ebt_partial_2 = EmbeddingsByTypeEmbedResponse(
    response_type="embeddings_by_type",
    id="2",
    embeddings=EmbedByTypeResponseEmbeddings(
        float_=[[7, 8, 9], [10, 11, 12]],
        int8=[[7, 8, 9], [10, 11, 12]],
        binary=[[14, 15, 16], [17, 18, 19]],
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

ebf_1 = EmbeddingsFloatsEmbedResponse(
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

ebf_2 = EmbeddingsFloatsEmbedResponse(
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

        if resp.meta is None:
            raise Exception("this is just for mpy")

        self.assertEqual(set(resp.meta.warnings or []), {"test_warning_1", "test_warning_2"})
        self.assertEqual(resp, EmbeddingsByTypeEmbedResponse(
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

        if resp.meta is None:
            raise Exception("this is just for mpy")

        self.assertEqual(set(resp.meta.warnings or []), {"test_warning_1", "test_warning_2"})
        self.assertEqual(resp, EmbeddingsFloatsEmbedResponse(
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

    def test_merge_partial_embeddings_floats(self) -> None:
        resp = merge_embed_responses([
            ebt_partial_1,
            ebt_partial_2
        ])

        if resp.meta is None:
            raise Exception("this is just for mpy")

        self.assertEqual(set(resp.meta.warnings or []), {"test_warning_1", "test_warning_2"})
        self.assertEqual(resp, EmbeddingsByTypeEmbedResponse(
            response_type="embeddings_by_type",
            id="1, 2",
            embeddings=EmbedByTypeResponseEmbeddings(
                float_=[[0, 1, 2], [3, 4, 5], [7, 8, 9], [10, 11, 12]],
                int8=[[0, 1, 2], [3, 4, 5], [7, 8, 9], [10, 11, 12]],
                binary=[[5, 6, 7], [8, 9, 10], [14, 15, 16], [17, 18, 19]],
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

    def test_image_tokens_field(self) -> None:
        """Test that image_tokens field is properly handled in ApiMetaBilledUnits.

        This is a regression test for issue #711 where the image_tokens field
        was missing from the response model.
        """
        # Test that image_tokens can be set and accessed
        billed_units = ApiMetaBilledUnits(
            input_tokens=100,
            image_tokens=2716
        )
        self.assertEqual(billed_units.image_tokens, 2716)
        self.assertEqual(billed_units.input_tokens, 100)

        # Test serialization includes image_tokens
        dumped = billed_units.model_dump()
        self.assertEqual(dumped["image_tokens"], 2716)

    def test_merge_with_image_tokens(self) -> None:
        """Test that image_tokens are properly merged across responses."""
        ebt_with_images_1 = EmbeddingsByTypeEmbedResponse(
            response_type="embeddings_by_type",
            id="1",
            embeddings=EmbedByTypeResponseEmbeddings(
                float_=[[0, 1, 2]],
            ),
            texts=["hello"],
            meta=ApiMeta(
                api_version=ApiMetaApiVersion(version="1"),
                billed_units=ApiMetaBilledUnits(
                    input_tokens=1,
                    image_tokens=100
                ),
            )
        )

        ebt_with_images_2 = EmbeddingsByTypeEmbedResponse(
            response_type="embeddings_by_type",
            id="2",
            embeddings=EmbedByTypeResponseEmbeddings(
                float_=[[3, 4, 5]],
            ),
            texts=["goodbye"],
            meta=ApiMeta(
                api_version=ApiMetaApiVersion(version="1"),
                billed_units=ApiMetaBilledUnits(
                    input_tokens=2,
                    image_tokens=200
                ),
            )
        )

        resp = merge_embed_responses([ebt_with_images_1, ebt_with_images_2])

        self.assertIsNotNone(resp.meta)
        self.assertIsNotNone(resp.meta.billed_units)
        self.assertEqual(resp.meta.billed_units.image_tokens, 300)
        self.assertEqual(resp.meta.billed_units.input_tokens, 3)
