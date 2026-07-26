import unittest

from cohere import EmbeddingsByTypeEmbedResponse, EmbedByTypeResponseEmbeddings, ApiMeta, ApiMetaBilledUnits, \
    ApiMetaApiVersion, ApiMetaTokens, EmbeddingsFloatsEmbedResponse
from cohere.overrides import get_fields
from cohere.utils import merge_embed_responses, merge_meta_field, sum_fields_if_not_none

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

    def test_merge_embed_responses_empty_list_raises_value_error(self) -> None:
        # An empty texts list (e.g. Client.embed(texts=[])) yields no responses;
        # merge_embed_responses must raise a clear ValueError instead of IndexError.
        with self.assertRaises(ValueError):
            merge_embed_responses([])

    def test_merge_embeddings_by_type_with_none_field_in_later_response(self) -> None:
        resp1 = EmbeddingsByTypeEmbedResponse(
            response_type="embeddings_by_type", id="1",
            embeddings=EmbedByTypeResponseEmbeddings(float_=[[1.0, 2.0]]))
        resp2 = EmbeddingsByTypeEmbedResponse(
            response_type="embeddings_by_type", id="2",
            embeddings=EmbedByTypeResponseEmbeddings(float_=None))
        result = merge_embed_responses([resp1, resp2])
        self.assertEqual(result.embeddings.float_, [[1.0, 2.0]])  # type: ignore

    def test_merge_meta_field_keeps_tokens_and_image_units(self) -> None:
        merged = merge_meta_field([
            ApiMeta(
                api_version=ApiMetaApiVersion(version="1"),
                billed_units=ApiMetaBilledUnits(input_tokens=1, images=1, image_tokens=10),
                tokens=ApiMetaTokens(input_tokens=11, output_tokens=0),
                cached_tokens=3,
            ),
            ApiMeta(
                api_version=ApiMetaApiVersion(version="1"),
                billed_units=ApiMetaBilledUnits(input_tokens=2, images=2, image_tokens=20),
                tokens=ApiMetaTokens(input_tokens=22, output_tokens=0),
                cached_tokens=4,
            ),
        ])

        if merged.billed_units is None or merged.tokens is None:
            raise Exception("this is just for mypy")

        self.assertEqual(merged.billed_units.input_tokens, 3)
        self.assertEqual(merged.billed_units.images, 3)
        self.assertEqual(merged.billed_units.image_tokens, 30)
        self.assertEqual(merged.tokens.input_tokens, 33)
        self.assertEqual(merged.tokens.output_tokens, 0)
        self.assertEqual(merged.cached_tokens, 7)

    def test_merge_meta_field_leaves_tokens_unset_when_absent(self) -> None:
        merged = merge_meta_field([
            ApiMeta(billed_units=ApiMetaBilledUnits(input_tokens=1)),
            ApiMeta(billed_units=ApiMetaBilledUnits(input_tokens=2)),
        ])

        self.assertIsNone(merged.tokens)
        self.assertIsNone(merged.cached_tokens)

    def test_merge_meta_field_sums_every_numeric_field_on_the_model(self) -> None:
        # merge_meta_field lists the fields it copies by hand, so any field added
        # to ApiMeta later is silently dropped from every merged response until
        # someone remembers to update it. That is how images, image_tokens,
        # tokens and cached_tokens went missing. Drive the assertion off the
        # model itself so the next added field fails here instead of in the wild.
        billed_fields = get_fields(ApiMetaBilledUnits())
        token_fields = get_fields(ApiMetaTokens())
        meta = ApiMeta(
            billed_units=ApiMetaBilledUnits(**{field: 1 for field in billed_fields}),
            tokens=ApiMetaTokens(**{field: 1 for field in token_fields}),
            cached_tokens=1,
        )

        merged = merge_meta_field([meta, meta])

        if merged.billed_units is None or merged.tokens is None:
            raise Exception("this is just for mypy")

        for field in billed_fields:
            self.assertEqual(getattr(merged.billed_units, field), 2, f"billed_units.{field} was dropped")
        for field in token_fields:
            self.assertEqual(getattr(merged.tokens, field), 2, f"tokens.{field} was dropped")
        self.assertEqual(merged.cached_tokens, 2)

    def test_sum_fields_if_not_none_with_none_entries(self) -> None:
        # billed_units list may contain None when ApiMeta.billed_units is unset;
        # sum_fields_if_not_none must skip None objects without raising AttributeError
        result = sum_fields_if_not_none([None, ApiMetaBilledUnits(input_tokens=5), None], "input_tokens")
        self.assertEqual(result, 5)

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
