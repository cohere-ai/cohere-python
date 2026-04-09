"""Utilities for streaming embed responses without loading all embeddings into memory."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Optional, Union


@dataclass
class StreamedEmbedding:
    """A single embedding yielded incrementally from embed_stream()."""
    index: int
    embedding: Union[List[float], List[int]]
    embedding_type: str
    text: Optional[str] = None


def extract_embeddings_from_response(
    response_data: dict,
    batch_texts: List[str],
    global_offset: int = 0,
) -> Iterator[StreamedEmbedding]:
    """
    Extract individual embeddings from a Cohere embed response dict.

    Works for both V1 (embeddings_floats / embeddings_by_type) and V2 response formats.

    Args:
        response_data: Parsed JSON response from embed endpoint
        batch_texts: The texts that were embedded in this batch
        global_offset: Starting index for this batch within the full dataset

    Yields:
        StreamedEmbedding objects
    """
    response_type = response_data.get("response_type", "")

    if response_type == "embeddings_floats":
        embeddings = response_data.get("embeddings", [])
        for i, embedding in enumerate(embeddings):
            yield StreamedEmbedding(
                index=global_offset + i,
                embedding=embedding,
                embedding_type="float",
                text=batch_texts[i] if i < len(batch_texts) else None,
            )

    elif response_type == "embeddings_by_type":
        embeddings_obj = response_data.get("embeddings", {})
        for emb_type, embeddings_list in embeddings_obj.items():
            type_name = emb_type.rstrip("_")
            if isinstance(embeddings_list, list):
                for i, embedding in enumerate(embeddings_list):
                    yield StreamedEmbedding(
                        index=global_offset + i,
                        embedding=embedding,
                        embedding_type=type_name,
                        text=batch_texts[i] if i < len(batch_texts) else None,
                    )

    else:
        # V2 format: embeddings is a dict with type keys directly
        embeddings_obj = response_data.get("embeddings", {})
        if isinstance(embeddings_obj, dict):
            for emb_type, embeddings_list in embeddings_obj.items():
                type_name = emb_type.rstrip("_")
                if isinstance(embeddings_list, list):
                    for i, embedding in enumerate(embeddings_list):
                        yield StreamedEmbedding(
                            index=global_offset + i,
                            embedding=embedding,
                            embedding_type=type_name,
                            text=batch_texts[i] if i < len(batch_texts) else None,
                        )
