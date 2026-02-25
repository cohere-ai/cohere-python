"""Utilities for streaming large responses without loading everything into memory."""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from typing import Iterator, List, Optional, Union

import httpx

try:
    import ijson  # type: ignore
    IJSON_AVAILABLE = True
except ImportError:
    IJSON_AVAILABLE = False


@dataclass
class StreamedEmbedding:
    """A single embedding that can be processed without loading all embeddings into memory."""
    index: int
    embedding: Union[List[float], List[int], str]  # float, int8, uint8, binary, ubinary, base64
    embedding_type: str
    text: Optional[str] = None


class StreamingEmbedParser:
    """
    Parses embed responses incrementally using ijson for memory efficiency.
    Falls back to regular JSON parsing if ijson is not available.
    """

    def __init__(self, response: httpx.Response, batch_texts: Optional[List[str]] = None):
        """
        Initialize the streaming parser.

        Args:
            response: The httpx response object
            batch_texts: The original texts for this batch (for correlation)
        """
        self.response = response
        self.batch_texts = batch_texts or []
        self.embeddings_yielded = 0
        self._response_content: Optional[bytes] = None

    def iter_embeddings(self) -> Iterator[StreamedEmbedding]:
        """
        Iterate over embeddings one at a time without loading all into memory.

        Yields:
            StreamedEmbedding objects as they are parsed from the response
        """
        # Try to buffer the response content first to allow fallback if ijson fails
        # This trades some memory for reliability
        if self._response_content is None:
            try:
                content = self.response.content
                if isinstance(content, bytes):
                    self._response_content = content
            except Exception:
                # Content not available as bytes, will use json() method
                pass

        if not IJSON_AVAILABLE or self._response_content is None:
            # Fallback to regular parsing if ijson not available or no bytes content
            yield from self._iter_embeddings_fallback()
            return

        try:
            # Use ijson for memory-efficient parsing
            parser = ijson.parse(io.BytesIO(self._response_content))
            yield from self._parse_with_ijson(parser)
        except Exception:
            # If ijson parsing fails, fallback to regular parsing
            yield from self._iter_embeddings_fallback()
    
    def _parse_with_ijson(self, parser) -> Iterator[StreamedEmbedding]:
        """Parse embeddings using ijson incremental parser."""
        current_embedding: List[Union[float, int]] = []
        response_type = None
        # Track index per embedding type to properly map to texts
        type_indices: dict[str, int] = {}

        for prefix, event, value in parser:
            # Detect response type
            if prefix == 'response_type':
                response_type = value

            # Handle embeddings based on response type
            if response_type == 'embeddings_floats':
                # Simple float array format
                if prefix.startswith('embeddings.item.item'):
                    current_embedding.append(value)
                elif prefix.startswith('embeddings.item') and event == 'end_array':
                    # Complete embedding
                    text_index = type_indices.get('float', 0)
                    text = self.batch_texts[text_index] if text_index < len(self.batch_texts) else None
                    yield StreamedEmbedding(
                        index=self.embeddings_yielded,
                        embedding=list(current_embedding),
                        embedding_type='float',
                        text=text
                    )
                    self.embeddings_yielded += 1
                    type_indices['float'] = text_index + 1
                    current_embedding = []

            elif response_type == 'embeddings_by_type':
                # Complex format with multiple embedding types
                # Pattern: embeddings.<type>.item.item
                for emb_type in ['float_', 'int8', 'uint8', 'binary', 'ubinary']:
                    type_name = emb_type.rstrip('_')
                    if prefix.startswith(f'embeddings.{emb_type}.item.item'):
                        current_embedding.append(value)
                    elif prefix.startswith(f'embeddings.{emb_type}.item') and event == 'end_array':
                        # Complete embedding of this type
                        # Use separate index per type to correctly map to texts
                        text_index = type_indices.get(type_name, 0)
                        text = self.batch_texts[text_index] if text_index < len(self.batch_texts) else None
                        yield StreamedEmbedding(
                            index=self.embeddings_yielded,
                            embedding=list(current_embedding),
                            embedding_type=type_name,
                            text=text
                        )
                        self.embeddings_yielded += 1
                        type_indices[type_name] = text_index + 1
                        current_embedding = []

                # Handle base64 embeddings (string format)
                if prefix.startswith('embeddings.base64.item') and event == 'string':
                    text_index = type_indices.get('base64', 0)
                    text = self.batch_texts[text_index] if text_index < len(self.batch_texts) else None
                    yield StreamedEmbedding(
                        index=self.embeddings_yielded,
                        embedding=value,  # base64 string
                        embedding_type='base64',
                        text=text
                    )
                    self.embeddings_yielded += 1
                    type_indices['base64'] = text_index + 1
    
    def _iter_embeddings_fallback(self) -> Iterator[StreamedEmbedding]:
        """Fallback method using regular JSON parsing."""
        # Use buffered content if available, otherwise read from response
        if self._response_content is not None and isinstance(self._response_content, bytes):
            data = json.loads(self._response_content)
        elif hasattr(self.response, 'json') and callable(self.response.json):
            data = self.response.json()
        elif hasattr(self.response, '_response'):
            data = self.response._response.json()  # type: ignore
        else:
            raise ValueError("Response object does not have a json() method")

        response_type = data.get('response_type', '')
        texts = data.get('texts', self.batch_texts)

        if response_type == 'embeddings_floats':
            embeddings = data.get('embeddings', [])
            for i, embedding in enumerate(embeddings):
                yield StreamedEmbedding(
                    index=self.embeddings_yielded,
                    embedding=embedding,
                    embedding_type='float',
                    text=texts[i] if i < len(texts) else None
                )
                self.embeddings_yielded += 1

        elif response_type == 'embeddings_by_type':
            embeddings_obj = data.get('embeddings', {})

            # Iterate through each embedding type
            for emb_type, embeddings_list in embeddings_obj.items():
                type_name = emb_type.rstrip('_')
                if isinstance(embeddings_list, list):
                    for i, embedding in enumerate(embeddings_list):
                        yield StreamedEmbedding(
                            index=self.embeddings_yielded,
                            embedding=embedding,
                            embedding_type=type_name,
                            text=texts[i] if i < len(texts) else None
                        )
                        self.embeddings_yielded += 1
                        
                        
def stream_embed_response(response: httpx.Response, texts: List[str]) -> Iterator[StreamedEmbedding]:
    """
    Convenience function to stream embeddings from a response.
    
    Args:
        response: The httpx response containing embeddings
        texts: The original texts that were embedded
        
    Yields:
        StreamedEmbedding objects
    """
    parser = StreamingEmbedParser(response, texts)
    yield from parser.iter_embeddings()