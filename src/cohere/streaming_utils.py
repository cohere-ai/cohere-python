"""Utilities for streaming large responses without loading everything into memory."""

from __future__ import annotations

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
        
    def iter_embeddings(self) -> Iterator[StreamedEmbedding]:
        """
        Iterate over embeddings one at a time without loading all into memory.
        
        Yields:
            StreamedEmbedding objects as they are parsed from the response
        """
        if not IJSON_AVAILABLE:
            # Fallback to regular parsing if ijson not available
            yield from self._iter_embeddings_fallback()
            return
            
        try:
            # Use ijson for memory-efficient parsing
            parser = ijson.parse(self.response.iter_bytes(chunk_size=65536))
            yield from self._parse_with_ijson(parser)
        except Exception:
            # If ijson parsing fails, fallback to regular parsing
            yield from self._iter_embeddings_fallback()
    
    def _parse_with_ijson(self, parser) -> Iterator[StreamedEmbedding]:
        """Parse embeddings using ijson incremental parser."""
        current_path: List[str] = []
        current_embedding = []
        embedding_index = 0
        embedding_type = "float"
        response_type = None
        in_embeddings = False
        
        for prefix, event, value in parser:
            # Track current path
            if event == 'map_key':
                if current_path and current_path[-1] == 'embeddings':
                    # This is an embedding type key (float_, int8, etc.)
                    embedding_type = value.rstrip('_')
            
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
                    text = self.batch_texts[embedding_index] if embedding_index < len(self.batch_texts) else None
                    yield StreamedEmbedding(
                        index=self.embeddings_yielded,
                        embedding=current_embedding,
                        embedding_type='float',
                        text=text
                    )
                    self.embeddings_yielded += 1
                    embedding_index += 1
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
                        text = self.batch_texts[embedding_index] if embedding_index < len(self.batch_texts) else None
                        yield StreamedEmbedding(
                            index=self.embeddings_yielded,
                            embedding=current_embedding,
                            embedding_type=type_name,
                            text=text
                        )
                        self.embeddings_yielded += 1
                        embedding_index += 1
                        current_embedding = []
                        
                # Handle base64 embeddings (string format)
                if prefix.startswith('embeddings.base64.item') and event == 'string':
                    text = self.batch_texts[embedding_index] if embedding_index < len(self.batch_texts) else None
                    yield StreamedEmbedding(
                        index=self.embeddings_yielded,
                        embedding=value,  # base64 string
                        embedding_type='base64',
                        text=text
                    )
                    self.embeddings_yielded += 1
                    embedding_index += 1
    
    def _iter_embeddings_fallback(self) -> Iterator[StreamedEmbedding]:
        """Fallback method using regular JSON parsing."""
        # This still loads the full response but at least provides the same interface
        data = self.response.json()
        response_type = data.get('response_type', '')
        
        if response_type == 'embeddings_floats':
            embeddings = data.get('embeddings', [])
            texts = data.get('texts', [])
            for i, embedding in enumerate(embeddings):
                yield StreamedEmbedding(
                    index=i,
                    embedding=embedding,
                    embedding_type='float',
                    text=texts[i] if i < len(texts) else None
                )
                
        elif response_type == 'embeddings_by_type':
            embeddings_obj = data.get('embeddings', {})
            texts = data.get('texts', [])
            
            # Iterate through each embedding type
            for emb_type, embeddings_list in embeddings_obj.items():
                type_name = emb_type.rstrip('_')
                if isinstance(embeddings_list, list):
                    for i, embedding in enumerate(embeddings_list):
                        yield StreamedEmbedding(
                            index=i,
                            embedding=embedding,
                            embedding_type=type_name,
                            text=texts[i] if i < len(texts) else None
                        )
                        
                        
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