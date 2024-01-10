from typing import Any, Dict, Iterator, List, Optional, Union

from cohere.responses.base import CohereObject

EMBEDDINGS_FLOATS_RESPONSE_TYPE = "embeddings_floats"
EMBEDDINGS_BY_TYPE_RESPONSE_TYPE = "embeddings_by_type"


class EmbeddingsByType(CohereObject):
    def __init__(
        self,
        float: Optional[List[List[float]]] = None,
        int8: Optional[List[List[int]]] = None,
        uint8: Optional[List[List[int]]] = None,
        binary: Optional[List[List[int]]] = None,
        ubinary: Optional[List[List[int]]] = None,
    ) -> None:
        self.float = float
        self.int8 = int8
        self.uint8 = uint8
        self.binary = binary
        self.ubinary = ubinary


class Embeddings(CohereObject):
    def __init__(
        self,
        embeddings: Union[List[List[float]], EmbeddingsByType],
        response_type: str,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.response_type = response_type
        self.embeddings = embeddings
        self.meta = meta

    def __iter__(self) -> Iterator:
        return iter(self.embeddings)

    def __len__(self) -> int:
        return len(self.embeddings)


class EmbeddingResponses:
    def __init__(
        self,
    ) -> None:
        self.response_type = None
        self.embeddings_floats = []
        self.embeddings_by_type = {}

    def add_response(self, response):
        self.response_type = response["response_type"]
        if self.response_type == EMBEDDINGS_FLOATS_RESPONSE_TYPE:
            self.embeddings_floats.extend(response["embeddings"])
        elif self.response_type == EMBEDDINGS_BY_TYPE_RESPONSE_TYPE:
            for k, v in response["embeddings"].items():
                if k not in self.embeddings_by_type:
                    self.embeddings_by_type[k] = []
                self.embeddings_by_type[k].extend(v)

    def get_embeddings(self) -> Union[List[List[float]], EmbeddingsByType]:
        if self.response_type == EMBEDDINGS_FLOATS_RESPONSE_TYPE:
            return self.embeddings_floats
        elif self.response_type == EMBEDDINGS_BY_TYPE_RESPONSE_TYPE:
            return EmbeddingsByType(**self.embeddings_by_type)
