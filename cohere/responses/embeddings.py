from typing import Any, Dict, Iterator, List, Optional

from cohere.responses.base import CohereObject


class Embeddings(CohereObject):
    def __init__(self, embeddings: List[List[float]], meta: Optional[Dict[str, Any]] = None) -> None:
        self.embeddings = embeddings
        self.meta = meta

    def __iter__(self) -> Iterator:
        return iter(self.embeddings)

    def __len__(self) -> int:
        return len(self.embeddings)
