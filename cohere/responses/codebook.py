from typing import Any, Dict, Iterator, List, Optional

from cohere.responses.base import CohereObject


class Codebook(CohereObject):
    def __init__(self, codebook: List[List[List[float]]], meta: Optional[Dict[str, Any]] = None) -> None:
        self.codebook = codebook
        self.meta = meta

    def __iter__(self) -> Iterator:
        return iter(self.codebook)

    def __len__(self) -> int:
        return len(self.codebook)
