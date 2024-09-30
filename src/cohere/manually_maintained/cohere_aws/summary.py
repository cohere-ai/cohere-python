from .error import CohereError
from .response import CohereObject
from typing import Any, Dict, Optional


class Summary(CohereObject):
    def __init__(self,
                 response: Optional[Dict[str, Any]] = None) -> None:
        assert response is not None
        if not response["summary"]:
            raise CohereError("Response lacks a summary")

        self.result = response["summary"]

    def __str__(self) -> str:
        return self.result
