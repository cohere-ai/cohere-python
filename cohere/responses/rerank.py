from typing import Any, Dict, Iterator, List, NamedTuple, Optional

from cohere.responses.base import CohereObject

RerankDocument = NamedTuple("Document", [("text", str)])
RerankDocument.__doc__ = """
Returned by co.rerank,
dict which always contains text but can also contain arbitrary fields
"""


class RerankResult(CohereObject):
    def __init__(
        self, document: Dict[str, Any] = None, index: int = None, relevance_score: float = None, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.document = document
        self.index = index
        self.relevance_score = relevance_score

    def __repr__(self) -> str:
        score = self.relevance_score
        index = self.index
        if self.document is None:
            return f"RerankResult<index: {index}, relevance_score: {score}>"
        else:
            text = self.document["text"]
            return f"RerankResult<document['text']: {text}, index: {index}, relevance_score: {score}>"


class Reranking(CohereObject):
    def __init__(
        self, response: Optional[Dict[str, Any]] = None, meta: Optional[Dict[str, Any]] = None, **kwargs
    ) -> None:
        super().__init__(**kwargs, id=response.get("id"))
        assert response is not None
        self.results = self._results(response)
        self.meta = response["meta"]

    def _results(self, response: Dict[str, Any]) -> List[RerankResult]:
        results = []
        for res in response["results"]:
            if "document" in res.keys():
                results.append(RerankResult(res["document"], res["index"], res["relevance_score"]))
            else:
                results.append(RerankResult(index=res["index"], relevance_score=res["relevance_score"]))
        return results

    def __str__(self) -> str:
        return str(self.results)

    def __repr__(self) -> str:
        return self.results.__repr__()

    def __iter__(self) -> Iterator:
        return iter(self.results)

    def __getitem__(self, index) -> RerankResult:
        return self.results[index]
