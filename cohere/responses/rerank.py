from typing import Any, Dict, Iterator, List, NamedTuple, Optional

from cohere.responses.base import CohereObject

RerankDocument = NamedTuple("Document", [("text", str)])
RerankDocument.__doc__ = """
Returned by co.rerank,
dict which always contains text but can also contain arbitrary fields
"""


class RerankSnippet(NamedTuple("Snippet", [("text", str), ("start_index", int)])):
    """
    Returned by co.rerank,
    object which contains `text` and `start_index`
    """

    def __repr__(self) -> str:
        return f"RerankSnippet<text: {self.text}, start_index: {self.start_index}>"


class RerankResult(CohereObject):
    def __init__(
        self,
        document: Dict[str, Any] = None,
        index: int = None,
        relevance_score: float = None,
        snippets: List[RerankSnippet] = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.document = document
        self.snippets = snippets
        self.index = index
        self.relevance_score = relevance_score

    def __repr__(self) -> str:
        score = self.relevance_score
        index = self.index
        document_repr = ""
        if self.document is not None:
            document_repr = f", document['text']: {self.document['text']}"

        snippet_repr = ""
        if self.snippets is not None:
            snippet_repr = f", snippets: {self.snippets}"

        return f"RerankResult<index: {index}, relevance_score: {score}{document_repr}{snippet_repr}>"


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
            document = res.get("document")

            if res.get("snippets") is not None:
                snippets = [
                    RerankSnippet(text=snippet["text"], start_index=snippet["start_index"])
                    for snippet in res["snippets"]
                ]
            else:
                snippets = None

            results.append(
                RerankResult(
                    document=document, index=res["index"], relevance_score=res["relevance_score"], snippets=snippets
                )
            )
        return results

    def __str__(self) -> str:
        return str(self.results)

    def __repr__(self) -> str:
        return self.results.__repr__()

    def __iter__(self) -> Iterator:
        return iter(self.results)

    def __getitem__(self, index) -> RerankResult:
        return self.results[index]
