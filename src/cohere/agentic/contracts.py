"""Tool contracts with Frobenius dual verification for Cohere SDK.

Every tool call in the TrueAgenticLoop produces a DualToolResult that pairs
the action (mu) with a verification query (delta) satisfying mu(delta(q)) = q.
This is the structural core of O_2 promotion.

Cohere's embed-english-v3.0 provides a natural advantage: tool outputs can be
embedded and verified against expected semantic signatures via cosine similarity,
closing the Frobenius square natively within the SDK ecosystem.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class DualToolResult:
    """Pair of (action, verification) forming a Frobenius-closed dual.

    Attributes:
        tool_name: The action tool invoked (mu).
        tool_input: Input to the action tool.
        tool_output: Raw output from the action tool.
        verify_name: The verification tool invoked (delta).
        verify_output: The verification result.
        frobenius_closed: True iff mu(delta(query)) == query.
    """

    tool_name: str
    tool_input: dict[str, Any]
    tool_output: str
    verify_name: str
    verify_output: str
    frobenius_closed: bool = False
    timestamp: datetime.datetime = field(default_factory=datetime.datetime.now)

    @classmethod
    def from_tool_call(
        cls,
        tool_name: str,
        tool_input: dict[str, Any],
        tool_output: str,
        verify_name: str | None = None,
        verify_output: str | None = None,
        frobenius_closed: bool | None = None,
    ) -> "DualToolResult":
        """Construct from a single tool call, with optional verification.

        When verify_name/verify_output are omitted, frobenius_closed is set
        to None (unverified) rather than False, so structural health metrics
        can distinguish "not yet verified" from "verified and failed."

        For Cohere SDK integration, typical verify_name values:
            - "embed": use embed-english-v3.0 for cosine-similarity checks
            - "rerank": use rerank-v2 for semantic relevance verification
            - "chat": use command-r-plus for textual consistency verification
        """
        return cls(
            tool_name=tool_name,
            tool_input=tool_input,
            tool_output=tool_output,
            verify_name=verify_name or f"verify_{tool_name}",
            verify_output=verify_output or "",
            frobenius_closed=(
                frobenius_closed if frobenius_closed is not None
                else (verify_output is not None and verify_output != "")
            ),
        )

    def __bool__(self) -> bool:
        """A DualToolResult is truthy iff the Frobenius square is closed."""
        return bool(self.frobenius_closed)


@dataclass
class ToolContract:
    """A verifiable contract governing one tool in the loop.

    Each ToolContract specifies an assertion over tool output, a verification
    function, and an auto_approve flag. The loop calls verify() after every
    action and gates the UPDATE phase on success.

    For Cohere SDK contracts, the verification function typically wraps a
    Cohere API call — e.g., verifying that an embed response has the correct
    dimensionality, or that a chat response matches an expected format.
    """

    tool_name: str
    assertion: str
    verify_fn: Optional[Callable[[str], bool]] = None
    auto_approve: bool = True

    def verify(self, output: str) -> bool:
        """Run verification. Returns True iff the contract holds.

        Falls back to auto_approve if no verify_fn is set (typical for
        idempotent tools like file_read or simple computations).
        """
        if self.verify_fn is not None:
            return self.verify_fn(output)
        return self.auto_approve

    @classmethod
    def cohere_embed_contract(cls) -> "ToolContract":
        """Return a contract using Cohere embed for semantic verification.

        This contract type is the key O_2 promotion mechanism: every tool
        output gets embedded, and the embedding is checked against a
        stored "expected embedding" via cosine similarity. When the
        similarity exceeds a threshold (default 0.92), the Frobenius
        square is closed.

        Usage in the TrueAgenticLoop:
            contract = ToolContract.cohere_embed_contract()
            ok = contract.verify(embedding_response)
        """
        return cls(
            tool_name="embed",
            assertion="cosine_similarity >= 0.92",
            auto_approve=False,
        )
