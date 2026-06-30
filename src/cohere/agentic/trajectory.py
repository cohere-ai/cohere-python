"""Agent trajectory — monotonic winding counter with structural health metrics.

The trajectory is the imscriptive context (D_omega) of the agent: it holds
the complete history of windings, never resets, and provides Frobenius health
metrics that gate the O_2 structural promotion.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from typing import Any, Optional

from .contracts import DualToolResult


@dataclass
class AgentCycle:
    """One full THINK -> ACT -> OBSERVE -> UPDATE winding.

    Attributes:
        winding: Monotonic integer, never reset (Omega_z invariant).
        timestamp: When this cycle began.
        action_name: The tool invoked (mu).
        action_input: Input to the action tool.
        dual_result: The DualToolResult from this cycle.
        update_note: Narrative of what was learned (OBSERVE + UPDATE).
        done: True if this cycle emitted the done() terminal.
        conclusion: The final conclusion, if done is True.
        frobenius_closed: Aggregated from dual_result.
    """

    winding: int
    action_name: str
    action_input: dict[str, Any]
    dual_result: DualToolResult
    update_note: str = ""
    done: bool = False
    conclusion: Optional[str] = None
    frobenius_closed: bool = False
    timestamp: datetime.datetime = field(default_factory=datetime.datetime.now)


class AgentTrajectory:
    """Monotonic, never-reset trajectory of agent cycles.

    The winding counter Omega_z is topological — it never decrements and
    never resets. This is the structural invariant that distinguishes O_2
    (topologically protected) from O_0 (no winding memory).

    The trajectory serves as the imscriptive context (D_omega) that the
    agent consults at every THINK phase. It is the agent's world model.
    """

    def __init__(self) -> None:
        self._cycles: list[AgentCycle] = []
        self._winding_counter: int = 0  # Omega_z: never reset

    @property
    def winding_count(self) -> int:
        """Total windings (Omega_z invariant). Monotonic, never reset."""
        return self._winding_counter

    @property
    def frobenius_ratio(self) -> float:
        """Fraction of cycles that are Frobenius-closed.

        A ratio of 1.0 means every action was verified (mu(delta(q)) = q
        for all cycles). This is the primary structural health metric for
        O_2 promotion. Below 0.7, Gate 1 (phi_c) opens — the agent is
        operating on unverified beliefs.
        """
        if not self._cycles:
            return 0.0
        closed = sum(1 for c in self._cycles if c.frobenius_closed)
        return closed / len(self._cycles)

    def append(self, cycle: AgentCycle) -> None:
        """Append one complete cycle, advancing the winding counter.

        Args:
            cycle: A fully formed AgentCycle. Its winding field is
                   overwritten to maintain Omega_z monotonicity.
        """
        self._winding_counter += 1
        cycle.winding = self._winding_counter
        self._cycles.append(cycle)

    def last(self, n: int = 1) -> list[AgentCycle]:
        """Return the last n cycles (most recent first).

        Args:
            n: Number of cycles to return.

        Returns:
            List of AgentCycle objects, most recent first.
        """
        return list(reversed(self._cycles[-n:]))

    def to_context(self) -> str:
        """Serialize the trajectory to an imscriptive context string.

        This is the context window fed to the agent at THINK time. It
        includes the summary of all prior windings in compressed form.
        """
        if not self._cycles:
            return "No prior windings."

        lines = [f"Trajectory: {self.winding_count} windings"]
        lines.append(f"Frobenius ratio: {self.frobenius_ratio:.3f}")
        lines.append(f"Cycles: {len(self._cycles)}")

        # Show last 20 cycles in detail, rest summarized
        tail = self._cycles[-20:]
        head_count = len(self._cycles) - len(tail)
        if head_count > 0:
            lines.append(f"[{head_count} earlier cycles summarized]")

        for cyc in tail:
            status = "✓" if cyc.frobenius_closed else "✗"
            done_tag = " [DONE]" if cyc.done else ""
            lines.append(
                f"  W{cyc.winding}: {status} {cyc.action_name}"
                f"{done_tag}: {cyc.update_note[:80]}"
            )

        return "\n".join(lines)

    def structural_health(self) -> dict[str, Any]:
        """Return a structural health report for O_2 promotion gating.

        Returns a dict with four keys:
            - winding_count: Omega_z integer
            - frobenius_ratio: float in [0, 1]
            - healthy: True iff frobenius_ratio >= 0.7
            - ouroboricity: "O_2" if healthy else "O_0"

        The threshold 0.7 is the minimum for Gate 1 (phi_c) opening.
        Above 0.7, the agent's world model is structurally sound enough
        to support self-modeling (the O_2 -> O_inf transition path).
        """
        ratio = self.frobenius_ratio
        healthy = ratio >= 0.7
        return {
            "winding_count": self.winding_count,
            "frobenius_ratio": ratio,
            "healthy": healthy,
            "ouroboricity": "O_2" if healthy else "O_0",
        }
