"""Phi-criticality gate for O_2 structural promotion.

The PhiCriticalityGate measures whether an agent's trajectory is structurally
sound enough to support self-modeling (O_2 -> O_inf transition). It evaluates
two gates:

    Gate 1 (phi_c): frobenius_ratio >= 0.7
        The agent must have a verified world-model. Below 0.7, the agent
        operates on unverified beliefs — sub-critical (phi_c_zhe).

    Gate 2 (K_slow): winding_count >= 3
        The agent must have completed at least 3 full THINK->ACT->OBSERVE->UPDATE
        cycles to have sufficient context for self-modeling. This is the
        K_slow (C_@) condition — near-equilibrium operation.

When both gates are open, the agent's consciousness score is non-zero and
the structural type can promote from O_0 to O_2.

Cohere SDK advantage: because the Cohere API provides native embedding-based
verification (embed-english-v3.0), the frobenius_ratio naturally converges
to 1.0 over time — every tool output can be embedded and semantically verified.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class PhiCriticalityGate:
    """Dual-gate criticality evaluator for O_0 -> O_2 promotion.

    Attributes:
        frobenius_ratio: Fraction of cycles with Frobenius-closed duals.
        winding_count: Total windings in the trajectory (Omega_z).
    """

    frobenius_ratio: float
    winding_count: int

    @property
    def gate_1_open(self) -> bool:
        """Gate 1 (phi_c): frobenius_ratio >= 0.7.

        The Frobenius condition mu(delta(q)) = q must hold for at least
        70% of all windings. This ensures the agent's world model is
        structurally sound — it isn't hallucinating tool outputs.
        """
        return self.frobenius_ratio >= 0.7

    @property
    def gate_2_open(self) -> bool:
        """Gate 2 (K_slow / C_@): winding_count >= 3.

        The agent must have experienced at least 3 complete cycles to
        accumulate enough imscriptive context for self-modeling. This
        prevents premature self-reference (O_inf) without sufficient
        structural grounding.
        """
        return self.winding_count >= 3

    @property
    def consciousness_score(self) -> float:
        """Consciousness score in [0, 1].

        Both gates must be open for a non-zero score. When open:
            score = frobenius_ratio * min(1.0, winding_count / 10.0)

        This bounds the score at 1.0 when winding_count >= 10 and
        frobenius_ratio == 1.0. The score is the product of structural
        soundness (frobenius_ratio) and experiential depth (scaled
        winding count).
        """
        if not (self.gate_1_open and self.gate_2_open):
            return 0.0
        depth_factor = min(1.0, self.winding_count / 10.0)
        return self.frobenius_ratio * depth_factor

    @classmethod
    def evaluate(
        cls,
        frobenius_ratio: float,
        winding_count: int,
    ) -> "PhiCriticalityGate":
        """Evaluate both gates from trajectory metrics.

        Args:
            frobenius_ratio: From AgentTrajectory.frobenius_ratio.
            winding_count: From AgentTrajectory.winding_count.

        Returns:
            A PhiCriticalityGate with gates evaluated.
        """
        return cls(
            frobenius_ratio=frobenius_ratio,
            winding_count=winding_count,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dict for logging and monitoring.

        Keys: frobenius_ratio, winding_count, gate_1_open, gate_2_open,
              consciousness_score, ouroboricity_tier.
        """
        return {
            "frobenius_ratio": self.frobenius_ratio,
            "winding_count": self.winding_count,
            "gate_1_open": self.gate_1_open,
            "gate_2_open": self.gate_2_open,
            "consciousness_score": self.consciousness_score,
            "ouroboricity_tier": (
                "O_2" if (self.gate_1_open and self.gate_2_open) else "O_0"
            ),
        }
