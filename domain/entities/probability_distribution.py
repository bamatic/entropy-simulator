from dataclasses import dataclass
from typing import Dict

@dataclass(frozen=True)
class ProbabilityDistribution:
    """Represents a discrete probability distribution over energy levels."""
    probabilities: Dict[float, float]

    def __post_init__(self):
        if not all(p >= 0 for p in self.probabilities.values()):
            raise ValueError("Probabilities must be non-negative.")
        if not abs(sum(self.probabilities.values()) - 1.0) < 1e-9:
            raise ValueError("Probabilities must sum to 1.")
