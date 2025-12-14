from dataclasses import dataclass

@dataclass(frozen=True)
class Entropy:
    """Represents an entropy value with its type."""
    value: float
    type: str  # "Boltzmann", "Gibbs", "VonNeumann"

    def __post_init__(self):
        if self.value < 0:
            raise ValueError("Entropy must be non-negative.")
