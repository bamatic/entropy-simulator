from dataclasses import dataclass
from typing import List

@dataclass
class SpinConfiguration:
    """
    Represents a single microstate (spin configuration) of the quantum system.
    """
    index: int  # Decimal index of the configuration (e.g., 0 to 2^N - 1)
    binary_representation: str # Binary string (e.g., "1110" for N=4)
    energy: float
    spins: List[int]  # List of 0s (down) and 1s (up)
    probability: float # Probability P(E) or eigenvalue λᵢ
    num_spins_up: int
    num_spins_down: int
