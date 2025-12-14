from dataclasses import dataclass

@dataclass(frozen=True)
class QuantumSystem:
    """Represents a system of N qubits."""
    n_qubits: int

    def __post_init__(self):
        if not 1 <= self.n_qubits <= 10:
            raise ValueError("Number of qubits must be between 1 and 10.")
