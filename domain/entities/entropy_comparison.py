from dataclasses import dataclass
from typing import List
from .entropy import Entropy
from .probability_distribution import ProbabilityDistribution
from .spin_configuration import SpinConfiguration # New import

@dataclass(frozen=True)
class EntropyComparison:
    """Aggregates three entropy values for comparison."""
    boltzmann_entropy: Entropy
    gibbs_entropy: Entropy
    von_neumann_entropy: Entropy
    control_mode: str  # "Energy" or "Temperature"
    control_value: float
    equivalent_temperature: float
    equivalent_energy: float
    gibbs_distribution: ProbabilityDistribution
    von_neumann_eigenvalues: List[float]
    degeneracy: int
    mean_energy: float
    std_dev_energy: float
    quantum_coherences: float
    boltzmann_microstates: List[SpinConfiguration] # New field
    gibbs_microstates: List[SpinConfiguration]     # New field
    von_neumann_microstates: List[SpinConfiguration] # New field
