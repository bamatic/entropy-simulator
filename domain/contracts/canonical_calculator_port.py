from abc import ABC, abstractmethod
from typing import List
from domain.entities import ProbabilityDistribution, Entropy, QuantumSystem, SpinConfiguration

class CanonicalCalculatorPort(ABC):
    """
    Port for calculating properties of a canonical ensemble (classical).
    """
    @abstractmethod
    def calculate_partition_function(self, quantum_system: QuantumSystem, temperature: float) -> float:
        """
        Calculates the partition function Z for a given temperature.
        """
        pass

    @abstractmethod
    def get_boltzmann_distribution(self, quantum_system: QuantumSystem, temperature: float) -> ProbabilityDistribution:
        """
        Generates the Boltzmann probability distribution P(E) for a given temperature.
        """
        pass

    @abstractmethod
    def calculate_gibbs_entropy(self, distribution: ProbabilityDistribution) -> Entropy:
        """
        Calculates the Gibbs entropy from a probability distribution.
        """
        pass

    @abstractmethod
    def calculate_mean_energy(self, quantum_system: QuantumSystem, temperature: float) -> float:
        """
        Calculates the mean energy <E> for a given temperature.
        """
        pass

    @abstractmethod
    def calculate_energy_std_dev(self, quantum_system: QuantumSystem, temperature: float) -> float:
        """
        Calculates the standard deviation of energy for a given temperature.
        """
        pass

    @abstractmethod
    def get_microstates(self, quantum_system: QuantumSystem, temperature: float) -> List[SpinConfiguration]:
        """
        Generates a list of SpinConfiguration objects for the canonical ensemble.
        """
        pass
