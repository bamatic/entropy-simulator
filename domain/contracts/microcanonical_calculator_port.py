from abc import ABC, abstractmethod
from typing import List
from domain.entities import ProbabilityDistribution, Entropy, QuantumSystem, SpinConfiguration

class MicrocanonicalCalculatorPort(ABC):
    """
    Port for calculating properties of a microcanonical ensemble.
    """
    @abstractmethod
    def calculate_degeneracy(self, quantum_system: QuantumSystem, energy: float) -> int:
        """
        Calculates the degeneracy (number of microstates) for a given energy.
        """
        pass

    @abstractmethod
    def calculate_boltzmann_entropy(self, degeneracy: int) -> Entropy:
        """
        Calculates the Boltzmann entropy from the degeneracy.
        """
        pass

    @abstractmethod
    def get_microcanonical_distribution(self, quantum_system: QuantumSystem, energy: float) -> ProbabilityDistribution:
        """
        Generates the probability distribution for a microcanonical ensemble (delta function).
        """
        pass

    @abstractmethod
    def get_microstates(self, quantum_system: QuantumSystem, energy: float) -> List[SpinConfiguration]:
        """
        Generates a list of SpinConfiguration objects for the given energy in the microcanonical ensemble.
        """
        pass
