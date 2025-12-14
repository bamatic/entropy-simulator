from abc import ABC, abstractmethod
from typing import List
from domain.entities import Entropy, QuantumSystem, SpinConfiguration

class VonNeumannCalculatorPort(ABC):
    """
    Port for calculating properties of a quantum canonical ensemble.
    """
    @abstractmethod
    def calculate_density_matrix_eigenvalues(self, quantum_system: QuantumSystem, temperature: float) -> List[float]:
        """
        Calculates the eigenvalues of the density matrix.
        """
        pass

    @abstractmethod
    def calculate_von_neumann_entropy(self, quantum_system: QuantumSystem, temperature: float) -> Entropy:
        """
        Calculates the von Neumann entropy from the density matrix for a given quantum system and temperature.
        """
        pass

    @abstractmethod
    def calculate_quantum_coherences(self, quantum_system: QuantumSystem, temperature: float) -> float:
        """
        Calculates the sum of squared off-diagonal elements of the density matrix.
        """
        pass

    @abstractmethod
    def get_microstates(self, quantum_system: QuantumSystem, temperature: float) -> List[SpinConfiguration]:
        """
        Generates a list of SpinConfiguration objects for the quantum canonical ensemble.
        """
        pass
