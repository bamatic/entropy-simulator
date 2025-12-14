from abc import ABC, abstractmethod
from domain.entities import QuantumSystem, TemporalEvolutionResult

class DecoherenceCalculatorPort(ABC):
    """
    Port for calculating the temporal evolution of a quantum system.
    """
    @abstractmethod
    def calculate_temporal_evolution(
        self,
        quantum_system: QuantumSystem,
        bath_temperature: float,
    ) -> TemporalEvolutionResult:
        """
        Calculates the time evolution of von Neumann entropy and the equilibrium Gibbs entropy.
        """
        pass
