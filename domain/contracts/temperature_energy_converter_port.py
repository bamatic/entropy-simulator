from abc import ABC, abstractmethod
from domain.entities import QuantumSystem

class TemperatureEnergyConverterPort(ABC):
    """
    Port for converting between temperature and energy.
    """
    @abstractmethod
    def get_equivalent_temperature(self, quantum_system: QuantumSystem, target_energy: float) -> float:
        """
        Given a target energy E, finds the temperature T such that <E>_canonical(T) = E.
        """
        pass

    @abstractmethod
    def get_equivalent_mean_energy(self, quantum_system: QuantumSystem, temperature: float) -> float:
        """
        Given a temperature T, calculates <E>_canonical(T).
        """
        pass
