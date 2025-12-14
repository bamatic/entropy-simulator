from domain.entities import QuantumSystem, EntropyComparison
from domain.contracts import (
    MicrocanonicalCalculatorPort,
    CanonicalCalculatorPort,
    VonNeumannCalculatorPort,
    TemperatureEnergyConverterPort,
)

class EntropyComparisonService:
    """
    Domain service to orchestrate the calculation of the three entropies.
    """

    def __init__(
        self,
        microcanonical_calculator: MicrocanonicalCalculatorPort,
        canonical_calculator: CanonicalCalculatorPort,
        von_neumann_calculator: VonNeumannCalculatorPort,
        temp_energy_converter: TemperatureEnergyConverterPort,
    ):
        self._microcanonical_calculator = microcanonical_calculator
        self._canonical_calculator = canonical_calculator
        self._von_neumann_calculator = von_neumann_calculator
        self._temp_energy_converter = temp_energy_converter

    def compare_entropies_for_energy(
        self, quantum_system: QuantumSystem, energy: float
    ) -> EntropyComparison:
        """
        Calculates and compares the three entropies for a fixed energy.
        """
        degeneracy = self._microcanonical_calculator.calculate_degeneracy(
            quantum_system, energy
        )
        boltzmann_entropy = self._microcanonical_calculator.calculate_boltzmann_entropy(
            degeneracy
        )
        boltzmann_microstates = self._microcanonical_calculator.get_microstates(
            quantum_system, energy
        )

        equivalent_temperature = self._temp_energy_converter.get_equivalent_temperature(
            quantum_system, energy
        )

        gibbs_distribution = self._canonical_calculator.get_boltzmann_distribution(
            quantum_system, equivalent_temperature
        )
        gibbs_entropy = self._canonical_calculator.calculate_gibbs_entropy(
            gibbs_distribution
        )
        mean_energy = self._canonical_calculator.calculate_mean_energy(
            quantum_system, equivalent_temperature
        )
        std_dev_energy = self._canonical_calculator.calculate_energy_std_dev(
            quantum_system, equivalent_temperature
        )
        gibbs_microstates = self._canonical_calculator.get_microstates(
            quantum_system, equivalent_temperature
        )

        von_neumann_entropy = self._von_neumann_calculator.calculate_von_neumann_entropy(
            quantum_system, equivalent_temperature
        )
        
        vn_eigenvalues = self._von_neumann_calculator.calculate_density_matrix_eigenvalues(
            quantum_system, equivalent_temperature
        )
        von_neumann_microstates = self._von_neumann_calculator.get_microstates(
            quantum_system, equivalent_temperature
        )
        
        quantum_coherences = self._von_neumann_calculator.calculate_quantum_coherences(
            quantum_system, equivalent_temperature
        )

        return EntropyComparison(
            boltzmann_entropy=boltzmann_entropy,
            gibbs_entropy=gibbs_entropy,
            von_neumann_entropy=von_neumann_entropy,
            control_mode="Energy",
            control_value=energy,
            equivalent_temperature=equivalent_temperature,
            equivalent_energy=mean_energy,
            gibbs_distribution=gibbs_distribution,
            von_neumann_eigenvalues=vn_eigenvalues,
            degeneracy=degeneracy,
            mean_energy=mean_energy,
            std_dev_energy=std_dev_energy,
            quantum_coherences=quantum_coherences,
            boltzmann_microstates=boltzmann_microstates,
            gibbs_microstates=gibbs_microstates,
            von_neumann_microstates=von_neumann_microstates,
        )

    def compare_entropies_for_temperature(
        self, quantum_system: QuantumSystem, temperature: float
    ) -> EntropyComparison:
        """
        Calculates and compares the three entropies for a fixed temperature.
        """
        equivalent_energy = self._temp_energy_converter.get_equivalent_mean_energy(
            quantum_system, temperature
        )

        degeneracy = self._microcanonical_calculator.calculate_degeneracy(
            quantum_system, equivalent_energy
        )
        boltzmann_entropy = self._microcanonical_calculator.calculate_boltzmann_entropy(
            degeneracy
        )
        boltzmann_microstates = self._microcanonical_calculator.get_microstates(
            quantum_system, equivalent_energy
        )

        gibbs_distribution = self._canonical_calculator.get_boltzmann_distribution(
            quantum_system, temperature
        )
        gibbs_entropy = self._canonical_calculator.calculate_gibbs_entropy(
            gibbs_distribution
        )
        mean_energy = self._canonical_calculator.calculate_mean_energy(
            quantum_system, temperature
        )
        std_dev_energy = self._canonical_calculator.calculate_energy_std_dev(
            quantum_system, temperature
        )
        gibbs_microstates = self._canonical_calculator.get_microstates(
            quantum_system, temperature
        )

        von_neumann_entropy = self._von_neumann_calculator.calculate_von_neumann_entropy(
            quantum_system, temperature
        )

        vn_eigenvalues = self._von_neumann_calculator.calculate_density_matrix_eigenvalues(
            quantum_system, temperature
        )
        von_neumann_microstates = self._von_neumann_calculator.get_microstates(
            quantum_system, temperature
        )

        quantum_coherences = self._von_neumann_calculator.calculate_quantum_coherences(
            quantum_system, temperature
        )

        return EntropyComparison(
            boltzmann_entropy=boltzmann_entropy,
            gibbs_entropy=gibbs_entropy,
            von_neumann_entropy=von_neumann_entropy,
            control_mode="Temperature",
            control_value=temperature,
            equivalent_temperature=temperature,
            equivalent_energy=equivalent_energy,
            gibbs_distribution=gibbs_distribution,
            von_neumann_eigenvalues=vn_eigenvalues,
            degeneracy=degeneracy,
            mean_energy=mean_energy,
            std_dev_energy=std_dev_energy,
            quantum_coherences=quantum_coherences,
            boltzmann_microstates=boltzmann_microstates,
            gibbs_microstates=gibbs_microstates,
            von_neumann_microstates=von_neumann_microstates,
        )
