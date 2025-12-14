from typing import Any, Dict, List
from domain.entities import QuantumSystem
from domain.services import EntropyComparisonService
from ..dtos import EntropyComparisonRequest, EntropyComparisonResponse

class CalculateEntropyComparisonUseCase:
    """
    Use case to calculate and compare the three entropies.
    """

    def __init__(self, entropy_comparison_service: EntropyComparisonService):
        self._service = entropy_comparison_service

    def execute(self, request: EntropyComparisonRequest) -> EntropyComparisonResponse:
        """
        Executes the use case.
        """
        quantum_system = QuantumSystem(n_qubits=request.n_qubits)

        if request.control_mode == "Energy":
            comparison = self._service.compare_entropies_for_energy(
                quantum_system, request.control_value
            )
        else:
            comparison = self._service.compare_entropies_for_temperature(
                quantum_system, request.control_value
            )

        s1 = comparison.boltzmann_entropy.value
        s2 = comparison.gibbs_entropy.value
        s3 = comparison.von_neumann_entropy.value

        diff_12 = abs(s2 - s1) / s1 if s1 > 0 else 0
        diff_23 = abs(s3 - s2) / s2 if s2 > 0 else 0
        max_diff = max(diff_12, diff_23) * 100

        return EntropyComparisonResponse(
            boltzmann_entropy=s1,
            gibbs_entropy=s2,
            von_neumann_entropy=s3,
            control_mode=comparison.control_mode,
            control_value=comparison.control_value,
            equivalent_temperature=comparison.equivalent_temperature,
            equivalent_energy=comparison.equivalent_energy,
            gibbs_distribution=comparison.gibbs_distribution.probabilities,
            von_neumann_eigenvalues=comparison.von_neumann_eigenvalues,
            degeneracy=comparison.degeneracy,
            mean_energy=comparison.mean_energy,
            std_dev_energy=comparison.std_dev_energy,
            quantum_coherences=comparison.quantum_coherences,
            max_diff=max_diff,
            boltzmann_label=f"S = k ln Ω(E)",
            gibbs_label=f"S = -k Σ P(E) ln P(E)",
            vn_label=f"S = -k Tr(ρ̂ ln ρ̂)",
            boltzmann_microstates=comparison.boltzmann_microstates,
            gibbs_microstates=comparison.gibbs_microstates,
            von_neumann_microstates=comparison.von_neumann_microstates,
        )
