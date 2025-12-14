from domain.entities import QuantumSystem
from domain.contracts import DecoherenceCalculatorPort
from application.dtos import DecoherenceRequest, DecoherenceResponse

class CalculateDecoherenceUseCase:
    """
    Use case to calculate the temporal evolution of a quantum system.
    """

    def __init__(self, decoherence_calculator: DecoherenceCalculatorPort):
        self._calculator = decoherence_calculator

    def execute(self, request: DecoherenceRequest) -> DecoherenceResponse:
        """
        Executes the use case.
        """
        # For this simulation, N is fixed at 4
        quantum_system = QuantumSystem(n_qubits=4)

        # Validate bath temperature
        T_bath = request.T_bath
        if T_bath < 0.01:
            T_bath = 0.01

        result = self._calculator.calculate_temporal_evolution(
            quantum_system=quantum_system,
            bath_temperature=T_bath,
            t1=1.0,
            t2=0.5,
            t_max=10.0,
            num_points=100,
        )

        return DecoherenceResponse(
            times=result.times,
            s_vn_t=result.s_vn_t,
            s_gibbs=result.s_gibbs,
        )
