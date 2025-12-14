import numpy as np
from scipy.linalg import expm
from scipy.integrate import solve_ivp
from typing import List, Tuple

from domain.contracts import DecoherenceCalculatorPort
from domain.entities import QuantumSystem, TemporalEvolutionResult

class ScipyDecoherenceCalculator(DecoherenceCalculatorPort):
    """
    SciPy-based implementation of the DecoherenceCalculatorPort.
    """

    def calculate_temporal_evolution(
        self,
        quantum_system: QuantumSystem,
        bath_temperature: float,
        t1: float = 1.0,
        t2: float = 0.5,
        t_max: float = 10.0,
        num_points: int = 100,
    ) -> TemporalEvolutionResult:
        """
        Simulate temporal evolution of a quantum system under decoherence.
        """
        dim = 2**quantum_system.n_qubits

        # 1. Build Hamiltonian and Lindblad operators
        h, lindblad_ops = self._build_lindblad_operators(quantum_system.n_qubits, bath_temperature, t1, t2)

        # 2. Initial state: Ground state |1111⟩ (index 2^N - 1)
        rho0 = np.zeros((dim, dim), dtype=complex)
        ground_state_index = dim - 1
        rho0[ground_state_index, ground_state_index] = 1.0

        # 3. Evolve according to Lindblad
        times, rhos = self._lindblad_evolution(rho0, h, lindblad_ops, t_max, num_points)

        # 4. Calculate von Neumann entropy at each step
        s_vn_t = [self._calculate_von_neumann_entropy_from_rho(rho) for rho in rhos]

        # 5. Calculate Gibbs entropy (equilibrium, constant)
        s_gibbs = self._calculate_gibbs_entropy_at_temperature(h, bath_temperature)

        return TemporalEvolutionResult(
            times=times,
            s_vn_t=np.array(s_vn_t),
            s_gibbs=s_gibbs,
        )

    def _build_lindblad_operators(
        self, n_qubits: int, T_bath: float, T1: float, T2: float
    ) -> Tuple[np.ndarray, List[Tuple[float, np.ndarray]]]:
        """
        Build Hamiltonian and Lindblad operators for N qubits.
        """
        dim = 2**n_qubits

        # Build Hamiltonian H = Σ σᶻ (diagonal)
        H = np.zeros((dim, dim))
        for i in range(dim):
            num_ones = bin(i).count('1')
            energy = n_qubits - 2 * num_ones
            H[i, i] = energy

        # Single-qubit Pauli operators
        sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
        sigma_minus = np.array([[0, 1], [0, 0]], dtype=complex)
        sigma_plus = np.array([[0, 0], [1, 0]], dtype=complex)

        # Helper function: tensor product at specific position
        def single_qubit_operator(operator_1q: np.ndarray, qubit_index: int) -> np.ndarray:
            """Apply 1-qubit operator at position qubit_index."""
            identity = np.eye(2, dtype=complex)
            result = np.array([[1]], dtype=complex)

            for i in range(n_qubits):
                if i == qubit_index:
                    result = np.kron(result, operator_1q)
                else:
                    result = np.kron(result, identity)

            return result

        # Decay rates
        gamma_dephase = 1.0 / T2
        gamma_decay = 1.0 / T1

        # Thermal population (Bose-Einstein occupation number)
        delta_E = 2.0
        k_B = 1.0

        if T_bath > 0:
            n_thermal = 1.0 / (np.exp(delta_E / T_bath) - 1.0)
        else:
            n_thermal = 0.0

        # Build list of Lindblad operators
        lindblad_ops = []

        for qubit_idx in range(n_qubits):
            # Dephasing
            L_dephase = single_qubit_operator(sigma_z, qubit_idx)
            lindblad_ops.append((gamma_dephase, L_dephase))

            # Decay (lower spin)
            L_decay = single_qubit_operator(sigma_minus, qubit_idx)
            lindblad_ops.append((gamma_decay * (n_thermal + 1), L_decay))

            # Excitation (raise spin)
            L_excite = single_qubit_operator(sigma_plus, qubit_idx)
            lindblad_ops.append((gamma_decay * n_thermal, L_excite))

        return H, lindblad_ops

    def _lindblad_evolution(
        self,
        rho0: np.ndarray,
        H: np.ndarray,
        lindblad_ops: List[Tuple[float, np.ndarray]],
        t_max: float,
        num_points: int,
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Solve the Lindblad master equation using solve_ivp for complex systems.
        """
        dim = rho0.shape[0]
        times = np.linspace(0, t_max, num_points)

        def lindblad_rhs(t, rho_vec):
            """Right-hand side of master equation in vectorized form."""
            rho = rho_vec.reshape((dim, dim))
            commutator = -1j * (H @ rho - rho @ H)
            lindblad_term = np.zeros_like(rho, dtype=complex)

            for gamma, L in lindblad_ops:
                L_dag = L.conj().T
                L_dag_L = L_dag @ L
                lindblad_term += gamma * (
                    L @ rho @ L_dag - 0.5 * (L_dag_L @ rho + rho @ L_dag_L)
                )

            drho_dt = commutator + lindblad_term
            return drho_dt.flatten()

        rho0_vec = rho0.flatten()
        
        # Use solve_ivp which handles complex values natively
        sol = solve_ivp(
            lindblad_rhs,
            (0, t_max),
            rho0_vec,
            t_eval=times,
            method='RK45' # A good general-purpose solver
        )

        # Transpose the solution to get states at each time point
        rho_vec_t = sol.y.T
        rhos = [rho_vec.reshape((dim, dim)) for rho_vec in rho_vec_t]

        return times, rhos

    def _calculate_von_neumann_entropy_from_rho(self, rho: np.ndarray) -> float:
        """
        Calculate S_vN = -Tr(ρ ln ρ) for a given density matrix.
        """
        eigenvalues = np.linalg.eigvalsh(rho)
        eigenvalues = eigenvalues[eigenvalues > 1e-15]
        entropy = -np.sum(eigenvalues * np.log(eigenvalues))
        return float(entropy)

    def _calculate_gibbs_entropy_at_temperature(self, H: np.ndarray, T: float) -> float:
        """
        Calculate Gibbs entropy for canonical ensemble at temperature T.
        """
        if T <= 0:
            return 0.0

        beta = 1.0 / T
        energies = np.diag(H)
        Z = np.sum(np.exp(-beta * energies))
        E_mean = np.sum(energies * np.exp(-beta * energies)) / Z
        S_gibbs = np.log(Z) + beta * E_mean
        return float(S_gibbs)
