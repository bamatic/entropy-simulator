from scipy.optimize import brentq
from domain.entities import QuantumSystem
from domain.contracts import (
    TemperatureEnergyConverterPort,
    CanonicalCalculatorPort,
)

class SciPyTemperatureEnergyConverter(TemperatureEnergyConverterPort):
    """
    SciPy-based implementation of the TemperatureEnergyConverterPort.
    """

    def __init__(self, canonical_calculator: CanonicalCalculatorPort):
        self._canonical_calculator = canonical_calculator

    def get_equivalent_temperature(
            self, quantum_system: QuantumSystem, target_energy: float
    ) -> float:
        """
        Finds the temperature T such that <E>_canonical(T) = target_energy.
        """
        N = quantum_system.n_qubits
        min_energy = -N
        max_energy = N

        # === VALIDATIONS ===

        # 1. Physical range
        if target_energy < min_energy or target_energy > max_energy:
            raise ValueError(
                f"target_energy={target_energy} outside physical range "
                f"[{min_energy}, {max_energy}] for N={N}"
            )

        # 2. Correct parity
        if (N - target_energy) % 2 != 0:
            valid_energies = list(range(min_energy, max_energy + 1, 2))
            raise ValueError(
                f"target_energy={target_energy} has incorrect parity. "
                f"Valid energies for N={N}: {valid_energies}"
            )

        # 3. Positive energies have no equivalent T > 0
        if target_energy > 0:
            raise ValueError(
                f"No positive temperature T > 0 exists such that ⟨E⟩ = {target_energy}. "
                f"In thermal equilibrium, lower energy states are favored, so ⟨E⟩ < 0 always. "
                f"Suggestion: Choose target_energy <= 0 for valid comparison."
            )

        # === SPECIAL CASES ===

        # Case 1: Exact ground state
        if abs(target_energy - min_energy) < 1e-9:
            return 0.0  # T → 0 K

        # Case 2: E = 0 (equiprobability)
        # This is only reached at T → ∞
        if abs(target_energy) < 1e-9:
            # Return a very high temperature as approximation
            # At high T, <E> approaches 0 asymptotically
            return 1000.0  # Effectively T → ∞

        # === NUMERICAL SEARCH ===

        def energy_difference(temp: float) -> float:
            """Objective function: <E>(T) - target_energy"""
            if temp <= 0:
                return min_energy - target_energy

            mean_e = self._canonical_calculator.calculate_mean_energy(
                quantum_system, temp
            )
            return mean_e - target_energy

        # Adaptive search range
        T_min = 0.001
        T_max = 10 * N

        # === CHECK IF TARGET IS ACCESSIBLE ===

        try:
            mean_e_min = self._canonical_calculator.calculate_mean_energy(
                quantum_system, T_min
            )
            mean_e_max = self._canonical_calculator.calculate_mean_energy(
                quantum_system, T_max
            )
        except Exception as e:
            raise ValueError(f"Cannot calculate energy bounds: {e}")

        # Check if target_energy is within accessible range
        # Add small tolerance for numerical errors
        tolerance = 0.01

        if target_energy < mean_e_min - tolerance:
            # Target is below accessible range (too close to ground state)
            raise ValueError(
                f"target_energy={target_energy} is below accessible range. "
                f"Minimum ⟨E⟩ at T={T_min}K is {mean_e_min:.3f}. "
                f"This energy is too close to the ground state."
            )

        if target_energy > mean_e_max + tolerance:
            # Target is above accessible range (would need T → ∞)
            raise ValueError(
                f"target_energy={target_energy} requires T → ∞ to reach. "
                f"Maximum ⟨E⟩ at T={T_max}K is {mean_e_max:.3f}. "
                f"For E ≈ 0, use a very high temperature as approximation."
            )

        # === ROOT FINDING ===

        try:
            temp, result = brentq(
                energy_difference,
                T_min,
                T_max,
                full_output=True,
                xtol=1e-6,
                maxiter=100
            )

            if result.converged:
                return float(temp)
            else:
                raise ValueError("Root finding did not converge")

        except ValueError as e:
            # Provide helpful error message
            raise ValueError(
                f"Could not find equivalent temperature for E={target_energy}. "
                f"Accessible range at T∈[{T_min}, {T_max}]: "
                f"⟨E⟩∈[{mean_e_min:.2f}, {mean_e_max:.2f}]. "
                f"Original error: {str(e)}"
            )

    def get_equivalent_mean_energy(
            self, quantum_system: QuantumSystem, temperature: float
    ) -> float:
        """
        Calculates the mean energy <E> of a canonical ensemble at given temperature.

        This is the inverse operation of get_equivalent_temperature:
        - get_equivalent_temperature: E → T (find T such that <E>(T) = E)
        - get_equivalent_mean_energy: T → <E> (calculate <E> at given T)

        Args:
            quantum_system: The quantum system of N qubits
            temperature: Temperature in Kelvin (must be > 0)

        Returns:
            Mean energy <E> in natural units (dimensionless)

        Raises:
            ValueError: If temperature is invalid (<= 0)

        Physical interpretation:
            - T → 0:    <E> → -N (ground state)
            - T → ∞:    <E> → 0  (equiprobability by symmetry)
            - T intermediate: <E> between -N and 0

        Example:
            >>> system = QuantumSystem(n_qubits=8)
            >>> converter = TemperatureEnergyConverter(...)
            >>> mean_e = converter.get_equivalent_mean_energy(system, temperature=2.0)
            >>> print(mean_e)  # Might print something like -3.5
        """
        # === VALIDATION ===

        if temperature <= 0:
            raise ValueError(
                f"Temperature must be positive. Got: {temperature} K. "
                "For T=0, the system is in the ground state with E = -N."
            )

        # === CALCULATION ===

        # Delegate to the canonical calculator which implements the physics
        mean_energy = self._canonical_calculator.calculate_mean_energy(
            quantum_system, temperature
        )

        # === SANITY CHECKS (optional but recommended) ===

        N = quantum_system.n_qubits
        min_energy = -N
        max_energy = N

        # Physical constraint: mean energy must be within bounds
        if mean_energy < min_energy or mean_energy > max_energy:
            raise RuntimeError(
                f"Calculated mean energy {mean_energy} is outside physical bounds "
                f"[{min_energy}, {max_energy}] for N={N}. This indicates a bug in "
                f"the canonical calculator implementation."
            )

        # For T > 0, mean energy should be negative (favors lower energy states)
        if mean_energy > 0.1:  # Small tolerance for numerical noise at high T
            import warnings
            warnings.warn(
                f"Mean energy {mean_energy} is positive at T={temperature} K. "
                f"This is unusual for systems with Hamiltonian H = Σ σ_z, "
                f"which favor negative energies at any finite temperature."
            )

        return float(mean_energy)