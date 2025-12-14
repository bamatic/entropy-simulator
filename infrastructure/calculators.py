import numpy as np
from scipy.special import comb
from scipy.linalg import expm, eigh
from typing import List, Dict

from domain.entities import (
    Entropy,
    ProbabilityDistribution,
    QuantumSystem,
    SpinConfiguration,
)
from domain.contracts import (
    MicrocanonicalCalculatorPort,
    CanonicalCalculatorPort,
    VonNeumannCalculatorPort,
)

class NumPyMicrocanonicalCalculator(MicrocanonicalCalculatorPort):
    """
    NumPy-based implementation of the MicrocanonicalCalculatorPort.
    """

    def get_possible_energies(self, n_qubits: int) -> List[float]:
        return [2 * m - n_qubits for m in range(n_qubits + 1)]

    def calculate_degeneracy(self, quantum_system: QuantumSystem, energy: float) -> int:
        n = quantum_system.n_qubits
        if (n - energy) % 2 != 0:
            return 0
        m = (n - energy) / 2
        if not (0 <= m <= n):
            return 0
        return int(comb(n, m, exact=True))

    def calculate_boltzmann_entropy(self, degeneracy: int) -> Entropy:
        if degeneracy == 0:
            return Entropy(value=0.0, type="Boltzmann")
        return Entropy(value=np.log(degeneracy), type="Boltzmann")

    def get_microcanonical_distribution(
        self, quantum_system: QuantumSystem, energy: float
    ) -> ProbabilityDistribution:
        probabilities = {e: 0.0 for e in self.get_possible_energies(quantum_system.n_qubits)}
        if energy in probabilities:
            probabilities[energy] = 1.0
        return ProbabilityDistribution(probabilities=probabilities)

    def _generate_all_microstates_data(self, n_qubits: int) -> List[SpinConfiguration]:
        """
        Generates all 2^N possible spin configurations with their properties.
        """
        all_microstates = []
        for i in range(2**n_qubits):
            binary_repr = bin(i)[2:].zfill(n_qubits)
            spins = [int(bit) for bit in binary_repr] # 0 for down, 1 for up
            num_spins_up = sum(spins)
            num_spins_down = n_qubits - num_spins_up
            energy = num_spins_up - num_spins_down # E = N_up - N_down

            all_microstates.append(
                SpinConfiguration(
                    index=i,
                    binary_representation=binary_repr,
                    energy=float(energy),
                    spins=spins,
                    probability=0.0, # Placeholder, to be filled by specific ensemble
                    num_spins_up=num_spins_up,
                    num_spins_down=num_spins_down,
                )
            )
        return all_microstates

    def get_microstates(self, quantum_system: QuantumSystem, energy: float) -> List[SpinConfiguration]:
        """
        Generates a list of SpinConfiguration objects for the given energy in the microcanonical ensemble.
        """
        all_microstates = self._generate_all_microstates_data(quantum_system.n_qubits)
        
        filtered_microstates = [ms for ms in all_microstates if ms.energy == energy]
        
        degeneracy = len(filtered_microstates)
        if degeneracy > 0:
            probability_per_state = 1.0 / degeneracy
            for ms in filtered_microstates:
                ms.probability = probability_per_state
        
        return filtered_microstates


class NumPyCanonicalCalculator(CanonicalCalculatorPort):
    """
    NumPy-based implementation of the CanonicalCalculatorPort.
    """

    def __init__(self, microcanonical_calculator: MicrocanonicalCalculatorPort):
        self._microcanonical_calculator = microcanonical_calculator

    def _get_degeneracies(self, quantum_system: QuantumSystem) -> Dict[float, int]:
        energies = self._microcanonical_calculator.get_possible_energies(quantum_system.n_qubits)
        return {e: self._microcanonical_calculator.calculate_degeneracy(quantum_system, e) for e in energies}

    def calculate_partition_function(self, quantum_system: QuantumSystem, temperature: float) -> float:
        if temperature <= 0:
            return 0.0
        beta = 1.0 / temperature
        degeneracies = self._get_degeneracies(quantum_system)
        return sum(g * np.exp(-beta * e) for e, g in degeneracies.items())

    def get_boltzmann_distribution(self, quantum_system: QuantumSystem, temperature: float) -> ProbabilityDistribution:
        if temperature <= 0:
            min_energy = -quantum_system.n_qubits
            return ProbabilityDistribution(probabilities={e: 1.0 if e == min_energy else 0.0 for e in self._get_degeneracies(quantum_system)})

        beta = 1.0 / temperature
        degeneracies = self._get_degeneracies(quantum_system)
        z = self.calculate_partition_function(quantum_system, temperature)
        if z == 0:
            return ProbabilityDistribution(probabilities={e: 0.0 for e in degeneracies})

        probabilities = {e: (g * np.exp(-beta * e)) / z for e, g in degeneracies.items()}
        return ProbabilityDistribution(probabilities=probabilities)

    def calculate_gibbs_entropy(self, distribution: ProbabilityDistribution) -> Entropy:
        s = -sum(p * np.log(p) for p in distribution.probabilities.values() if p > 0)
        return Entropy(value=s, type="Gibbs")

    def calculate_mean_energy(
            self, quantum_system: QuantumSystem, temperature: float
    ) -> float:
        """
        Calculates mean energy using log-space arithmetic for maximum stability.

        Args:
            quantum_system: The quantum system
            temperature: Temperature in Kelvin

        Returns:
            Mean energy in natural units
        """
        # Edge case
        if temperature <= 0:
            return float(-quantum_system.n_qubits)

        from scipy.special import logsumexp

        beta = 1.0 / temperature
        degeneracies = self._get_degeneracies(quantum_system)

        if not degeneracies:
            raise RuntimeError("No degeneracies found")

        # Extract energies and degeneracies
        energies = np.array(list(degeneracies.keys()))
        omegas = np.array(list(degeneracies.values()))

        # Filter out zero degeneracies
        mask = omegas > 0
        energies = energies[mask]
        omegas = omegas[mask]

        # Compute log-weights: log(Ω(E)) - β*E
        log_weights = np.log(omegas) - beta * energies

        # Compute log(Z) using logsumexp (numerically stable)
        log_Z = logsumexp(log_weights)

        # Compute probabilities: P(E) = exp(log_weight - log_Z)
        log_probs = log_weights - log_Z
        probs = np.exp(log_probs)

        # Mean energy: <E> = Σ E * P(E)
        mean_energy = np.sum(energies * probs)

        # Validation
        if not np.isfinite(mean_energy):
            raise RuntimeError(f"Mean energy is {mean_energy} at T={temperature}K")

        N = quantum_system.n_qubits
        if mean_energy < -N or mean_energy > N:
            raise RuntimeError(
                f"Mean energy {mean_energy} outside bounds [{-N}, {N}]"
            )

        return float(mean_energy)

    def calculate_energy_std_dev(self, quantum_system: QuantumSystem, temperature: float) -> float:
        if temperature <= 0:
            return 0.0
        beta = 1.0 / temperature
        degeneracies = self._get_degeneracies(quantum_system)
        z = self.calculate_partition_function(quantum_system, temperature)
        if z == 0:
            return 0.0
        mean_e = self.calculate_mean_energy(quantum_system, temperature)
        mean_e2 = sum((e**2) * g * np.exp(-beta * e) for e, g in degeneracies.items()) / z
        return np.sqrt(mean_e2 - mean_e**2)

    def get_microstates(self, quantum_system: QuantumSystem, temperature: float) -> List[SpinConfiguration]:
        """
        Generates a list of SpinConfiguration objects for the canonical ensemble.
        """
        all_microstates = self._microcanonical_calculator._generate_all_microstates_data(quantum_system.n_qubits)
        
        if temperature <= 0:
            min_energy = -quantum_system.n_qubits
            for ms in all_microstates:
                ms.probability = 1.0 if ms.energy == min_energy else 0.0
            return [ms for ms in all_microstates if ms.probability > 0]

        beta = 1.0 / temperature
        z = self.calculate_partition_function(quantum_system, temperature)
        
        if z == 0: # Should not happen with stable partition function, but for safety
            return []

        for ms in all_microstates:
            ms.probability = np.exp(-beta * ms.energy) / z
            
        return all_microstates


class NumPyVonNeumannCalculator(VonNeumannCalculatorPort):
    """
    NumPy-based implementation of von Neumann entropy calculations.

    Provides two entropy calculation methods:
    - calculate_von_neumann_entropy(eigenvalues): Direct calculation
    - calculate_von_neumann_entropy_physical(system, T): Grouped by energy (correct for comparison with Gibbs)
    """
    def __init__(self, microcanonical_calculator: MicrocanonicalCalculatorPort):
        self._microcanonical_calculator = microcanonical_calculator

    def _get_hamiltonian(self, n_qubits: int) -> np.ndarray:
        """Constructs H = Σᵢ σᵢᶻ diagonal in computational basis."""
        dim = 2 ** n_qubits
        h = np.zeros((dim, dim))

        for i in range(dim):
            num_ones = bin(i).count('1')
            energy = n_qubits - 2 * num_ones
            h[i, i] = energy

        return h

    def _get_density_matrix(
            self, quantum_system: QuantumSystem, temperature: float
    ) -> np.ndarray:
        """Calculates ρ = exp(-βH) / Z with numerical stability."""
        dim = 2 ** quantum_system.n_qubits

        if temperature <= 0:
            rho = np.zeros((dim, dim))
            ground_state_index = dim - 1 # Corresponds to binary '11...1' which has energy -N
            rho[ground_state_index, ground_state_index] = 1.0
            return rho

        beta = 1.0 / temperature
        h = self._get_hamiltonian(quantum_system.n_qubits)

        min_energy = np.min(np.diag(h))
        h_shifted = h - min_energy * np.eye(dim)

        rho_shifted = expm(-beta * h_shifted)
        trace = np.trace(rho_shifted)

        if trace == 0 or not np.isfinite(trace):
            rho = np.zeros((dim, dim))
            ground_state_index = dim - 1 # Corresponds to binary '11...1' which has energy -N
            rho[ground_state_index, ground_state_index] = 1.0
            return rho

        rho = rho_shifted / trace
        rho = (rho + rho.conj().T) / 2

        return rho

    def calculate_density_matrix_eigenvalues(
            self, quantum_system: QuantumSystem, temperature: float
    ) -> List[float]:
        """Calculates eigenvalues of thermal density matrix."""
        rho = self._get_density_matrix(quantum_system, temperature)
        eigenvalues = eigh(rho, eigvals_only=True)
        eigenvalues = [max(0.0, float(lam)) for lam in eigenvalues]
        return sorted(eigenvalues, reverse=True)

    def calculate_von_neumann_entropy_from_eignevalues(self, eigenvalues: List[float]) -> Entropy:
        """
        Calculates S = -Σ λᵢ ln λᵢ from eigenvalues.

        WARNING: Does NOT group by degeneracy.
        For comparison with Gibbs, use calculate_von_neumann_entropy_physical().
        """
        threshold = 1e-15
        s = 0.0
        for lam in eigenvalues:
            if lam > threshold:
                s -= lam * np.log(lam)
        s = max(0.0, float(s))
        return Entropy(value=s, type="VonNeumann")

    def calculate_von_neumann_entropy(
            self, quantum_system: QuantumSystem, temperature: float
    ) -> Entropy:
        """
        Calculates von Neumann entropy by grouping degenerate energy levels.

        This is the CORRECT method for comparing with Gibbs entropy.
        For diagonal H, gives S_vN = S_Gibbs.
        """
        rho = self._get_density_matrix(quantum_system, temperature)
        h = self._get_hamiltonian(quantum_system.n_qubits)

        energy_to_prob = {}
        for i in range(len(rho)):
            energy_rounded = round(float(h[i, i]), 10)
            prob_state = float(rho[i, i])
            if energy_rounded not in energy_to_prob:
                energy_to_prob[energy_rounded] = 0.0
            energy_to_prob[energy_rounded] += prob_state

        threshold = 1e-15
        s = 0.0
        for prob_energy in energy_to_prob.values():
            if prob_energy > threshold:
                s -= prob_energy * np.log(prob_energy)

        return Entropy(value=max(0.0, float(s)), type="VonNeumann")

    def calculate_quantum_coherences(
            self, quantum_system: QuantumSystem, temperature: float
    ) -> float:
        """Calculates Σᵢ≠ⱼ |ρᵢⱼ|² (always 0 for diagonal H)."""
        rho = self._get_density_matrix(quantum_system, temperature)
        dim = rho.shape[0]
        mask = ~np.eye(dim, dtype=bool)
        return float(np.sum(np.abs(rho[mask]) ** 2))

    def get_microstates(self, quantum_system: QuantumSystem, temperature: float) -> List[SpinConfiguration]:
        """
        Generates a list of SpinConfiguration objects for the quantum canonical ensemble.
        """
        all_microstates = self._microcanonical_calculator._generate_all_microstates_data(quantum_system.n_qubits)
        
        rho = self._get_density_matrix(quantum_system, temperature)
        
        for ms in all_microstates:
            # For diagonal Hamiltonian, the diagonal elements of rho are the probabilities
            ms.probability = float(rho[ms.index, ms.index])
            
        return all_microstates
