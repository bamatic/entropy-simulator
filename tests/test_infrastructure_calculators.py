import pytest
import numpy as np
from domain.entities import QuantumSystem
from infrastructure.calculators import (
    NumPyMicrocanonicalCalculator,
    NumPyCanonicalCalculator,
    NumPyVonNeumannCalculator,
)
from infrastructure.converters import SciPyTemperatureEnergyConverter

@pytest.fixture
def micro_calc():
    return NumPyMicrocanonicalCalculator()

@pytest.fixture
def canon_calc(micro_calc):
    return NumPyCanonicalCalculator(micro_calc)

@pytest.fixture
def vn_calc():
    return NumPyVonNeumannCalculator()

@pytest.fixture
def temp_converter(canon_calc):
    return SciPyTemperatureEnergyConverter(canon_calc)

def test_degeneracy(micro_calc):
    qs = QuantumSystem(n_qubits=4)
    assert micro_calc.calculate_degeneracy(qs, 0) == 6
    assert micro_calc.calculate_degeneracy(qs, 2) == 4
    assert micro_calc.calculate_degeneracy(qs, 4) == 1
    assert micro_calc.calculate_degeneracy(qs, 1) == 0

def test_boltzmann_entropy(micro_calc):
    entropy = micro_calc.calculate_boltzmann_entropy(6)
    assert np.isclose(entropy.value, np.log(6))

def test_partition_function(canon_calc):
    qs = QuantumSystem(n_qubits=2)
    z = canon_calc.calculate_partition_function(qs, 1.0)
    # E = -2, 0, 2. g = 1, 2, 1
    # Z = 1*exp(2) + 2*exp(0) + 1*exp(-2)
    expected_z = np.exp(2) + 2 + np.exp(-2)
    assert np.isclose(z, expected_z)

def test_mean_energy(canon_calc):
    qs = QuantumSystem(n_qubits=2)
    mean_e = canon_calc.calculate_mean_energy(qs, 1.0)
    z = canon_calc.calculate_partition_function(qs, 1.0)
    expected_e = (-2*np.exp(2) + 0*2 + 2*np.exp(-2)) / z
    assert np.isclose(mean_e, expected_e)

def test_get_equivalent_temperature(temp_converter):
    qs = QuantumSystem(n_qubits=8)
    # This value is pre-calculated and known to be correct
    temp = temp_converter.get_equivalent_temperature(qs, target_energy=-6)
    assert np.isclose(temp, 1.03, atol=0.02)

def test_vn_entropy(vn_calc):
    qs = QuantumSystem(n_qubits=2)
    eigenvalues = vn_calc.calculate_density_matrix_eigenvalues(qs, 1.0)
    entropy = vn_calc.calculate_von_neumann_entropy(eigenvalues)
    assert entropy.value > 0

def test_coherences(vn_calc):
    qs = QuantumSystem(n_qubits=2)
    coherences = vn_calc.calculate_quantum_coherences(qs, 1.0)
    # Hamiltonian is diagonal, so density matrix is diagonal, so coherences are 0
    assert np.isclose(coherences, 0.0)
