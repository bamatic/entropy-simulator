import pytest
from entropy_unification_simulator.domain.entities import QuantumSystem, ProbabilityDistribution, Entropy

def test_quantum_system_valid():
    qs = QuantumSystem(n_qubits=4)
    assert qs.n_qubits == 4

def test_quantum_system_invalid_min():
    with pytest.raises(ValueError):
        QuantumSystem(n_qubits=1)

def test_quantum_system_invalid_max():
    with pytest.raises(ValueError):
        QuantumSystem(n_qubits=11)

def test_probability_distribution_valid():
    pd = ProbabilityDistribution(probabilities={'a': 0.5, 'b': 0.5})
    assert pd.probabilities['a'] == 0.5

def test_probability_distribution_invalid_sum():
    with pytest.raises(ValueError):
        ProbabilityDistribution(probabilities={'a': 0.5, 'b': 0.6})

def test_probability_distribution_invalid_negative():
    with pytest.raises(ValueError):
        ProbabilityDistribution(probabilities={'a': -0.5, 'b': 1.5})

def test_entropy_valid():
    e = Entropy(value=1.0, type="Boltzmann")
    assert e.value == 1.0

def test_entropy_invalid_negative():
    with pytest.raises(ValueError):
        Entropy(value=-1.0, type="Boltzmann")
