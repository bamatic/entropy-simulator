from dataclasses import dataclass
from typing import Dict, List
import numpy as np
from domain.entities import SpinConfiguration

@dataclass
class EntropyComparisonRequest:
    """DTO for the use case input."""
    n_qubits: int
    control_mode: str
    control_value: float

@dataclass
class EntropyComparisonResponse:
    """DTO for the use case output."""
    boltzmann_entropy: float
    gibbs_entropy: float
    von_neumann_entropy: float
    control_mode: str
    control_value: float
    equivalent_temperature: float
    equivalent_energy: float
    gibbs_distribution: Dict[float, float]
    von_neumann_eigenvalues: List[float]
    degeneracy: int
    mean_energy: float
    std_dev_energy: float
    quantum_coherences: float
    max_diff: float
    boltzmann_label: str
    gibbs_label: str
    vn_label: str
    boltzmann_microstates: List[SpinConfiguration]
    gibbs_microstates: List[SpinConfiguration]
    von_neumann_microstates: List[SpinConfiguration]

@dataclass
class DecoherenceRequest:
    """DTO for the decoherence simulation input."""
    T_bath: float

@dataclass
class DecoherenceResponse:
    """DTO for the decoherence simulation output."""
    times: np.ndarray
    s_vn_t: np.ndarray
    s_gibbs: float
