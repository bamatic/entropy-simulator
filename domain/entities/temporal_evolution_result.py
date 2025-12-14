from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class TemporalEvolutionResult:
    """
    Represents the result of a temporal evolution simulation.
    """
    times: np.ndarray
    s_vn_t: np.ndarray
    s_gibbs: float
