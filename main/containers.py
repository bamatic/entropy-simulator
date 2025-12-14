from dependency_injector import containers, providers

from application.use_cases import (
    CalculateEntropyComparisonUseCase,
    CalculateDecoherenceUseCase,
)
from domain.services import EntropyComparisonService
from infrastructure.calculators import (
    NumPyMicrocanonicalCalculator,
    NumPyCanonicalCalculator,
    NumPyVonNeumannCalculator,
)
from infrastructure.converters import SciPyTemperatureEnergyConverter
from infrastructure.decoherence_calculator import ScipyDecoherenceCalculator

class Container(containers.DeclarativeContainer):
    """
    Dependency injection container.
    """
    config = providers.Configuration()

    # Calculators
    microcanonical_calculator = providers.Singleton(NumPyMicrocanonicalCalculator)
    canonical_calculator = providers.Singleton(
        NumPyCanonicalCalculator,
        microcanonical_calculator=microcanonical_calculator,
    )
    von_neumann_calculator = providers.Singleton(
        NumPyVonNeumannCalculator,
        microcanonical_calculator=microcanonical_calculator,
    )
    decoherence_calculator = providers.Singleton(ScipyDecoherenceCalculator)

    # Converter
    temp_energy_converter = providers.Singleton(
        SciPyTemperatureEnergyConverter,
        canonical_calculator=canonical_calculator,
    )

    # Domain Service
    entropy_comparison_service = providers.Singleton(
        EntropyComparisonService,
        microcanonical_calculator=microcanonical_calculator,
        canonical_calculator=canonical_calculator,
        von_neumann_calculator=von_neumann_calculator,
        temp_energy_converter=temp_energy_converter,
    )

    # Use Cases
    calculate_entropy_use_case = providers.Factory(
        CalculateEntropyComparisonUseCase,
        entropy_comparison_service=entropy_comparison_service,
    )
    calculate_decoherence_use_case = providers.Factory(
        CalculateDecoherenceUseCase,
        decoherence_calculator=decoherence_calculator,
    )
