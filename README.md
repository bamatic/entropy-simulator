# Interactive Entropy Unification Simulator

A scientific web application demonstrating the unification of statistical mechanics through von Neumann entropy formalism. This simulator provides interactive visualization of how Boltzmann, Gibbs, and von Neumann entropy definitions are particular cases of a single theoretical framework.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=Streamlit&logoColor=white)](https://streamlit.io)

🔗 **[Live Demo](https://your-app-url.streamlit.app)** | 📄 **[Academic Paper](link-to-paper.pdf)**

---

## 📚 Table of Contents

- [Physical Context](#physical-context)
- [Features](#features)
- [Quick Start](#quick-start)
- [Usage Guide](#usage-guide)
  - [Tab 1: Thermal Equilibrium](#tab-1-thermal-equilibrium)
  - [Tab 2: Temporal Evolution (Decoherence)](#tab-2-temporal-evolution-decoherence)
- [Architecture](#architecture)
- [Technical Details](#technical-details)
- [Development](#development)
- [Citation](#citation)
- [License](#license)

---

## 🔬 Physical Context

This simulator implements the **density matrix formalism** as a unifying framework for statistical mechanics. It demonstrates that the three entropy expressions taught in standard statistical mechanics courses are special cases of a single general definition:

### **Entropy Unification Hierarchy**
```
von Neumann Entropy (Most General)
    S = -k Tr(ρ ln ρ)
         ↓
    ┌────┴────┐
    ↓         ↓
Gibbs       Boltzmann
(Canonical)  (Microcanonical)
S = -k Σ P(R)ln P(R)   S = k ln Ω(E)
```

### **Key Physical Results**

1. **Thermal Equilibrium (Tab 1):** For diagonal Hamiltonians H = Σσᶻ, demonstrates that S_vN = S_Gibbs exactly, with numerical precision < 10⁻¹⁰.

2. **Non-Equilibrium Dynamics (Tab 2):** Shows how a pure quantum state (S=0) evolves to thermal equilibrium (S=S_Gibbs) through decoherence, illustrating the microscopic foundation of the Second Law of Thermodynamics.

### **Physical System**

- **Model:** N independent qubits (N ∈ [2,10])
- **Hamiltonian:** H = Σᵢ σᶻᵢ (diagonal in computational basis)
- **Energy spectrum:** E = N - 2m, where m = number of spin-ups
- **Degeneracy:** Ω(E) = C(N,m) with m = (N-E)/2

---

## ✨ Features

### **Tab 1: Thermal Equilibrium Comparison**
- Interactive visualization of three entropy formulations
- Real-time calculation of Ω(E), P(E), and eigenvalues
- Numerical verification of S_vN = S_Gibbs for diagonal H
- Adjustable parameters: N (qubits), T (temperature), E (fixed energy)

### **Tab 2: Decoherence Dynamics**
- Lindblad master equation numerical integration
- Real-time evolution from pure state to thermal equilibrium
- Visualization of S_vN(t) → S_Gibbs convergence
- Demonstrates information conservation vs. entropy growth

### **Numerical Methods**
- Eigenvalue decomposition using LAPACK (via NumPy)
- Time integration with RK45 (Runge-Kutta 5(4) method)
- Tolerances: rtol=10⁻⁸, atol=10⁻¹⁰
- Validation: Trace conservation < 10⁻⁸

---

## 🚀 Quick Start

### **Prerequisites**
- Python 3.10 or higher
- pip package manager

### **Installation**
```bash
# Clone the repository
git clone https://github.com/yourusername/entropy-simulator.git
cd entropy-simulator

# Install dependencies
pip install -r requirements.txt

# Install in editable mode (for development)
pip install -e .

# Run the application
streamlit run main/app.py
```

The application will open in your default browser at `http://localhost:8501`

---

## 📖 Usage Guide

### **Tab 1: Thermal Equilibrium**

This tab compares the three entropy definitions for systems in thermal equilibrium.

#### **Controls**

**Number of Qubits (N):** Select from 2 to 10
- Determines system size (2ᴺ states)
- Affects computational cost (scales as O(2³ᴺ))
- Recommended: N=4 for optimal balance

**Temperature (T) or Energy (E):**
- **Canonical ensemble:** Set temperature T (Kelvin)
- **Microcanonical ensemble:** Set fixed energy E
- Valid energies: E ∈ {-N, -N+2, ..., N-2, N}

#### **Understanding the Output**

**1. Boltzmann Entropy (Microcanonical)**
```
S₁ = k ln Ω(E)
```
- Shows number of microstates Ω(E) with energy E
- Assumes all accessible states are equiprobable
- Example: N=4, E=-2 → Ω=4 → S=k ln(4)=1.386

**2. Gibbs Entropy (Canonical)**
```
S₂ = -k Σ P(E) ln P(E)
```
- Probability distribution over ALL energy levels
- P(E) = Ω(E)exp(-βE)/Z with β=1/(kT)
- Explores states beyond fixed energy

**3. von Neumann Entropy (Quantum)**
```
S₃ = -k Tr(ρ ln ρ) = -k Σ λᵢ ln λᵢ
```
- Calculated from density matrix eigenvalues λᵢ
- For diagonal H: **S₃ = S₂ exactly** (numerical verification)
- Purity Tr(ρ²) indicates mixed vs. pure state

#### **Physical Interpretation**

**Coherences = 0.0000**
- Density matrix is diagonal (no off-diagonal elements)
- System is in classical statistical mixture
- No quantum superpositions in thermal equilibrium

**Purity Tr(ρ²) < 1**
- Indicates mixed state (thermal distribution)
- Effective number of states: N_eff = 1/Tr(ρ²)
- Example: Tr(ρ²)=0.153 → ~6.5 states contribute significantly

**Key Result:** For diagonal Hamiltonians, S_vN = S_Gibbs exactly, confirming that classical and quantum statistical mechanics give identical predictions in equilibrium.

---

### **Tab 2: Temporal Evolution (Decoherence)**

This tab shows how a pure quantum state evolves to thermal equilibrium through interaction with an environment.

#### **Controls**

**Bath Temperature (T_bath):** Adjustable from 0.1 to 5.0 K
- Determines final equilibrium entropy S_Gibbs
- Higher T → higher final entropy

**Fixed Parameters:**
- N = 4 qubits
- Initial state: |1111⟩ (ground state, S=0)
- Relaxation time: T₁ = 1.0
- Dephasing time: T₂ = 0.5
- Simulation time: t_max = 10.0

#### **Understanding the Output**

**Evolution Plot:**
- **Blue curve:** S_vN(t) - actual system entropy
- **Green line:** S_Gibbs - equilibrium target entropy
- **Convergence:** S_vN(t→∞) → S_Gibbs

**Physical Process:**

1. **t = 0:** System in pure state |↑↑↑↑⟩
   - S_vN(0) = 0
   - Tr(ρ²) = 1 (pure)

2. **0 < t < ~5:** Thermalization process
   - Loss of quantum coherences: ρ₀₁(t) ∝ exp(-t/T₂)
   - Entropy growth: dS/dt > 0 (Second Law)
   - Information migrates to system-environment correlations

3. **t → ∞:** Thermal equilibrium
   - S_vN(∞) → S_Gibbs
   - Density matrix becomes thermal: ρ_eq = exp(-βH)/Z
   - Tr(ρ²) < 1 (mixed state)

#### **Physical Interpretation**

**Lindblad Master Equation:**
```
dρ/dt = -i[H,ρ] + Σ γᵢ(LᵢρL†ᵢ - ½{L†ᵢLᵢ,ρ})
```

**Terms:**
- `-i[H,ρ]`: Unitary evolution (reversible)
- `Lindblad terms`: Dissipation and decoherence (irreversible)

**Operators:**
- **Dephasing:** L_dephase = √(1/T₂) σᶻ
- **Decay:** L_decay = √(1/T₁) σ⁻
- **Excitation:** L_excite = √(1/T₁) σ⁺

**Key Result:** The monotonic growth of S_vN(t) demonstrates the microscopic origin of the Second Law: entropy increases because the subsystem becomes entangled with environmental degrees of freedom that we cannot observe.

#### **Information Conservation vs. Entropy Growth**

**Apparent Paradox:**
- Microscopic evolution is unitary → information conserved
- Macroscopic observation shows entropy increase → information lost?

**Resolution:**
- **Total system + environment:** S_total = 0 (constant, unitary)
- **Subsystem only:** S_A > 0 (growing, non-unitary reduced dynamics)
- **Information migrates** to correlations with ~10²³ environmental DoF
- These correlations are **practically inaccessible** → effective irreversibility

---

## 🏗️ Architecture

The project follows **Clean Architecture** (Hexagonal Architecture) and **Domain-Driven Design** principles for maintainability, testability, and framework independence.

### **Layer Structure**
```
entropy-simulator/
├── domain/              # Core business logic (framework-agnostic)
│   ├── entities/        # System state representations
│   │   ├── quantum_system.py
│   │   └── temporal_evolution_result.py
│   ├── value_objects/   # Immutable value types
│   │   └── entropy.py
│   └── contracts/       # Port interfaces (protocols)
│       ├── microcanonical_calculator.py
│       ├── canonical_calculator.py
│       ├── von_neumann_calculator.py
│       └── decoherence_calculator.py
│
├── application/         # Use cases (orchestration)
│   └── use_cases/
│       ├── compare_entropies.py
│       └── simulate_decoherence.py
│
├── infrastructure/      # External dependencies implementation
│   ├── calculators/
│   │   ├── numpy_microcanonical.py    # Implements MicrocanonicalPort
│   │   ├── numpy_canonical.py         # Implements CanonicalPort
│   │   ├── numpy_von_neumann.py       # Implements VonNeumannPort
│   │   └── scipy_decoherence.py       # Implements DecoherencePort
│   └── persistence/     # (Optional) Storage adapters
│
└── main/                # Application entry point
    ├── app.py           # Streamlit UI
    └── container.py     # Dependency injection
```

### **Design Principles Applied**

#### **1. Dependency Inversion Principle (DIP)**
```python
# Domain defines the contract (Port)
class VonNeumannCalculatorPort(Protocol):
    def calculate_entropy(...) -> Entropy: ...

# Infrastructure implements it (Adapter)
class NumPyVonNeumannCalculator(VonNeumannCalculatorPort):
    def calculate_entropy(...) -> Entropy:
        # NumPy/SciPy implementation
```

**Benefit:** Domain doesn't depend on NumPy. Can swap implementations without touching business logic.

#### **2. Single Responsibility Principle (SRP)**
- Each calculator handles ONE type of entropy
- Use cases orchestrate, don't calculate
- Entities hold data, don't compute

#### **3. Open/Closed Principle (OCP)**
- New entropy definitions? Add new calculator (open for extension)
- Existing calculators unchanged (closed for modification)

#### **4. Interface Segregation Principle (ISP)**
- Each port defines minimal interface
- UI depends only on use cases, not calculators

### **Data Flow**
```
User Input (Streamlit)
    ↓
Use Case (Application Layer)
    ↓
Multiple Calculators (via Ports)
    ↓
Domain Entities & Value Objects
    ↓
Results back to UI
```

### **Example: Entropy Comparison Flow**
```python
# 1. User sets N=4, T=1.82
# 2. UI calls use case
result = compare_entropies_use_case.execute(
    quantum_system=QuantumSystem(n_qubits=4),
    temperature=1.82
)

# 3. Use case orchestrates (application/use_cases/compare_entropies.py)
s_boltzmann = microcanonical_calc.calculate_entropy(...)
s_gibbs = canonical_calc.calculate_entropy(...)
s_von_neumann = von_neumann_calc.calculate_entropy(...)

# 4. Calculators use domain entities (infrastructure/calculators/*.py)
# 5. Results returned as domain value objects (Entropy)
# 6. UI displays results
```

### **Why This Architecture?**

**For Physics:**
- Domain layer contains only physics equations (no framework noise)
- Easy to verify correctness of physical implementations
- Clear separation between math/physics and visualization

**For Software Engineering:**
- Testable: Mock ports in tests, verify use cases independently
- Maintainable: Change UI framework without touching calculations
- Extensible: Add new entropy types without breaking existing code
- Professional: Demonstrates software engineering best practices

**Trade-offs:**
- More files/folders (may seem over-engineered for small project)
- Requires understanding of architectural patterns
- Worth it for: portfolios, production code, team projects
- Overkill for: throwaway scripts, simple notebooks

---

## 🔧 Technical Details

### **Numerical Methods**

**Eigenvalue Decomposition:**
- Algorithm: LAPACK `eigh()` for Hermitian matrices
- Complexity: O(d³) where d=2ᴺ
- Threshold: λ < 10⁻¹⁵ filtered to avoid log(0)

**Time Integration (Tab 2):**
- Method: RK45 (Runge-Kutta 5th order with 4th order error estimate)
- Adaptive step size for efficiency
- Tolerances: rtol=10⁻⁸, atol=10⁻¹⁰

**Validation:**
- Trace conservation: |Tr(ρ)-1| < 10⁻⁸
- Hermiticity: ||ρ - ρ†|| < 10⁻¹⁰
- Positivity: All eigenvalues ≥ -10⁻¹² (numerical noise)

### **Computational Complexity**

| N | States (2ᴺ) | Hamiltonian | Eigenvalues | Tab 1 Time | Tab 2 Time |
|---|-------------|-------------|-------------|------------|------------|
| 2 | 4 | 16 bytes | ~1 μs | <10 ms | <50 ms |
| 4 | 16 | 2 KB | ~10 μs | <50 ms | <200 ms |
| 6 | 64 | 32 KB | ~100 μs | <200 ms | ~1 s |
| 8 | 256 | 512 KB | ~1 ms | ~1 s | ~5 s |
| 10 | 1024 | 8 MB | ~10 ms | ~5 s | ~30 s |

**Recommendation:** N≤6 for interactive use, N≤8 for batch calculations.

### **Technology Stack**

- **Python 3.10+** - Core language
- **NumPy 1.24+** - Linear algebra, array operations
- **SciPy 1.11+** - Integration (`solve_ivp`), special functions
- **Streamlit 1.28+** - Web interface
- **Plotly** - Interactive visualizations

---

## 👨‍💻 Development

### **Project Structure Philosophy**

This project demonstrates professional software engineering practices alongside scientific computing. The architecture is intentionally over-engineered compared to typical physics projects to showcase:

1. **Clean Architecture** for long-term maintainability
2. **Domain-Driven Design** for complex domain modeling
3. **SOLID principles** for extensible code
4. **Ports & Adapters** for framework independence

If you're looking for a simpler structure for a quick project, consider collapsing the layers into a single module. This architecture makes sense for:
- Production applications
- Team collaboration
- Portfolio demonstration
- Long-term maintained projects

### **Running Tests**
```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run all tests
pytest

# Run with coverage
pytest --cov=domain --cov=application --cov=infrastructure

# Run specific test file
pytest tests/domain/test_quantum_system.py
```

### **Code Style**
```bash
# Format code
black .

# Check types
mypy domain/ application/ infrastructure/

# Lint
ruff check .
```

### **Adding a New Entropy Definition**

1. Define port in `domain/contracts/`:
```python
class NewEntropyCalculatorPort(Protocol):
    def calculate_entropy(...) -> Entropy: ...
```

2. Implement in `infrastructure/calculators/`:
```python
class NewEntropyCalculator(NewEntropyCalculatorPort):
    def calculate_entropy(...) -> Entropy:
        # Implementation
```

3. Register in dependency container (`main/container.py`)

4. Add to use case if needed

5. Update UI to display results

---

## 📊 Citation

If you use this simulator in academic work, please cite:
```bibtex
@misc{entropy-simulator-2024,
  author = {Your Name},
  title = {Interactive Entropy Unification Simulator},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/yourusername/entropy-simulator}
}
```

Related academic work:
```bibtex
@unpublished{yourname2024vonneumann,
  author = {Your Name},
  title = {Von Neumann Entropy: Statistical Unification and the Second Law},
  note = {Statistical Mechanics - UNED},
  year = {2024}
}
```

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **UNED Statistical Mechanics Course** - Theoretical foundation
- **von Neumann (1932)** - Density matrix formalism
- **Lindblad (1976)** - Master equation for open quantum systems

---

## 📧 Contact

**Your Name** - [your.email@example.com](mailto:your.email@example.com)

Project Link: [https://github.com/yourusername/entropy-simulator](https://github.com/yourusername/entropy-simulator)

---

## 🗺️ Roadmap

**Completed:**
- ✅ Thermal equilibrium comparison (3 entropies)
- ✅ Decoherence dynamics simulation
- ✅ Interactive Streamlit UI
- ✅ Clean Architecture implementation

**Future Enhancements:**
- [ ] Add transverse field Ising model (non-diagonal H)
- [ ] Implement partial trace visualization for subsystems
- [ ] Add export functionality (data, plots)
- [ ] Performance optimization for N>10
- [ ] Add more decoherence channels (amplitude damping, etc.)
- [ ] Educational mode with step-by-step explanations

---

**Made with ❤️ for physics and clean code**