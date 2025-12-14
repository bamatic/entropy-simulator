# Interactive Entropy Unification Simulator

This project is an interactive scientific simulator that compares three fundamental definitions of entropy in statistical mechanics and quantum mechanics: Boltzmann, Gibbs, and von Neumann entropy.

## Architecture

The project follows the principles of Domain-Driven Design (DDD) and Clean Architecture (Hexagonal Architecture).

- **domain**: Contains the core business logic and is independent of any external libraries. It defines the entities, value objects, domain services, and ports (interfaces).
- **application**: Contains the use cases that orchestrate the domain services. It is independent of the UI and infrastructure.
- **infrastructure**: Implements the ports defined in the domain layer using external libraries like NumPy and SciPy.
- **main**: Contains the Streamlit UI and the dependency injection container.

## How to Run

1.  **Install core dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
2.  **Install project in editable mode (one-time setup):**
    This makes the 'application', 'domain', 'infrastructure', and 'main' packages discoverable.
    ```bash
    pip install -e .
    ```
3.  **Run the Streamlit application:**
    ```bash
    streamlit run main/app.py
    ```
