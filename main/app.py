import sys
import os
import streamlit as st
import pandas as pd
import altair as alt
from typing import List, Dict, Any
import math

# Add the project root to sys.path to enable absolute imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, project_root)

from main.containers import Container
from application.dtos import EntropyComparisonRequest, DecoherenceRequest
from domain.entities import SpinConfiguration

# Helper function to render a single spin configuration
def _render_spin_configuration(
    spin_config: SpinConfiguration,
    max_probability: float,
    ensemble_type: str, # "Boltzmann", "Gibbs", "VonNeumann"
    key_prefix: str = ""
):
    # Calculate opacity (min 10% for visibility)
    opacity = max(0.1, spin_config.probability / max_probability) if max_probability > 0 else 1.0

    # Define colors for spins
    spin_up_color = "#1f77b4" # Blue
    spin_down_color = "#ff7f0e" # Orange

    # Create spin visualization string
    spin_chars = []
    for spin in spin_config.spins:
        if spin == 1: # Spin Up
            spin_chars.append(f"<span style='color:{spin_up_color}; font-size:24px;'>↑</span>")
        else: # Spin Down
            spin_chars.append(f"<span style='color:{spin_down_color}; font-size:24px;'>↓</span>")
    spin_display = "".join(spin_chars)

    # Tooltip content
    tooltip_content = (
        f"**State:** |{spin_config.binary_representation}⟩<br>"
        f"**Index:** {spin_config.index}<br>"
        f"**Energy:** {spin_config.energy}<br>"
        f"**Spins Up:** {spin_config.num_spins_up}<br>"
        f"**Spins Down:** {spin_config.num_spins_down}<br>"
        f"**Probability:** {spin_config.probability:.4f}"
    )

    # Probability bar
    prob_bar_width = int(spin_config.probability / max_probability * 100) if max_probability > 0 else 0
    prob_bar = f"<div style='width:100%; height:5px; background-color:#eee; border-radius:3px;'><div style='width:{prob_bar_width}%; height:100%; background-color:#2ca02c; border-radius:3px;'></div></div>"

    st.markdown(
        f"<div style='display:flex; align-items:center; margin-bottom:5px; opacity:{opacity};' title='{tooltip_content}'>"
        f"<div style='width:120px; text-align:left;'>{spin_display}</div>"
        f"<div style='width:80px; font-size:12px;'>E={spin_config.energy:.0f}</div>"
        f"<div style='width:100px; font-size:12px;'>P={spin_config.probability:.3f}</div>"
        f"<div style='flex-grow:1;'>{prob_bar}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )


# Helper function to render a list of spin configurations with pagination/filtering
def _render_microstate_viewer(
    title: str,
    microstates: List[SpinConfiguration],
    n_qubits: int,
    ensemble_type: str, # "Boltzmann", "Gibbs", "VonNeumann"
    key_prefix: str = ""
):
    st.subheader(title)

    if not microstates:
        st.info("No microstates to display for current selection.")
        return

    # Determine max probability for scaling opacity
    max_probability = max(ms.probability for ms in microstates) if microstates else 1.0

    # --- Sorting and Filtering ---
    display_microstates = list(microstates) # Create a mutable copy for sorting/filtering

    # Boltzmann has no sorting/filtering options as all are equiprobable and filtered by energy already
    if ensemble_type != "Boltzmann":
        with st.expander(f"Display Options for {title}", expanded=False):
            # Sorting
            sort_options = {
                "Probability (desc)": lambda ms: ms.probability,
                "Energy (asc)": lambda ms: ms.energy,
                "State Index (asc)": lambda ms: ms.index,
            }
            sort_by = st.selectbox(
                "Sort by",
                list(sort_options.keys()),
                key=f"{key_prefix}_sort_by"
            )
            reverse_sort = True if "desc" in sort_by else False
            display_microstates.sort(key=sort_options[sort_by], reverse=reverse_sort)

            # Filtering for N >= 7
            if n_qubits >= 7:
                st.markdown("---")
                st.write("**Filters**")
                
                # Probability Threshold
                min_prob_threshold_percent = st.slider(
                    "Minimum Probability Threshold (%)",
                    min_value=0.0,
                    max_value=10.0,
                    value=0.1,
                    step=0.1,
                    format="%.1f%%",
                    key=f"{key_prefix}_prob_threshold"
                )
                min_prob_threshold = min_prob_threshold_percent / 100.0
                display_microstates = [ms for ms in display_microstates if ms.probability >= min_prob_threshold]

                # Energy Level Filter
                all_energies = sorted(list(set(ms.energy for ms in microstates)))
                selected_energies = st.multiselect(
                    "Filter by Energy Level",
                    options=all_energies,
                    default=all_energies,
                    key=f"{key_prefix}_energy_filter"
                )
                display_microstates = [ms for ms in display_microstates if ms.energy in selected_energies]
        
        if not display_microstates:
            st.info("No configurations match current filters.")
            # Only show reset button if filters were applied
            if n_qubits >= 7 and (min_prob_threshold_percent > 0.0 or len(selected_energies) < len(all_energies)):
                if st.button("Reset Filters", key=f"{key_prefix}_reset_filters"):
                    st.session_state[f"{key_prefix}_prob_threshold"] = 0.1 # Reset to default
                    st.session_state[f"{key_prefix}_energy_filter"] = all_energies # Reset to default
                    st.experimental_rerun() 
            return

    total_configs_after_filter = len(display_microstates)
    
    # --- Display Logic ---
    if n_qubits <= 4: # Show all
        st.write(f"Displaying all {total_configs_after_filter} configurations.")
        for ms in display_microstates:
            _render_spin_configuration(ms, max_probability, ensemble_type, key_prefix)
    elif n_qubits <= 6: # Pagination
        states_per_page = 12
        total_pages = math.ceil(total_configs_after_filter / states_per_page)
        
        # Initialize session state for current page if not exists
        if f"{key_prefix}_current_page" not in st.session_state:
            st.session_state[f"{key_prefix}_current_page"] = 0

        current_page = st.session_state[f"{key_prefix}_current_page"]

        start_idx = current_page * states_per_page
        end_idx = min(start_idx + states_per_page, total_configs_after_filter)
        
        st.write(f"Showing {start_idx + 1}-{end_idx} of {total_configs_after_filter} configurations.")

        # Pagination buttons
        col_prev, col_page_info, col_next = st.columns([1, 2, 1])
        with col_prev:
            if st.button("Previous", disabled=(current_page == 0), key=f"{key_prefix}_prev_page"):
                st.session_state[f"{key_prefix}_current_page"] -= 1
                st.experimental_rerun()
        with col_page_info:
            st.markdown(f"<div style='text-align:center;'>Page {current_page + 1} of {total_pages}</div>", unsafe_allow_html=True)
        with col_next:
            if st.button("Next", disabled=(current_page >= total_pages - 1), key=f"{key_prefix}_next_page"):
                st.session_state[f"{key_prefix}_current_page"] += 1
                st.experimental_rerun()

        # Render states for current page
        for ms in display_microstates[start_idx:end_idx]:
            _render_spin_configuration(ms, max_probability, ensemble_type, key_prefix)

    else: # N >= 7: Show top 20 + filters
        num_to_display = 20
        total_initial_configs = len(microstates) # Total before any filtering
        num_filtered_configs = len(display_microstates)
        
        st.write(f"Showing top {min(num_to_display, num_filtered_configs)} of {num_filtered_configs} configurations (from {total_initial_configs} total).")
        if num_filtered_configs > num_to_display:
            st.write(f"({num_filtered_configs - num_to_display} configurations not shown or have probability below threshold)")

        for ms in display_microstates[:num_to_display]:
            _render_spin_configuration(ms, max_probability, ensemble_type, key_prefix)

    # Educational notes
    if ensemble_type == "Boltzmann":
        st.markdown(
            """
            <p style='font-size:12px; font-style:italic;'>
            In the microcanonical ensemble, all microstates with energy E are equally probable.
            This is the foundation of Boltzmann entropy: S = k ln Ω(E)
            </p>
            """,
            unsafe_allow_html=True
        )
    elif ensemble_type == "Gibbs":
        st.markdown(
            """
            <p style='font-size:12px; font-style:italic;'>
            In the canonical ensemble, all microstates are accessible, but lower energy states are more probable
            at finite temperature. Opacity indicates probability.
            </p>
            """,
            unsafe_allow_html=True
        )
    elif ensemble_type == "VonNeumann":
        st.markdown(
            """
            <p style='font-size:12px; font-style:italic;'>
            For H = Σ σᶻ (diagonal Hamiltonian), the density matrix is diagonal. This means no quantum
            superpositions exist in thermal equilibrium. Each eigenvalue λᵢ is the probability of measuring
            state |i⟩. Result: S_von Neumann = S_Gibbs (exactly). Quantum effects only appear with
            non-diagonal Hamiltonians.
            </p>
            """,
            unsafe_allow_html=True
        )


def run_equilibrium_simulation(n_qubits, control_mode, control_value):
    """Helper to run equilibrium simulation and cache results."""
    container = Container()
    use_case = container.calculate_entropy_use_case()
    request = EntropyComparisonRequest(
        n_qubits=n_qubits,
        control_mode=control_mode,
        control_value=control_value,
    )
    return use_case.execute(request)

@st.cache_data
def run_decoherence_simulation(T_bath):
    """Helper to run decoherence simulation and cache results."""
    container = Container()
    use_case = container.calculate_decoherence_use_case()
    request = DecoherenceRequest(T_bath=T_bath)
    return use_case.execute(request)


def build_equilibrium_tab():
    # --- Control Panel ---
    st.sidebar.header("Equilibrium Controls")
    n_qubits = st.sidebar.slider("Number of Qubits (N)", 1, 10, 4, key="eq_n_qubits")

    control_mode = st.sidebar.radio("Control by:", ("Energy (E)", "Temperature (T)"), key="eq_control_mode")
    control_mode_str = "Energy" if "Energy" in control_mode else "Temperature"

    if control_mode_str == "Energy":
        min_e = -n_qubits
        max_e = 0 # Limit max energy to 0
        # Ensure the default value is valid
        energy_value = 0 if min_e <= 0 <= max_e else min_e
        control_value = st.sidebar.slider("Fixed Energy (E)", min_e, max_e, energy_value, 2, key="eq_control_value_e")
        st.sidebar.info(
            "ℹ️ Energy slider is limited to E ≤ 0 because in thermal equilibrium "
            "at any finite temperature T > 0, lower energy states are always favored, "
            "so ⟨E⟩ < 0. For E = 0, the system would need T → ∞ (equiprobability)."
        )
    else:
        max_temp = 2 * n_qubits
        control_value = st.sidebar.slider("Heat Bath Temperature (T)", 1.0, float(max_temp), 1.0, 0.1, key="eq_control_value_t")

    # --- Run Simulation ---
    try:
        result = run_equilibrium_simulation(n_qubits, control_mode_str, control_value)
    except ValueError as e:
        st.error(f"An error occurred: {e}")
        return

    # --- Equivalent Value Display ---
    st.sidebar.header("Calculated Equivalent")
    if control_mode_str == "Energy":
        st.sidebar.metric("Equivalent Temperature", f"{result.equivalent_temperature:.2f} K")
    else:
        st.sidebar.metric("Equivalent Mean Energy", f"{result.equivalent_energy:.2f}")

    # --- Main Display ---
    st.header("Entropy Comparison Summary")
    # Adjust column widths to make space for the microstate viewer
    col_main_1, col_viewer_1 = st.columns([0.6, 0.4])
    col_main_2, col_viewer_2 = st.columns([0.6, 0.4])
    col_main_3, col_viewer_3 = st.columns([0.6, 0.4])


    # --- Section 1: Boltzmann ---
    with col_main_1:
        st.header("1. Boltzmann Entropy")
        st.caption("Microcanonical - Isolated System")
        
        df_boltzmann = pd.DataFrame({
            'Energy': list(result.gibbs_distribution.keys()),
            'Probability': [1.0 if e == result.control_value else 0.0 for e in result.gibbs_distribution.keys()]
        })
        chart_b = alt.Chart(df_boltzmann).mark_bar().encode(
            x=alt.X('Energy', type='quantitative', axis=alt.Axis(title='Energy (E)')),
            y=alt.Y('Probability', type='quantitative', axis=alt.Axis(title='Probability P(E)')),
            tooltip=['Energy', 'Probability']
        ).properties(height=200)
        st.altair_chart(chart_b, use_container_width=True)

        st.metric("Number of microstates Ω(E)", f"{result.degeneracy}")
        st.metric("S₁ (Boltzmann)", f"{result.boltzmann_entropy:.3f}", delta=None)
        st.markdown(f"_{result.boltzmann_label}_")
    with col_viewer_1:
        _render_microstate_viewer(
            "Boltzmann Microstates",
            result.boltzmann_microstates,
            n_qubits,
            "Boltzmann",
            key_prefix="boltzmann"
        )

    # --- Section 2: Gibbs ---
    with col_main_2:
        st.header("2. Gibbs Entropy")
        st.caption("Canonical - Classical System")

        df_gibbs = pd.DataFrame({
            'Energy': list(result.gibbs_distribution.keys()),
            'Probability': list(result.gibbs_distribution.values())
        })
        chart_g = alt.Chart(df_gibbs).mark_bar().encode(
            x=alt.X('Energy', type='quantitative', axis=alt.Axis(title='Energy (E)')),
            y=alt.Y('Probability', type='quantitative', axis=alt.Axis(title='Probability P(E)')),
            tooltip=['Energy', 'Probability']
        ).properties(height=200)
        st.altair_chart(chart_g, use_container_width=True)

        st.metric("Mean Energy ⟨E⟩", f"{result.mean_energy:.2f} (σ_E={result.std_dev_energy:.2f})")
        delta_g = (result.gibbs_entropy - result.boltzmann_entropy) / result.boltzmann_entropy * 100 if result.boltzmann_entropy > 0 else 0
        st.metric("S₂ (Gibbs)", f"{result.gibbs_entropy:.3f}", delta=f"{delta_g:.1f}% vs Boltzmann")
        st.markdown(f"_{result.gibbs_label}_")
    with col_viewer_2:
        _render_microstate_viewer(
            "Gibbs Microstates",
            result.gibbs_microstates,
            n_qubits,
            "Gibbs",
            key_prefix="gibbs"
        )

    # --- Section 3: von Neumann ---
    with col_main_3:
        st.header("3. von Neumann Entropy")
        st.caption("Canonical - Quantum System")

        eigenvalues = result.von_neumann_eigenvalues
        df_vn = pd.DataFrame({
            'Index': range(len(eigenvalues)),
            'Eigenvalue': eigenvalues
        }).nlargest(50, 'Eigenvalue') # Show top 50
        chart_vn = alt.Chart(df_vn).mark_bar().encode(
            x=alt.X('Index', type='ordinal', sort=None, axis=alt.Axis(title='Eigenvalue Index')),
            y=alt.Y('Eigenvalue', type='quantitative', axis=alt.Axis(title='Magnitude λᵢ')),
            tooltip=['Index', 'Eigenvalue']
        ).properties(height=200)
        st.altair_chart(chart_vn, use_container_width=True)

        st.metric("Significant Eigenvalues", f"{len([e for e in eigenvalues if e > 1e-9])} of {len(eigenvalues)}")
        st.metric("Quantum Coherences |ρᵢⱼ|²", f"{result.quantum_coherences:.4f}")
        delta_vn = (result.von_neumann_entropy - result.gibbs_entropy) / result.gibbs_entropy * 100 if result.gibbs_entropy > 0 else 0
        st.metric("S₃ (von Neumann)", f"{result.von_neumann_entropy:.3f}", delta=f"{delta_vn:.1f}% vs Gibbs")
        
        # Calculate dynamic values for the educational message
        purity = sum(lam**2 for lam in result.von_neumann_eigenvalues)
        dim = 2**n_qubits
        n_eff = 1.0 / purity if purity > 0 else float('inf')
        max_mixed_purity = 1.0 / dim

        educational_message = f"""
        <div style='font-size: 16px; line-height: 1.6;'>
            <hr>
            <h5>Diagonal Hamiltonian (H = Σ σᶻ)</h5>
            <p>In thermal equilibrium with a diagonal Hamiltonian:</p>
            <ul>
                <li><b>Quantum Coherences:</b> {result.quantum_coherences:.4f}
                    <br>→ All off-diagonal elements of ρ are zero.
                    <br>→ No quantum superpositions exist.
                </li>
                <br>
                <li><b>Purity Tr(ρ²):</b> {purity:.3f}
                    <br>→ Purity = 1.0 would mean a <b>pure state</b>.
                    <br>→ Purity < 1.0 indicates a <b>mixed state</b> (thermal distribution).
                    <br>→ <b>Effective states:</b> ~{n_eff:.1f} states contribute significantly.
                    <br>→ <b>Comparison:</b>
                    <ul>
                        <li>Pure state: Tr(ρ²) = 1.0</li>
                        <li>Maximally mixed: Tr(ρ²) = {max_mixed_purity:.4f}</li>
                        <li>This state: Tr(ρ²) = {purity:.3f}</li>
                    </ul>
                </li>
            </ul>
            <h5>INTERPRETATION:</h5>
            <p>
            The system is in a <b>CLASSICAL STATISTICAL MIXTURE</b> over the computational 
            basis states |i⟩, each with probability λᵢ. There are no quantum coherences 
            between different energy eigenstates.
            </p>
            <p>
            This is fundamentally different from a quantum superposition like 
            <code>(|0⟩ + |1⟩)/√2</code>, which would have non-zero off-diagonal elements.
            </p>
            <h5>RESULT: S_von Neumann = S_Gibbs (exactly)</h5>
            <p>
            For diagonal Hamiltonians, von Neumann entropy equals Gibbs entropy 
            because both calculate the Shannon entropy over the same probability 
            distribution {{λᵢ}}.
            </p>
            <p>
            Quantum effects (S_vN ≠ S_Gibbs) only appear with <b>NON-DIAGONAL</b> 
            Hamiltonians that create coherent superpositions, such as:
            <br><code>H = Σσᶻ + h·Σσˣ</code> (transverse field Ising model)
            </p>
        </div>
        """
        st.markdown(educational_message, unsafe_allow_html=True)

    with col_viewer_3:
        _render_microstate_viewer(
            "Von Neumann Microstates",
            result.von_neumann_microstates,
            n_qubits,
            "VonNeumann",
            key_prefix="vonneumann"
        )
        # Additional info for Von Neumann
        st.metric("Quantum Coherences", f"{result.quantum_coherences:.4f}")
        # For diagonal H, Tr(rho^2) = sum(lambda_i^2)
        purity = sum(lam**2 for lam in result.von_neumann_eigenvalues)
        st.metric("Purity Tr(ρ²)", f"{purity:.3f}")


def build_decoherence_tab():
    st.header("Temporal Evolution: Quantum Thermalization")
    
    st.sidebar.header("Decoherence Controls")
    T_bath = st.sidebar.slider(
        "Bath Temperature T (K)", 
        min_value=0.1, 
        max_value=5.0, 
        value=2.0, 
        step=0.1,
        key="deco_T_bath"
    )

    if st.sidebar.button("Update Simulation", key="deco_update"):
        # This button click will trigger a rerun, and the cached function will be called with the new value
        pass

    with st.spinner("Simulating temporal evolution..."):
        result = run_decoherence_simulation(T_bath)

    st.markdown(
        """
        This tab shows how an initially pure quantum system 
        (all spins up, `S_vN = 0`) evolves towards thermal equilibrium 
        through decoherence and relaxation.

        Unlike Tab 1 (equilibrium), here `S_vN ≠ S_Gibbs` 
        during evolution, although they asymptotically converge.

        This illustrates that the equality `S_vN = S_Gibbs` only holds 
        in thermal equilibrium.
        """
    )

    # Create a DataFrame for plotting
    df = pd.DataFrame({
        'Time': result.times,
        'S_vN(t)': result.s_vn_t,
        'S_Gibbs': result.s_gibbs
    })

    # Melt the DataFrame for Altair
    df_melted = df.melt('Time', var_name='Entropy Type', value_name='Entropy')

    # Create the chart
    line_chart = alt.Chart(df_melted).mark_line().encode(
        x=alt.X('Time', title='Time (t)'),
        y=alt.Y('Entropy', title='Entropy (k_B units)'),
        color=alt.Color('Entropy Type', 
                        scale=alt.Scale(
                            domain=['S_vN(t)', 'S_Gibbs'], 
                            range=['#1f77b4', '#2ca02c'] # Blue, Green
                        ),
                        legend=alt.Legend(title="Entropy Type")),
        strokeDash=alt.condition(
            alt.datum['Entropy Type'] == 'S_Gibbs',
            alt.value([5, 5]),  # Dashed line for Gibbs
            alt.value([0]),     # Solid line for von Neumann
        )
    ).properties(
        title="Thermalization: Pure state → Thermal equilibrium"
    ).interactive()

    st.altair_chart(line_chart, use_container_width=True)

    with st.expander("Technical Notes"):
        st.markdown(
            """
            - **Initial state:** `|↑↑↑↑⟩` (pure state, `S_vN = 0`)
            - **Evolution:** Lindblad master equation with dephasing and thermal relaxation.
            - **Final state:** Thermal equilibrium `ρ_eq = exp(-βH)/Z`
            - **Numerical stability:** Master equation integration can be numerically unstable for very small `T₂`.
            - **Performance:** For N=4 (dim=16), simulation is fast (<1 second). For N>5, consider reducing `num_points` or using more efficient methods.
            """
        )


def main():
    st.set_page_config(layout="wide", page_title="Entropy Unification Simulator")
    st.title("Interactive Entropy Unification Simulator")

    tab1, tab2 = st.tabs(["Thermal Equilibrium", "Temporal Evolution (Decoherence)"])

    with tab1:
        build_equilibrium_tab()
    
    with tab2:
        build_decoherence_tab()
    
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        Created by **Victor ZUGADI GARCIA**
        <br>[GitHub](https://github.com/bamatic/entropy-simulator) | 
        [LinkedIn](https://www.linkedin.com/in/victor-zugadi-38595655/)
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
