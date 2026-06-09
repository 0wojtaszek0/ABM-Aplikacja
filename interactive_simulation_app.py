"""
Interaktywna aplikacja symulacji czynników ryzyka.
Pozwala użytkownikowi zmieniać wartości czynników ryzyka i uruchamiać symulacje.
"""

import os

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Tuple, List
import json
import math
import multiprocessing as mp
from datetime import datetime

from simulation_engine import SimulationEngine
from disease_model import DiseaseModel
from citizen import Citizen


# =====================================================================
# STAŁE GLOBALNE — RF i scenariusze (presety)
# =====================================================================

# Klucze RF muszą być angielskie — silnik symulacji (DiseaseModel.HAZARD_BETA,
# Citizen.DEFAULT_RISK_FACTORS) i logika run_simulation_with_rf używają
# angielskich identyfikatorów. Polskie etykiety wyświetlane w UI są w RF_LABELS_PL.
RF_NAMES = [
    "smoking",
    "obesity",
    "physical_inactivity",
    "alcohol_abuse",
    "high_cholesterol",
    "hypertension_stage0",
    "family_history",
]

RF_LABELS_PL = {
    "smoking":              "Palenie",
    "obesity":              "Otyłość (BMI)",
    "physical_inactivity":  "Brak aktywności fizycznej",
    "alcohol_abuse":        "Nadużywanie alkoholu",
    "high_cholesterol":     "Hipercholesterolemia",
    "hypertension_stage0":  "Nadciśnienie",
    "family_history":       "Obciążenie genetyczne",
}

# "Niestandardowy" oznacza brak presetu — wartości suwaków bez zmian.
SCENARIOS: Dict[str, Dict[str, float]] = {
    "Niestandardowy":                   {},
    "Nierealnie zdrowa populacja":      {rf: 0.5 for rf in RF_NAMES},
    "Wysokie ryzyko":                   {rf: 1.5 for rf in RF_NAMES},
    "Realnie optymistyczny scenariusz": {rf: 0.7 for rf in RF_NAMES},
}


def _json_safe(obj):
    """Kodek dla typów numpy w json.dumps(...)."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Type {type(obj).__name__} not JSON serializable")


def serialize_results(results: Dict, params: Dict) -> str:
    """Zwróć wyniki symulacji jako string JSON gotowy do pobrania.

    Konwertuje klucze int (lata w yearly_stats) na str, bo JSON nie wspiera
    nielist-owych kluczy nie-string. Deserializacja odwraca tę operację.
    """
    yearly_stats_str_keys = {
        str(year): year_data
        for year, year_data in (results.get("yearly_stats") or {}).items()
    }

    payload = {
        "timestamp": datetime.now().isoformat(),
        "schema_version": 1,
        "parameters": params,
        "results": {
            "initial_pop":         results["initial_pop"],
            "final_pop":           results["final_pop"],
            "avg_age_initial":     results["avg_age_initial"],
            "avg_age_final":       results["avg_age_final"],
            "deaths":              results.get("deaths", 0),
            "births":              results.get("births", 0),
            "cvd_count":           results["cvd_count"],
            "lung_cancer_count":   results["lung_cancer_count"],
            "multimorbidity_pct":  results["multimorbidity_pct"],
            "disease_prevalence":  results["disease_prevalence"],
            "rf_impact":           results["rf_impact"],
            "final_pyramid":       results.get("final_pyramid"),
            "yearly_stats":        yearly_stats_str_keys,
        },
    }
    return json.dumps(payload, indent=2, default=_json_safe, ensure_ascii=False)


def deserialize_results(payload: Dict) -> Tuple[Dict, Dict]:
    """Odtwórz (params, results) z wczytanego payloadu JSON.

    Przywraca int-owe klucze w yearly_stats (lata) — slider piramidy ich
    wymaga jako numeric.
    """
    params = payload.get("parameters", {})
    results = dict(payload.get("results", {}))
    if "yearly_stats" in results and results["yearly_stats"]:
        results["yearly_stats"] = {
            int(k): v for k, v in results["yearly_stats"].items()
        }
    return params, results


# =====================================================================
# RENDER WSPÓLNY DLA OBU ZAKŁADEK (Simulation + Wczytaj dane)
# =====================================================================

def render_results(results: Dict, rf_multipliers: Dict[str, float],
                   params: Dict = None, show_export: bool = True,
                   key_prefix: str = "sim") -> None:
    """Wyświetl pełny zestaw wyników: metryki, piramida, prevalencje,
    wpływ RF, trendy, tabele.

    Args:
        results: słownik wyników z run_simulation_with_rf (lub wczytany z JSON)
        rf_multipliers: mnożniki RF użyte do tej symulacji (do tabeli ustawień)
        params: opcjonalnie — pełne parametry (FM, MM, population_size, years)
                wykorzystane do eksportu
        show_export: czy pokazać sekcję eksportu (False dla zakładki "Wczytaj dane")
        key_prefix: prefiks kluczy widgetów (musi być różny dla każdej zakładki,
                    w której wywołujemy tę funkcję, inaczej Streamlit rzuca
                    StreamlitDuplicateElementKey)
    """
    # --- Metryki ---
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Początkowa populacja", f"{results['initial_pop']:,}",
                  help="Liczba agentów na początku symulacji")
    with col2:
        final_pop = results['final_pop']
        change = final_pop - results['initial_pop']
        st.metric("Populacja końcowa", f"{final_pop:,}", delta=f"{change:+,}",
                  help="Liczba agentów po zakończeniu okresu symulacji")
    with col3:
        survival_rate = (final_pop / results['initial_pop']) * 100 if results['initial_pop'] > 0 else 0
        st.metric("Wskaźnik przeżycia", f"{survival_rate:.1f}%",
                  help="Odsetek populacji początkowej żyjącej na koniec")
    with col4:
        st.metric("Średni wiek", f"{results['avg_age_final']:.1f} lat",
                  help="Średni wiek żyjącej populacji na koniec symulacji")

    st.divider()

    # --- Piramida wieku z suwakiem ---
    st.subheader("Piramida wieku populacji dla wybranego roku")
    yearly_stats = results.get('yearly_stats') or {}
    if yearly_stats:
        available_years = sorted(yearly_stats.keys())
        selected_year = st.slider(
            "Wybierz rok:",
            min_value=min(available_years), max_value=max(available_years),
            value=available_years[0], step=1,
            help="Wybierz rok, aby zobaczyć piramidę wieku populacji",
            key=f"{key_prefix}_pyramid_year_slider",
        )
        if selected_year in yearly_stats and yearly_stats[selected_year].get('age_pyramid'):
            pyramid_data = yearly_stats[selected_year]['age_pyramid']
            year_pop = yearly_stats[selected_year].get('total_population', 0)
            fig_pyramid = create_age_pyramid(pyramid_data, selected_year, year_pop)
            st.plotly_chart(fig_pyramid, use_container_width=True,
                            key=f"{key_prefix}_pyramid_chart")
        else:
            st.warning(f"Brak danych piramidy wieku dla roku {selected_year}")
    else:
        st.warning("Brak dostępnych statystyk rocznych")

    # --- Prewalencja chorób + wpływ RF ---
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Rozpowszechnienie chorób")
        fig_disease = create_disease_chart(results['disease_prevalence'])
        st.plotly_chart(fig_disease, use_container_width=True,
                        key=f"{key_prefix}_disease_chart")
    with col2:
        st.subheader("Wpływ czynników ryzyka")
        fig_rf = create_risk_factor_chart(results['rf_impact'])
        st.plotly_chart(fig_rf, use_container_width=True,
                        key=f"{key_prefix}_rf_chart")

    st.divider()

    # --- Trendy populacji ---
    st.subheader("Trendy populacji w czasie")
    fig_trends = create_trends_chart(yearly_stats)
    st.plotly_chart(fig_trends, use_container_width=True,
                    key=f"{key_prefix}_trends_chart")

    # --- Tabela szczegółowych statystyk ---
    st.subheader("Szczegółowe statystyki")
    stats_df = pd.DataFrame({
        'Metryka': [
            'Początkowa populacja', 'Populacja końcowa', 'Zmiana liczebności populacji',
            'Wskaźnik przeżycia (%)', 'Średni wiek (początkowy)', 'Średni wiek (końcowy)',
            'Przypadki chorób układu krążenia', 'Przypadki raka płuc', 'Wielochorobowość (%)',
        ],
        'Wartość': [
            f"{results['initial_pop']:,.0f}",
            f"{results['final_pop']:,.0f}",
            f"{results['final_pop'] - results['initial_pop']:+,.0f}",
            f"{(results['final_pop']/results['initial_pop']*100):.2f}%",
            f"{results['avg_age_initial']:.1f}",
            f"{results['avg_age_final']:.1f}",
            f"{results['cvd_count']:,.0f}",
            f"{results['lung_cancer_count']:,.0f}",
            f"{results['multimorbidity_pct']:.1f}%",
        ]
    })
    st.dataframe(stats_df, use_container_width=True, hide_index=True)

    # --- Tabela ustawień RF ---
    st.subheader("Zastosowane ustawienia czynników ryzyka")
    rf_df = pd.DataFrame({
        'Czynnik ryzyka': [RF_LABELS_PL.get(rf, rf.replace('_', ' ').title()) for rf in RF_NAMES],
        'Mnożnik':  [rf_multipliers.get(rf, 1.0) for rf in RF_NAMES],
        'Wpływ': [
            'Zmniejszony' if rf_multipliers.get(rf, 1.0) < 1 else
            'Zwiększony' if rf_multipliers.get(rf, 1.0) > 1 else
            'Podstawowy'
            for rf in RF_NAMES
        ]
    })
    st.dataframe(rf_df, use_container_width=True, hide_index=True)

    # --- Eksport (tylko z zakładki Simulation) ---
    if show_export and params is not None:
        st.subheader("Eksport wyników")
        export_col1, export_col2 = st.columns(2)
        with export_col1:
            results_json = serialize_results(results, params)
            st.download_button(
                label="Pobierz wyniki (JSON)",
                data=results_json,
                file_name=f"wyniki_symulacji_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True,
            )
        with export_col2:
            st.info("Eksport zawiera wszystkie dane potrzebne do odtworzenia "
                    "wyników w zakładce **Wczytaj dane** (piramidy roczne, "
                    "trendy, prewalencje, parametry).")


def main():
    """Main Streamlit application."""
    
    st.set_page_config(
        page_title="Symulator czynników ryzyka",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("Interaktywna symulacja czynników ryzyka")
    st.markdown("""
    Dostosuj mnożniki rozpowszechnienia czynników ryzyka i uruchom symulację demograficzną
    aby zobaczyć wpływ na zdrowie populacji. Domyślnie symulacja obejmuje 50 000 agentów i 50 lat.
    """)
    st.image("heatmap_gridsearch_full_abm_no_rf_20260511_114958.png")
    
    # =====================================================================
    # SIDEBAR CONTROLS
    # =====================================================================
    st.sidebar.header("Parametry symulacji")
    
    with st.sidebar:
        st.subheader("Ustawienia populacji")
        
        population_size = st.slider(
            "Wielkość populacji początkowej",
            min_value=1000,
            max_value=100000,
            value=50000,
            step=5000,
            help="Liczba agentów do zasymulowania"
        )
        
        simulation_years = st.slider(
            "Czas trwania symulacji (lata)",
            min_value=5,
            max_value=50,
            value=50,
            step=5,
            help="Liczba lat symulacji"
        )
        
        fertility_mult = st.slider(
            "Mnożnik płodności",
            min_value=0.4,
            max_value=2.5,
            value=1.0,
            step=0.01,
            help="Skoryguj wskaźnik urodzeń (1.0 = standard)"
        )
        
        mortality_mult = st.slider(
            "Mnożnik śmiertelności",
            min_value=0.3,
            max_value=1.6,
            value=1.0,
            step=0.01,
            help="Skoryguj wskaźnik zgonów (1.0 = standard)"
        )
        
        st.divider()

        # ---------------- PRESETS (przed suwakami — preset "prowadzi") ----------------
        st.subheader("Szybkie scenariusze")

        # Inicjalizacja session_state dla suwaków RF (default = 1.0 = podstawowy)
        for rf in RF_NAMES:
            key = f"rf_{rf}"
            if key not in st.session_state:
                st.session_state[key] = 1.0

        def _apply_scenario():
            """Callback on selectbox change — nadpisuje wartości suwaków w session_state."""
            chosen = st.session_state.get("scenario_select", "Niestandardowy")
            preset = SCENARIOS.get(chosen, {})
            for rf, value in preset.items():
                st.session_state[f"rf_{rf}"] = float(value)

        st.selectbox(
            "Wczytaj ustawienie:",
            options=list(SCENARIOS.keys()),
            key="scenario_select",
            on_change=_apply_scenario,
            help="Wybierz ustawienie, aby natychmiast ustawić wszystkie suwaki czynników ryzyka.",
        )

        st.divider()
        st.subheader("Dostosowanie czynników ryzyka")
        st.markdown("*Mnożniki: 1.0 = standard, <1.0 = zmniejszone, >1.0 = zwiększone*")

        # Suwaki RF — wartości synchronizowane przez session_state[key]
        rf_multipliers: Dict[str, float] = {}
        for rf in RF_NAMES:
            rf_multipliers[rf] = st.slider(
                RF_LABELS_PL.get(rf, rf.replace('_', ' ').title()),
                min_value=0.0,
                max_value=3.0,
                step=0.1,
                key=f"rf_{rf}",
                help=f"Mnożnik rozpowszechnienia dla {RF_LABELS_PL.get(rf, rf.replace('_', ' ').title())}",
            )

        st.divider()
        run_simulation = st.button(
            "Uruchom symulację",
            key="run_button",
            use_container_width=True,
            type="primary",
        )
    
    tabs = st.tabs(["Symulacja", "Mapa procesów", "Wczytaj dane"])

    # Renderujemy zakładki "Mapa procesów" oraz "Wczytaj dane" PRZED "Symulacja".
    # Powód: w bloku Symulacja są `return` (gdy brak wyników lub błąd),
    # które kończą main() — gdyby pozostałe zakładki były zdefiniowane po
    # Symulacja, nigdy nie zostałyby narysowane. Kolejność wizualna w UI
    # zależy od listy w st.tabs([...]), nie od kolejności bloków `with`.
    with tabs[1]:
        render_process_map_tab()

    with tabs[2]:
        render_load_data_tab()

    with tabs[0]:
        # Clear cached results if parameters change
        current_params = {
            'Rozmiar populacji': population_size,
            'Liczba lat symulacji': simulation_years,
            'Mnożnik płodności': fertility_mult,
            'Mnożnik śmiertelności': mortality_mult,
            'Mnożniki czynników ryzyka': rf_multipliers
        }
    
        if 'last_params' not in st.session_state:
            st.session_state['last_params'] = current_params
        elif st.session_state['last_params'] != current_params:
            # Parameters changed, clear cached results
            st.session_state['simulation_results'] = None
            st.session_state['last_params'] = current_params
    
        # =====================================================================
        # MAIN CONTENT
        # =====================================================================
    
        # Check if we have simulation results
        if 'simulation_results' not in st.session_state or st.session_state['simulation_results'] is None:
            if not run_simulation:
                st.info("""
                ### Jak korzystać:
                1. **Dostosuj parametry** w pasku bocznym (po lewej)
                2. **Wybierz mnożniki czynników ryzyka** (skala 0.0-3.0)
                3. **Kliknij „Uruchom symulację”** aby rozpocząć
                4. **Zobacz wyniki**: piramida wieku, rozkład chorób i analiza czynników ryzyka
            
                ### Co mierzy symulacja:
                - **Wzrost populacji**: populacja końcowa po okresie symulacji
                - **Piramida wieku**: rozkład demograficzny według wieku i płci
                - **Wpływ chorób**: rozpowszechnienie chorób układu krążenia i raka płuc
                - **Analiza czynników ryzyka**: wkład poszczególnych czynników ryzyka w obciążenie chorobami
                """)
                return
        
            # Run simulation
            st.session_state['run_count'] = st.session_state.get('run_count', 0) + 1
        
            with st.spinner("Uruchamianie symulacji... (może potrwać 1-2 minuty)"):
                results = run_simulation_with_rf(
                    population_size=population_size,
                    years=simulation_years,
                    fertility_multiplier=fertility_mult,
                    mortality_multiplier=mortality_mult,
                    rf_multipliers=rf_multipliers,
                    worker_count=mp.cpu_count()
                )
        
            if results is None:
                st.error("Symulacja nie powiodła się. Sprawdź parametry i spróbuj ponownie.")
                return
        
            # Store results in session state to prevent re-running on slider changes
            st.session_state['simulation_results'] = results
            st.success("Symulacja zakończona pomyślnie!")
        else:
            results = st.session_state['simulation_results']
    
        # =====================================================================
        # RESULTS DISPLAY — delegowane do funkcji wspólnej z zakładką
        # "Wczytaj dane" (render_results).
        # =====================================================================
        export_params = {
            'Rozmiar populacji':      population_size,
            'Liczba lat symulacji':                simulation_years,
            'Mnożnik płodności': fertility_mult,
            'Mnożnik śmiertelności': mortality_mult,
            'Mnożniki czynników ryzyka':         rf_multipliers,
        }
        render_results(results, rf_multipliers=rf_multipliers,
                       params=export_params, show_export=True,
                       key_prefix="sim")


# =====================================================================
# MAPA PROCESÓW — Sankey: Risk Factors → Choroby
#
# Wykres generowany bezpośrednio z DiseaseModel.HAZARD_BETA, więc zawsze
# odzwierciedla aktualny stan modelu (bez parsowania zewnętrznego HTML).
# Logika identyczna z funkcją create_sankey() w
# analiza_ABM_gridsearch/graf_ryzyko_choroby.py — bierzemy tylko górny
# panel z pełną szerokością.
# =====================================================================

RF_COLORS = {
    "smoking":              "#34495e",
    "obesity":              "#e67e22",
    "physical_inactivity":  "#95a5a6",
    "alcohol_abuse":        "#8e44ad",
    "high_cholesterol":     "#f39c12",
    "hypertension_stage0":  "#16a085",
    "family_history":       "#7f8c8d",
}

DISEASE_COLORS = {
    "CVD":         "#c0392b",
    "Lung Cancer": "#2c3e50",
}

DISEASE_LABELS_PL = {
    "CVD": "Choroby układu sercowo-naczyniowego",
    "Lung Cancer": "Rak płuc",
}


def build_rf_disease_sankey() -> go.Figure:
    """Zbuduj diagram Sankey: czynniki ryzyka → choroby na podstawie HAZARD_BETA."""
    dm = DiseaseModel()
    rfs = list(RF_LABELS_PL.keys())
    diseases = dm.diseases

    nodes = [RF_LABELS_PL[r] for r in rfs] + [DISEASE_LABELS_PL.get(d, d) for d in diseases]
    node_colors = (
        [RF_COLORS[r] for r in rfs]
        + [DISEASE_COLORS.get(d, "#7f8c8d") for d in diseases]
    )

    rf_to_idx = {r: i for i, r in enumerate(rfs)}
    disease_to_idx = {d: len(rfs) + i for i, d in enumerate(diseases)}

    source, target, value, label, link_colors = [], [], [], [], []
    for disease, beta_map in dm.HAZARD_BETA.items():
        for rf, beta in beta_map.items():
            if beta <= 0:
                continue
            hr = math.exp(beta)
            source.append(rf_to_idx[rf])
            target.append(disease_to_idx[disease])
            value.append(beta)
            label.append(f"HR={hr:.2f}, β={beta:.2f}")
            c = RF_COLORS[rf]
            r_, g_, b_ = int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16)
            link_colors.append(f"rgba({r_},{g_},{b_},0.45)")

    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(
            pad=15, 
            thickness=22,
            line=dict(color="white", width=1),
            label=nodes,
            color=node_colors,
            hovertemplate="%{label}<extra></extra>",
            
            # --- KLUCZOWA ZMIANA: Wymuszenie stylu czcionki dla węzłów ---
            font=dict(color="black", size=14, family="Arial Black, Arial, sans-serif")
        ),
        link=dict(
            source=source, target=target, value=value,
            label=label, color=link_colors,
            hovertemplate="%{source.label} → %{target.label}<br>%{label}<extra></extra>",
        ),
    ))
    
    fig.update_layout(
        title=dict(
            text=(
                "<b>Sankey: Przepływ ryzyka (szerokość ∝ β = ln HR)</b><br>"
                "<sub>Współczynniki β bezpośrednio z DiseaseModel.HAZARD_BETA</sub>"
            ),
            x=0.5, xanchor="center", 
            font=dict(size=18, color="black"), # Wyraźny tytuł
        ),
        height=600,
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=40, r=40, t=90, b=40), # Zwiększone marginesy na dłuższe napisy chorób
    )
    return fig

def build_edges_dataframe() -> pd.DataFrame:
    """Tabela powiązań czynnik ryzyka → choroba z HR i β."""
    dm = DiseaseModel()
    rows = []
    for disease, beta_map in dm.HAZARD_BETA.items():
        for rf, beta in beta_map.items():
            if beta <= 0:
                continue
            rows.append({
                "Czynnik ryzyka": RF_LABELS_PL.get(rf, rf.replace('_', ' ').title()),
                "Choroba": DISEASE_LABELS_PL.get(disease, disease),
                "HR": round(math.exp(beta), 2),
                "β = ln(HR)": round(beta, 3),
            })
    df = pd.DataFrame(rows).sort_values("HR", ascending=False).reset_index(drop=True)
    return df


def render_process_map_tab():
    st.header("Mapa procesów")
    st.markdown(
        "Diagram Sankey pokazuje przepływ ryzyka od **czynników ryzyka** (lewa "
        "strona) do **chorób** (prawa strona). Szerokość każdego strumienia jest "
        "proporcjonalna do współczynnika **β = ln(HR)** z modelu Coxa."
    )

    fig = build_rf_disease_sankey()
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Tabela powiązań (HR)", expanded=False):
        df = build_edges_dataframe()
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.caption(
            "HR > 1 oznacza zwiększone ryzyko. Palenie ma dominujący wpływ na "
            "raka płuc (HR=15)."
        )


# =====================================================================
# ZAKŁADKA "WCZYTAJ DANE" — uploader JSON + render wyników
# =====================================================================

def render_load_data_tab():
    st.header("Wczytaj dane symulacji")
    st.markdown(
        "Wgraj plik **JSON** wyeksportowany z zakładki *Symulacja* "
        "(`Pobierz wyniki (JSON)`). Po wczytaniu zobaczysz te same wykresy "
        "i statystyki — piramidę wieku z suwakiem rocznym, prewalencje, "
        "trendy populacji, tabele."
    )

    uploaded = st.file_uploader(
        "Plik JSON z eksportu",
        type=["json"],
        key="load_file_uploader",
        help="Plik wygenerowany przyciskiem 'Pobierz wyniki (JSON)' w zakładce Symulacja.",
    )

    if uploaded is None:
        st.info("Wgraj plik JSON, aby zobaczyć wyniki.")
        return

    try:
        raw = uploaded.read().decode("utf-8")
        payload = json.loads(raw)
        params, results = deserialize_results(payload)
    except json.JSONDecodeError as exc:
        st.error(f"Nie udało się sparsować JSON: {exc}")
        return
    except Exception as exc:
        st.error(f"Błąd podczas wczytywania pliku: {exc}")
        return

    schema_version = payload.get("schema_version", 0)
    if schema_version < 1:
        st.warning(
            "Plik pochodzi ze starszej wersji aplikacji (brak `schema_version`). "
            "Niektóre sekcje mogą być niepełne."
        )

    timestamp = payload.get("timestamp", "—")
    st.success(f"Wczytano dane z **{timestamp}** (plik: `{uploaded.name}`)")

    with st.expander("Parametry symulacji", expanded=False):
        if params:
            param_df = pd.DataFrame({
                "Parametr": list(params.keys()),
                "Wartość":  [
                    json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else str(v)
                    for v in params.values()
                ],
            })
            st.dataframe(param_df, use_container_width=True, hide_index=True)
        else:
            st.info("Brak parametrów w pliku.")

    st.divider()

    # Renderuj wyniki — ta sama funkcja, co używana w zakładce Symulacja
    rf_mults = params.get("risk_factors", {rf: 1.0 for rf in RF_NAMES})
    render_results(results, rf_multipliers=rf_mults, params=params,
                   show_export=False, key_prefix="load")

# ========================================================================
# HELPER FUNCTIONS
# ========================================================================

def run_simulation_with_rf(
    population_size: int,
    years: int,
    fertility_multiplier: float,
    mortality_multiplier: float,
    rf_multipliers: Dict[str, float],
    worker_count: int = 1,
) -> Dict:
    """
    Uruchom symulację z dostosowanymi czynnikami ryzyka.
    
    Args:
        population_size: liczba agentów na start
        years: czas trwania symulacji w latach
        fertility_multiplier: mnożnik płodności
        mortality_multiplier: mnożnik śmiertelności
        rf_multipliers: słownik mnożników RF
    
    Returns:
        Słownik wyników lub None w przypadku błędu
    """
    try:
        # Create disease model
        disease_model = DiseaseModel()
        
        # Create engine
        engine = SimulationEngine(
            disease_model=disease_model,
            seed=42
        )
        engine.parallel_workers = max(1, min(worker_count, mp.cpu_count()))
        
        # Set parameters
        engine.fertility_rate = fertility_multiplier
        engine.mortality_multiplier = mortality_multiplier
        
        # Modify risk factor probabilities in the engine's initialization method
        # We'll do this by patching the initialization
        original_init_rf = engine._init_risk_factors
        
        def modified_init_rf(citizen):
            """Modified RF initialization with multipliers."""
            rf = original_init_rf(citizen)
            
            # Apply multipliers to RF probabilities during initialization
            # (Note: current implementation initializes RFs based on age, 
            #  we modify them here with a second pass)
            
            # Re-initialize with multiplied probabilities
            age_years = citizen.age_years
            
            if age_years >= 15:
                # Smoking
                smoking_prob = 0.0
                if 20 <= age_years <= 70:
                    peak_age = 45
                    smoking_prob = 0.25 * (1 - ((age_years - peak_age) ** 2) / (50 ** 2))
                    smoking_prob = max(smoking_prob, 0.10)
                smoking_prob *= rf_multipliers.get("smoking", 1.0)
                if engine.rng.random() < min(smoking_prob, 1.0):
                    rf["smoking"] = 1
                
                # Obesity
                obesity_prob = 0.15 + (age_years - 20) * 0.008 if age_years > 20 else 0.05
                obesity_prob = min(obesity_prob, 0.45) * rf_multipliers.get("obesity", 1.0)
                if engine.rng.random() < min(obesity_prob, 1.0):
                    rf["obesity"] = 1
                
                # Physical inactivity
                inactivity_prob = 0.2 + (age_years - 20) * 0.005 if age_years > 20 else 0.1
                inactivity_prob *= rf_multipliers.get("physical_inactivity", 1.0)
                if engine.rng.random() < min(inactivity_prob, 1.0):
                    rf["physical_inactivity"] = 1
                
                # Alcohol
                alcohol_prob = (0.08 if 20 <= age_years <= 65 else 0.02) * rf_multipliers.get("alcohol_abuse", 1.0)
                if engine.rng.random() < min(alcohol_prob, 1.0):
                    rf["alcohol_abuse"] = 1
                
                # Cholesterol
                cholesterol_prob = ((age_years - 20) * 0.006 if age_years > 20 else 0.01) * rf_multipliers.get("high_cholesterol", 1.0)
                if engine.rng.random() < min(cholesterol_prob, 1.0):
                    rf["high_cholesterol"] = 1
                
                # Hypertension
                hypertension_prob = ((age_years - 30) * 0.008 if age_years > 30 else 0.01) * rf_multipliers.get("hypertension_stage0", 1.0)
                if engine.rng.random() < min(hypertension_prob, 1.0):
                    rf["hypertension_stage0"] = 1
                
                # Family history
                family_prob = 0.15 * rf_multipliers.get("family_history", 1.0)
                if engine.rng.random() < min(family_prob, 1.0):
                    rf["family_history"] = 1
            
            return rf
        
        engine._init_risk_factors = modified_init_rf
        
        # Create population
        engine._create_synthetic_population(population_size)
        initial_pop = len([c for c in engine.citizens.values() if c.alive])
        avg_age_initial = np.mean([c.age_years for c in engine.citizens.values() if c.alive])
        
        # Run simulation
        engine.run(months=years * 12)
        
        # Collect results
        final_citizens = [c for c in engine.citizens.values() if c.alive]
        final_pop = len(final_citizens)
        avg_age_final = np.mean([c.age_years for c in final_citizens]) if final_pop > 0 else 0
        
        # Get disease counts
        cvd_count = sum(1 for c in final_citizens if c.diseases.get("CVD", 0) == 1)
        lung_cancer_count = sum(1 for c in final_citizens if c.diseases.get("Lung Cancer", 0) == 1)
        multimorbidity_count = sum(1 for c in final_citizens if c.num_conditions() >= 2)
        multimorbidity_pct = (multimorbidity_count / final_pop * 100) if final_pop > 0 else 0
        
        # Build pyramid
        pyramid = build_age_pyramid(final_citizens)
        
        # Calculate RF impact
        rf_impact = calculate_rf_impact(final_citizens)
        
        # Disease prevalence
        disease_prev = {
            'CVD': (cvd_count / final_pop * 100) if final_pop > 0 else 0,
            'Lung Cancer': (lung_cancer_count / final_pop * 100) if final_pop > 0 else 0
        }
        
        # Deaths and births
        deaths = initial_pop - final_pop + len([c for c in engine.citizens.values() if c.alive])
        # Approximate births from population change and known deaths
        births = final_pop - initial_pop + deaths
        
        return {
            'initial_pop': initial_pop,
            'final_pop': final_pop,
            'avg_age_initial': avg_age_initial,
            'avg_age_final': avg_age_final,
            'deaths': deaths,
            'births': max(births, 0),
            'cvd_count': cvd_count,
            'lung_cancer_count': lung_cancer_count,
            'multimorbidity_pct': multimorbidity_pct,
            'disease_prevalence': disease_prev,
            'rf_impact': rf_impact,
            'final_pyramid': pyramid,
            'yearly_stats': engine.yearly_stats
        }
    
    except Exception as e:
        st.error(f"Błąd podczas symulacji: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None


def build_age_pyramid(citizens: List[Citizen]) -> Dict[str, Dict[str, int]]:
    """Build age pyramid data."""
    age_groups = ['0-4', '5-9', '10-14', '15-19', '20-24', '25-29', '30-34', '35-39',
                  '40-44', '45-49', '50-54', '55-59', '60-64', '65-69', '70-74', '75-79',
                  '80-84', '85-89', '90+']
    
    pyramid = {group: {'male': 0, 'female': 0} for group in age_groups}
    
    for citizen in citizens:
        age = int(citizen.age_years)
        
        if age < 5:
            group = '0-4'
        elif age < 10:
            group = '5-9'
        elif age < 15:
            group = '10-14'
        else:
            group_idx = (age - 15) // 5
            if group_idx < len(age_groups) - 1:
                start = 15 + group_idx * 5
                group = f'{start}-{start+4}'
            else:
                group = '90+'
        
        if group in pyramid:
            pyramid[group][citizen.sex] += 1
    
    return pyramid


def create_age_pyramid(pyramid: Dict[str, Dict[str, int]], year: int = None, population: int = None) -> go.Figure:
    """Utwórz wizualizację piramidy wieku.
    
    Args:
        pyramid: słownik grup wiekowych z wartościami {'male': count, 'female': count}
        year: opcjonalny rok do wyświetlenia w tytule
        population: opcjonalna liczba populacji do wyświetlenia w tytule
    """
    ages = list(pyramid.keys())
    males = [-pyramid[age]['male'] for age in ages]
    females = [pyramid[age]['female'] for age in ages]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=ages,
        x=males,
        orientation='h',
        name='Mężczyźni',
        marker=dict(color='#3498db')
    ))
    
    fig.add_trace(go.Bar(
        y=ages,
        x=females,
        orientation='h',
        name='Kobiety',
        marker=dict(color='#e74c3c')
    ))
    
    # Zbuduj tytuł z opcjonalnym rokiem i liczbą populacji
    title = 'Piramida wieku populacji'
    if year is not None:
        title = f'Piramida wieku populacji - rok {year}'
    if population is not None and population > 0:
        title += f' (n={population:,})'
    
    fig.update_layout(
        barmode='overlay',
        title=title,
        xaxis_title='Liczba osób',
        yaxis_title='Grupa wiekowa',
        height=500,
        hovermode='closest'
    )
    
    return fig


def create_disease_chart(disease_prev: Dict[str, float]) -> go.Figure:
    """Utwórz wykres prewalencji chorób."""
    disease_labels = [DISEASE_LABELS_PL.get(d, d) for d in disease_prev.keys()]
    fig = go.Figure(data=[
        go.Bar(
            x=disease_labels,
            y=list(disease_prev.values()),
            marker=dict(color=['#e74c3c', '#f39c12'])
        )
    ])
    
    fig.update_layout(
        title='Rozpowszechnienie chorób (%)',
        xaxis_title='Choroba',
        yaxis_title='Rozpowszechnienie (%)',
        height=400,
        showlegend=False
    )
    
    return fig


def create_risk_factor_chart(rf_impact: Dict[str, float]) -> go.Figure:
    """Utwórz wykres wpływu czynników ryzyka."""
    rfs = [RF_LABELS_PL.get(rf, rf.replace('_', ' ').title()) for rf in rf_impact.keys()]
    impacts = list(rf_impact.values())
    
    fig = go.Figure(data=[
        go.Bar(
            x=rfs,
            y=impacts,
            marker=dict(color='#16a085')
        )
    ])
    
    fig.update_layout(
        title='Wpływ czynników ryzyka na choroby',
        xaxis_title='Czynnik ryzyka',
        yaxis_title='Relatywny wpływ',
        height=400,
        xaxis=dict(tickangle=-45),
        showlegend=False
    )
    
    return fig


def create_trends_chart(yearly_stats: Dict) -> go.Figure:
    """Utwórz wykres trendów populacyjnych."""
    if not yearly_stats:
        return go.Figure().add_annotation(text="Brak dostępnych danych")
    
    years = sorted(yearly_stats.keys())
    populations = [yearly_stats[y].get('total_population', 0) for y in years]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=years,
        y=populations,
        mode='lines+markers',
        name='Populacja',
        line=dict(color='#3498db', width=3),
        marker=dict(size=5)
    ))
    
    fig.update_layout(
        title='Wzrost populacji w czasie',
        xaxis_title='Rok',
        yaxis_title='Populacja',
        height=400,
        hovermode='x unified'
    )
    
    return fig


def calculate_rf_impact(citizens: List[Citizen]) -> Dict[str, float]:
    """Oblicz względny wpływ każdego czynnika ryzyka na obciążenie chorobowe."""
    rf_names = Citizen.DEFAULT_RISK_FACTORS
    
    # Count RF presence
    rf_counts = {rf: 0 for rf in rf_names}
    rf_disease_burden = {rf: 0 for rf in rf_names}
    
    for citizen in citizens:
        disability = citizen.disability_score
        
        for rf, value in citizen.risk_factors.items():
            if value == 1:
                rf_counts[rf] += 1
                rf_disease_burden[rf] += disability
    
    total_citizens = len(citizens) if citizens else 1
    
    # Normalize to relative impact
    rf_impact = {}
    for rf in rf_names:
        if rf_counts[rf] > 0:
            rf_impact[rf] = rf_disease_burden[rf] / total_citizens
        else:
            rf_impact[rf] = 0
    
    return rf_impact


if __name__ == "__main__":
    main()
