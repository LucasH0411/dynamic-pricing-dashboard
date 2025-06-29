import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Dynamic Pricing - ShopTrend24", layout="wide")

st.markdown(
    """
    <style>

        :root {
            --primary-color: #004A99;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Grundparameter
BASE_PRICE = 109.99
WEEKS = 12
BASE_DEMAND = 1500
PRICE_ELASTICITY = -2.0
UNIT_COST = 20
FIXED_COSTS = 10000
INITIAL_CUSTOMERS = 1000
COMPETITION_INTENSITY = 0.4
COMPETITION_SENSITIVITY = 0.2
CHURN_SENSITIVITY = 0.3
SAMPLES = 100
COMPETITOR_VOLATILITY = 0.05  # Zufällige Schwankung des Wettbewerbs
DEMAND_VOLATILITY = 0.1  # Zufällige Nachfrageschwankung pro Woche

# ------------------------- Simulation ----------------------------

def simulate(strategy: str, base_price: float, seed: int | None = None) -> pd.DataFrame:
    """Simulate weekly performance for the chosen pricing strategy.

    * **Entscheidungstheoretisch** – wählt aus einem festen Preisraster den
     jenigen Preis mit maximaler Gewinnerwartung, basierend auf zufällig
     simulierten Wettbewerberpreisen.
    * **Kundenbasiert** – passt den Preis mithilfe des bisherigen
     Nachfrageverlaufs an und reagiert damit auf Veränderungen der
     Kundennachfrage.
    * **Wettbewerbsanpassung** – orientiert sich am aktuellen Marktpreis.

    Übersteigt unser Preis dauerhaft den Wettbewerb, sinkt die Kundenbasis
    gemäß eines einfachen Churn-Modells. Zusätzlich schwankt die Nachfrage
    jede Woche zufällig, um realistische Marktbedingungen abzubilden.
    """
    rng = np.random.default_rng(seed)

    time = np.arange(1, WEEKS + 1)
    price = np.zeros(WEEKS)
    demand = np.zeros(WEEKS)
    revenue = np.zeros(WEEKS)
    profit = np.zeros(WEEKS)
    cumulative_profit = np.zeros(WEEKS)
    competitor_price = np.zeros(WEEKS)
    customers = np.zeros(WEEKS)

    price[0] = base_price
    # Wettbewerber folgt einem stochastischen Pfad, der nicht vom eigenen
    # Startpreis beeinflusst wird. Dadurch lassen sich unterschiedliche
    # Preisniveaus fair vergleichen.
    competitor_price[0] = BASE_PRICE * rng.uniform(0.95, 1.05)
    customers[0] = INITIAL_CUSTOMERS

    price_min = base_price * 0.8
    price_max = base_price * 1.2

    competitor_min = BASE_PRICE * 0.8
    competitor_max = BASE_PRICE * 1.2

    for t in range(WEEKS):
        if t > 0:
            competitor_price[t] = np.clip(
                competitor_price[t-1]
                * rng.uniform(
                    0.95 - COMPETITION_INTENSITY * COMPETITOR_VOLATILITY,
                    1.05 + COMPETITION_INTENSITY * COMPETITOR_VOLATILITY,
                ),
                competitor_min,
                competitor_max,
            )
        else:
            competitor_price[t] = competitor_price[0]

        if t == 0:
            # In Woche eins bleibt der festgelegte Startpreis bestehen
            pass
        elif strategy == "Entscheidungstheoretisch":
            # Dynamisches Preisraster um den zuletzt gewählten Preis
            price_grid = np.linspace(
                max(price_min, price[t-1] * 0.95),
                min(price_max, price[t-1] * 1.05),
                9,
            )
            competitor_samples = np.clip(
                competitor_price[t]
                * rng.uniform(
                    0.95 - COMPETITION_INTENSITY * COMPETITOR_VOLATILITY,
                    1.05 + COMPETITION_INTENSITY * COMPETITOR_VOLATILITY,
                    size=SAMPLES,
                ),
                price_min,
                price_max,
            )
            expected_profits = []
            for p in price_grid:
                price_factor = (p / base_price) ** PRICE_ELASTICITY
                competition_factor = 1 - COMPETITION_SENSITIVITY * np.maximum(0, p - competitor_samples) / p
                demand_samples = (
                    BASE_DEMAND
                    * price_factor
                    * competition_factor
                    * (customers[t-1] / INITIAL_CUSTOMERS if t > 0 else 1)
                    * rng.normal(1, DEMAND_VOLATILITY, size=SAMPLES)
                )
                profits = (p - UNIT_COST) * demand_samples - FIXED_COSTS / WEEKS
                expected_profits.append(np.mean(profits))
            best_idx = int(np.argmax(expected_profits))
            price[t] = price_grid[best_idx]
        elif strategy == "Kundenbasiert":
            demand_ratio = demand[t-1] / BASE_DEMAND
            adjustment = 0.05 * (demand_ratio - 1)
            price[t] = np.clip(
                price[t-1] * (1 + adjustment), price_min, price_max
            )
        else:  # Wettbewerbsanpassung
            price[t] = np.clip(competitor_price[t] * 0.98, price_min, price_max)

        price_factor = (price[t] / base_price) ** PRICE_ELASTICITY
        competition_factor = 1 - COMPETITION_SENSITIVITY * max(0, price[t] - competitor_price[t]) / price[t]
        demand_shock = rng.normal(1, DEMAND_VOLATILITY)
        demand[t] = (
            BASE_DEMAND
            * price_factor
            * competition_factor
            * (customers[t-1] / INITIAL_CUSTOMERS if t > 0 else 1)
            * demand_shock
        )
        demand[t] = max(demand[t], 0)
        revenue[t] = price[t] * demand[t]
        profit[t] = (price[t] - UNIT_COST) * demand[t] - FIXED_COSTS / WEEKS
        cumulative_profit[t] = profit[t] if t == 0 else cumulative_profit[t-1] + profit[t]

        prev_cust = customers[t-1] if t > 0 else INITIAL_CUSTOMERS
        if price[t] > competitor_price[t]:
            change = CHURN_SENSITIVITY * (price[t] - competitor_price[t]) / price[t] * prev_cust
            customers[t] = max(0, prev_cust - change)
        else:
            change = CHURN_SENSITIVITY * (competitor_price[t] - price[t]) / competitor_price[t] * prev_cust
            customers[t] = prev_cust + change

    return pd.DataFrame({
        "Woche": time,
        "Eigener Preis (EUR)": price,
        "Wettbewerbspreis (EUR)": competitor_price,
        "Nachfrage": demand,
        "Umsatz (EUR)": revenue,
        "Gewinn (EUR)": profit,
        "Kumul. Gewinn (EUR)": cumulative_profit,
        "Kunden": customers,
    })


# ------------------------- UI Helper -----------------------------

def kpi_columns(df: pd.DataFrame):
    k1, k2, k3 = st.columns(3)
    k1.metric("Gesamtumsatz (EUR)", f"{df['Umsatz (EUR)'].sum():,.2f}")
    k2.metric("Kumul. Gewinn (EUR)", f"{df['Kumul. Gewinn (EUR)'].iloc[-1]:,.2f}")
    k3.metric("Kundenbasis", f"{df['Kunden'].iloc[-1]:,.0f}")


def show_charts(df: pd.DataFrame):
    time = df["Woche"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("ShopTrend24 Simulationsergebnisse", fontsize=14)

    axes[0, 0].plot(time, df["Eigener Preis (EUR)"], label="ShopTrend24", color="#004A99")
    axes[0, 0].plot(time, df["Wettbewerbspreis (EUR)"], label="Wettbewerber", color="tab:gray")
    axes[0, 0].set_title("Preisentwicklung")
    axes[0, 0].set_xlabel("Woche")
    axes[0, 0].set_ylabel("Preis (EUR)")
    axes[0, 0].legend()

    axes[0, 1].plot(time, df["Nachfrage"], color="tab:orange")
    axes[0, 1].set_title("Nachfrage")
    axes[0, 1].set_xlabel("Woche")
    axes[0, 1].set_ylabel("Nachfrage")

    axes[1, 0].plot(time, df["Gewinn (EUR)"], color="#004A99")
    axes[1, 0].set_title("Gewinn pro Woche")
    axes[1, 0].set_xlabel("Woche")
    axes[1, 0].set_ylabel("EUR")

    axes[1, 1].plot(time, df["Kumul. Gewinn (EUR)"], color="#004A99")
    axes[1, 1].set_title("Kumulierter Gewinn")
    axes[1, 1].set_xlabel("Woche")
    axes[1, 1].set_ylabel("EUR")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    st.pyplot(fig)



# ------------------------- Pages --------------------------------

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Seite",
    ["Simulation", "Sensitivitätsanalyse"],
    key="page_select",
)

if st.session_state.get("last_page") != page:
    if page == "Sensitivitätsanalyse":
        st.session_state.pop("price_delta", None)
    st.session_state["last_page"] = page
strategy = st.sidebar.selectbox(
    "Pricing-Agent",
    ["Entscheidungstheoretisch", "Kundenbasiert", "Wettbewerbsanpassung"],
    key="strategy_select",
)

def main_page():
    st.title("Dynamic Pricing - ShopTrend24")
    with st.expander("Wie funktioniert der Pricing-Agent?"):
        st.markdown(
            """
            **Entscheidungstheoretisch** – Für jedes Preisniveau wird der
            erwartete Gewinn auf Basis zufälliger Wettbewerberpreise berechnet.
            Der gewinnmaximierende Preis wird gewählt. Liegt er zu weit über dem
            Marktpreis, wandern Kunden ab.

            **Kundenbasiert** – Die Preise orientieren sich am bisherigen
            Nachfrageverlauf. Steigt die Nachfrage, erhöht sich auch der Preis;
            sinkt sie, wird der Preis reduziert.

            **Wettbewerbsanpassung** – Der Preis orientiert sich am aktuellen
            Wettbewerbsniveau und liegt geringfügig darunter.

            Die Nachfrage reagiert auf Preis, Wettbewerb und Abwanderung;
            gesteuert werden kann letztlich nur der eigene Preis.
            """
        )
    df = simulate(strategy, BASE_PRICE)
    kpi_columns(df)
    st.subheader("Entwicklungen")
    show_charts(df)
    st.subheader("Detailtabelle")
    st.dataframe(df.set_index("Woche").style.format({
        "Eigener Preis (EUR)": "{:.2f}",
        "Wettbewerbspreis (EUR)": "{:.2f}",
        "Umsatz (EUR)": "{:.2f}",
        "Gewinn (EUR)": "{:.2f}",
        "Kumul. Gewinn (EUR)": "{:.2f}",
        "Kunden": "{:.0f}",
    }))

def sensitivity_page():
    st.title("Sensitivitätsanalyse")
    delta = st.sidebar.slider(
        "Preisänderung (%)",
        -20,
        20,
        0,
        step=1,
        key="price_delta",
    )
    base_df = simulate(strategy, BASE_PRICE, seed=42)
    test_df = simulate(strategy, BASE_PRICE * (1 + delta / 100), seed=42)
    k1, k2, k3 = st.columns(3)
    k1.metric(
        "Gesamtumsatz (EUR)",
        f"{test_df['Umsatz (EUR)'].sum():,.2f}",
        f"{test_df['Umsatz (EUR)'].sum() - base_df['Umsatz (EUR)'].sum():+.2f}"
    )
    k2.metric(
        "Kumul. Gewinn (EUR)",
        f"{test_df['Kumul. Gewinn (EUR)'].iloc[-1]:,.2f}",
        f"{test_df['Kumul. Gewinn (EUR)'].iloc[-1] - base_df['Kumul. Gewinn (EUR)'].iloc[-1]:+.2f}"
    )
    k3.metric(
        "Kundenbasis", f"{test_df['Kunden'].iloc[-1]:,.0f}",
        f"{test_df['Kunden'].iloc[-1] - base_df['Kunden'].iloc[-1]:+.0f}"
    )
    st.subheader("Entwicklungen")
    show_charts(test_df)

if page == "Simulation":
    main_page()
else:  # Sensitivitätsanalyse
    sensitivity_page()
