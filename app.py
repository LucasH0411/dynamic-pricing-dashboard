import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Dynamic Pricing - ShopTrend24", layout="wide")
st.title("Dynamic Pricing - ShopTrend24")

st.markdown(
    """
     <style>
        .main { background-color: #ffffff; }
        .block-container { padding-top: 2rem; }
        .stMetric { background-color: #f8f9fa; padding: 20px; border-radius: 10px; border: 1px solid #dee2e6; }
    </style>
    """,
    unsafe_allow_html=True
)

# Fixe Marktparameter
base_price = 109.99
weeks = 12
base_demand = 1500
price_elasticity = -2.0
unit_cost = 40
fixed_costs = 10000
initial_customers = 1000

# Steuerbare Wettbewerbsintensität
st.sidebar.header("Marktumfeld")
competition_intensity = st.sidebar.slider(
    "Wettbewerbsintensität (0 = kein Druck, 1 = hoher Druck)", 0.0, 1.0, 0.4, step=0.05,
    help="Je höher, desto häufiger greifen Wettbewerber mit Preissenkungen an."
)

churn_sensitivity = 0.3

# Initialisierung
time = np.arange(1, weeks + 1)
price = np.zeros(weeks)
demand = np.zeros(weeks)
revenue = np.zeros(weeks)
profit = np.zeros(weeks)
cumulative_profit = np.zeros(weeks)
competitor_price = np.zeros(weeks)
customers = np.zeros(weeks)

price[0] = base_price
competitor_price[0] = base_price * np.random.uniform(0.95, 1.05)
customers[0] = initial_customers

# Entscheidungstheoretisches Modell
# Diskrete Preisoptionen für die Optimierung
price_grid = np.linspace(base_price * 0.8, base_price * 1.2, 9)
competition_sensitivity = 0.2
samples = 100

# Simulation mit erwartungswertbasiertem Preisentscheid
for t in range(weeks):
    # Wettbewerbsunsicherheit durch Stichprobe abbilden
    competitor_samples = base_price * np.random.uniform(
        0.9 - competition_intensity * 0.1, 1.05, size=samples
    )

    # Erwarteten Gewinn für jede Preisoption berechnen
    expected_profits = []
    for p in price_grid:
        price_factor = (p / base_price) ** price_elasticity
        competition_factor = 1 - competition_sensitivity * np.maximum(0, p - competitor_samples) / p
        demand_samples = (
            base_demand
            * price_factor
            * competition_factor
            * (customers[t-1] / initial_customers if t > 0 else 1)
        )
        profits = (p - unit_cost) * demand_samples - fixed_costs / weeks
        expected_profits.append(np.mean(profits))

    # Preis mit maximalem Erwartungswert wählen
    best_idx = int(np.argmax(expected_profits))
    price[t] = price_grid[best_idx]
    competitor_price[t] = np.mean(competitor_samples)

    # Tatsächliche Nachfrage und Gewinne berechnen
    price_factor = (price[t] / base_price) ** price_elasticity
    competition_factor = 1 - competition_sensitivity * np.maximum(0, price[t] - competitor_price[t]) / price[t]
    demand[t] = (
        base_demand
        * price_factor
        * competition_factor
        * (customers[t-1] / initial_customers if t > 0 else 1)
    )

    # Umsatz und Gewinn
    revenue[t] = price[t] * demand[t]
    profit[t] = (price[t] - unit_cost) * demand[t] - fixed_costs / weeks
    cumulative_profit[t] = profit[t] if t == 0 else cumulative_profit[t-1] + profit[t]

    # Kundenabwanderung bei zu hohem Preis
    overpricing = max(0, price[t] - competitor_price[t]) / price[t]
    churn = churn_sensitivity * overpricing * (customers[t-1] if t > 0 else initial_customers)
    customers[t] = max(0, (customers[t-1] if t > 0 else initial_customers) - churn)

# DataFrame
df = pd.DataFrame({
    "Woche": time,
    "Eigener Preis (EUR)": price,
    "Wettbewerbspreis (EUR)": competitor_price,
    "Nachfrage": demand,
    "Umsatz (EUR)": revenue,
    "Gewinn (EUR)": profit,
    "Kumul. Gewinn (EUR)": cumulative_profit,
    "Kunden": customers
})

# KPIs
st.subheader("Strategisches Ziel: Gewinnoptimierung")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Gesamtumsatz (EUR)", f"{revenue.sum():,.2f}")
k2.metric("Kumul. Gewinn (EUR)", f"{cumulative_profit[-1]:,.2f}")
k3.metric("Kundenbasis", f"{customers[-1]:,.0f}")
k4.metric("Wettbewerbsdruck", f"{competition_intensity * 100:.0f}%")

# Visualisierungen
st.subheader("Entwicklungen")
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

axes[0, 0].plot(time, price, label="Eigener Preis")
axes[0, 0].plot(time, competitor_price, label="Wettbewerber")
axes[0, 0].set_xlabel("Woche")
axes[0, 0].set_ylabel("Preis (EUR)")
axes[0, 0].legend()

axes[0, 1].plot(time, demand, color="tab:orange")
axes[0, 1].set_xlabel("Woche")
axes[0, 1].set_ylabel("Nachfrage")

axes[1, 0].plot(time, profit, label="Gewinn")
axes[1, 0].set_xlabel("Woche")
axes[1, 0].set_ylabel("EUR")
axes[1, 0].legend()

axes[1, 1].plot(time, cumulative_profit, label="Kumul. Gewinn")
axes[1, 1].set_xlabel("Woche")
axes[1, 1].set_ylabel("EUR")
axes[1, 1].legend()

plt.tight_layout()
st.pyplot(fig)

# Detailtabelle
st.subheader("Detailtabelle")
st.dataframe(df.set_index("Woche").style.format({
    "Eigener Preis (EUR)": "{:.2f}",
    "Wettbewerbspreis (EUR)": "{:.2f}",
    "Umsatz (EUR)": "{:.2f}",
    "Gewinn (EUR)": "{:.2f}",
    "Kumul. Gewinn (EUR)": "{:.2f}",
    "Kunden": "{:.0f}"
}))
