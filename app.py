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

review_sensitivity = 0.3
review_decay = 0.1
churn_sensitivity = 0.4

# Initialisierung
time = np.arange(1, weeks + 1)
price = np.zeros(weeks)
demand = np.zeros(weeks)
revenue = np.zeros(weeks)
profit = np.zeros(weeks)
cumulative_profit = np.zeros(weeks)
review_score = np.zeros(weeks)
competitor_price = np.zeros(weeks)

review_score[0] = 4.5
price[0] = base_price
competitor_price[0] = base_price * np.random.uniform(0.95, 1.05)

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
        demand_samples = (
            base_demand
            * price_factor
            * (1 - competition_sensitivity * np.maximum(0, p - competitor_samples) / p)
        )
        profits = (p - unit_cost) * demand_samples - fixed_costs / weeks
        expected_profits.append(np.mean(profits))

    # Preis mit maximalem Erwartungswert wählen
    best_idx = int(np.argmax(expected_profits))
    price[t] = price_grid[best_idx]
    competitor_price[t] = np.mean(competitor_samples)

    # Tatsächliche Nachfrage und Gewinne berechnen
    price_factor = (price[t] / base_price) ** price_elasticity
    demand[t] = (
        base_demand
        * price_factor
        * (1 - competition_sensitivity * np.maximum(0, price[t] - competitor_price[t]) / price[t])
    )

    # Umsatz, Gewinn, Bewertung
    revenue[t] = price[t] * demand[t]
    profit[t] = (price[t] - unit_cost) * demand[t] - fixed_costs / weeks
    cumulative_profit[t] = profit[t] if t == 0 else cumulative_profit[t-1] + profit[t]

    # Konsumentenreaktion
    overpricing = max(0, price[t] - competitor_price[t]) / price[t]
    review_score[t] = max(1.0, (review_score[t-1] - review_sensitivity * overpricing + review_decay))

# DataFrame
df = pd.DataFrame({
    "Woche": time,
    "Eigener Preis (EUR)": price,
    "Wettbewerbspreis (EUR)": competitor_price,
    "Nachfrage": demand,
    "Umsatz (EUR)": revenue,
    "Gewinn (EUR)": profit,
    "Kumul. Gewinn (EUR)": cumulative_profit,
    "Ø Bewertung": review_score
})

# KPIs
st.subheader("Strategisches Ziel: Gewinnoptimierung")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Gesamtumsatz (EUR)", f"{revenue.sum():,.2f}")
k2.metric("Kumul. Gewinn (EUR)", f"{cumulative_profit[-1]:,.2f}")
k3.metric("Ø Bewertung", f"{np.mean(review_score):.2f}")
k4.metric("Wettbewerbsdruck", f"{competition_intensity * 100:.0f}%")

# Visualisierungen
st.subheader("Preis- und Wettbewerbsverlauf")
fig1, ax1 = plt.subplots()
ax1.plot(time, price, label="Eigener Preis")
ax1.plot(time, competitor_price, label="Wettbewerber")
ax1.set_xlabel("Woche")
ax1.set_ylabel("Preis (EUR)")
ax1.legend()
st.pyplot(fig1)

st.subheader("Finanzkennzahlen")
fig2, ax2 = plt.subplots()
ax2.plot(time, revenue, label="Umsatz")
ax2.plot(time, profit, label="Gewinn")
ax2.plot(time, cumulative_profit, label="Kumul. Gewinn")
ax2.set_xlabel("Woche")
ax2.set_ylabel("EUR")
ax2.legend()
st.pyplot(fig2)

st.subheader("Konsumentenbewertung")
fig3, ax3 = plt.subplots()
ax3.plot(time, review_score, label="Ø Bewertung")
ax3.set_xlabel("Woche")
ax3.set_ylim(1.0, 5.0)
ax3.legend()
st.pyplot(fig3)

# Detailtabelle
st.subheader("Detailtabelle")
st.dataframe(df.set_index("Woche").style.format({
    "Eigener Preis (EUR)": "{:.2f}",
    "Wettbewerbspreis (EUR)": "{:.2f}",
    "Umsatz (EUR)": "{:.2f}",
    "Gewinn (EUR)": "{:.2f}",
    "Kumul. Gewinn (EUR)": "{:.2f}",
    "Ø Bewertung": "{:.2f}"
}))
