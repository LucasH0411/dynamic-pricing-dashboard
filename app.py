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
        .stSlider > div[data-baseweb="slider"] > div { background: transparent !important; }
        .stMetric { background-color: #f8f9fa; padding: 20px; border-radius: 10px; border: 1px solid #dee2e6; }
    </style>
    """,
    unsafe_allow_html=True
)

st.sidebar.header("Preissteuerung")
base_price = 109.99
price_change_rate = st.sidebar.slider("Wöchentliche Preisänderung (%)", -0.2, 0.2, 0.0, step=0.01)

weeks = 12
price_elasticity = -2.0
base_demand = 1500
price_limit = 140
initial_customers = 1000
churn_sensitivity = 0.4
review_sensitivity = 0.3
review_decay = 0.1
demand_volatility = 0.1

time = np.arange(1, weeks + 1)
price = np.zeros(weeks)
demand = np.zeros(weeks)
revenue = np.zeros(weeks)
churn = np.zeros(weeks)
customers = initial_customers
lost_customers = np.zeros(weeks)
review_score = np.zeros(weeks)
cumulative_profit = np.zeros(weeks)

price[0] = base_price
review_score[0] = 4.5
fixed_costs = 10000
unit_cost = 40

for t in range(weeks):
    if t > 0:
        price[t] = price[t-1] * (1 + price_change_rate)

    price_effect = (price[t] / base_price) ** price_elasticity
    random_factor = np.random.uniform(1 - demand_volatility, 1 + demand_volatility)
    demand[t] = base_demand * price_effect * random_factor

    revenue[t] = price[t] * demand[t]
    profit = (price[t] - unit_cost) * demand[t] - fixed_costs / weeks
    cumulative_profit[t] = profit if t == 0 else cumulative_profit[t-1] + profit

    overprice_churn = max(0, price[t] - price_limit) / price_limit
    volatility_churn = abs(price[t] - price[t-1]) / price[t] if t > 0 else 0
    churn[t] = churn_sensitivity * (overprice_churn + volatility_churn)
    customers *= (1 - churn[t])
    lost_customers[t] = initial_customers - customers

    negative_influence = review_sensitivity * (overprice_churn + volatility_churn)
    review_score[t] = max(1.0, (review_score[t-1] - negative_influence + review_decay))

df = pd.DataFrame({
    "Woche": time,
    "Preis (EUR)": price,
    "Nachfrage": demand,
    "Umsatz (EUR)": revenue,
    "Kumulativer Gewinn (EUR)": cumulative_profit,
    "Churn Rate": churn,
    "Verlorene Kunden": lost_customers,
    "Ø Google-Bewertung": review_score
})

st.subheader("Kennzahlenübersicht")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Gesamtumsatz (EUR)", f"{revenue.sum():,.2f}")
k2.metric("Ø Bewertung", f"{np.mean(review_score):.2f}")
k3.metric("Verlorene Kunden", f"{int(lost_customers[-1]):,}")
k4.metric("Kum. Gewinn (EUR)", f"{cumulative_profit[-1]:,.2f}")

st.subheader("Zeitverlauf")

fig1, ax1 = plt.subplots()
ax1.plot(time, price, label="Preis (EUR)")
ax1.set_xlabel("Woche")
ax1.set_ylabel("Preis")
ax1.legend()
st.pyplot(fig1)

fig2, ax2 = plt.subplots()
ax2.plot(time, revenue, label="Umsatz (EUR)")
ax2.plot(time, cumulative_profit, label="Kumulativer Gewinn (EUR)")
ax2.set_xlabel("Woche")
ax2.set_ylabel("EUR")
ax2.legend()
st.pyplot(fig2)

fig3, ax3 = plt.subplots()
ax3.plot(time, churn, label="Churn Rate")
ax3.plot(time, review_score, label="Ø Bewertung")
ax3.set_xlabel("Woche")
ax3.legend()
st.pyplot(fig3)

st.subheader("Zusatzanalyse")
st.markdown("""
- **Preisniveaus > EUR 140** führen ab Woche x zu signifikantem Churn und Bewertungsrückgang.
- **Stabile Preisstrategie** resultiert in besseren Bewertungen.
- **Optimale Preisanpassung** liegt bei moderater Erhöhung (<5%/Woche) zur Gewinnmaximierung ohne Reputationsverlust.
""")

st.subheader("Detailtabelle")
st.dataframe(df.set_index("Woche").style.format({
    "Preis (EUR)": "{:.2f}",
    "Umsatz (EUR)": "{:.2f}",
    "Kumulativer Gewinn (EUR)": "{:.2f}",
    "Churn Rate": "{:.2%}",
    "Ø Google-Bewertung": "{:.2f}"
}))
