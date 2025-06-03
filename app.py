import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(page_title="Dynamic Pricing - ShopTrend24", layout="wide")
st.title("Dynamic Pricing - ShopTrend24")

st.markdown(
    '''
    <style>
        .main { background-color: #f8f9fa; }
        .block-container { padding-top: 2rem; }
        .stSlider > div[data-baseweb="slider"] > div { background: #0e1117; }
        .stMetric { background-color: #ffffff; padding: 20px; border-radius: 10px; border: 1px solid #e6e6e6; }
    </style>
    ''',
    unsafe_allow_html=True
)

# --- Sidebar Inputs with Detailed Explanations ---
st.sidebar.header("Parameter-Einstellungen")

with st.sidebar.expander("Preisstrategie"):
    base_price = st.slider("Startpreis für Turnschuhe (€)", 50, 200, 109, step=1,
                           help="Initialer Preis des Produkts. Dieser dient als Ausgangspunkt der Preisstrategie.")
    price_change_rate = st.slider("Wöchentliche Preisänderung (%)", -0.5, 0.5, 0.05, step=0.01,
                                  help="Gibt an, wie stark der Preis jede Woche angepasst wird. Positive Werte bedeuten Preiserhöhung.")
    price_limit = st.slider("Preisakzeptanzgrenze (€)", 80, 200, 140,
                            help="Obergrenze, ab der Konsumenten die Preise als überhöht empfinden und mit Kaufverzicht reagieren.")

with st.sidebar.expander("Nachfrageverhalten"):
    base_demand = st.slider("Basisnachfrage pro Woche", 100, 3000, 1500, step=50,
                            help="Maximale Nachfrage, wenn der Preis optimal ist und keine negativen Reaktionen auftreten.")
    price_elasticity = st.slider("Preiselastizität", -5.0, -0.1, -2.0, step=0.1,
                                 help="Beschreibt, wie empfindlich die Nachfrage auf Preisänderungen reagiert. Je negativer, desto stärker der Rückgang bei Preiserhöhungen.")
    demand_volatility = st.slider("Zufällige Nachfrageschwankungen", 0.0, 0.5, 0.1,
                                  help="Zufallskomponente, die saisonale oder externe Einflüsse auf die Nachfrage simuliert.")

with st.sidebar.expander("Konsumentenreaktionen"):
    review_sensitivity = st.slider("Negatives Word-of-Mouth Sensitivität", 0.0, 1.0, 0.3,
                                   help="Je höher dieser Wert, desto stärker beeinflussen Preisstress oder Volatilität die Kundenmeinungen.")
    review_decay = st.slider("Rückgang negativer Bewertungen über Zeit", 0.0, 0.5, 0.1,
                             help="Bewertungen verbessern sich wieder, wenn Preisstrategie stabil bleibt.")

with st.sidebar.expander("Churn-Mechanik"):
    churn_sensitivity = st.slider("Kundenabwanderungssensitivität", 0.0, 1.0, 0.4,
                                  help="Bestimmt, wie empfindlich Kunden auf zu hohe Preise oder zu viele Preisänderungen reagieren.")
    initial_customers = 1000

# --- Simulation Setup ---
weeks = 12
time = np.arange(1, weeks + 1)
price = np.zeros(weeks)
demand = np.zeros(weeks)
revenue = np.zeros(weeks)
churn = np.zeros(weeks)
customers = initial_customers
lost_customers = np.zeros(weeks)
review_score = np.zeros(weeks)

# Initial values
price[0] = base_price
review_score[0] = 4.5

# --- Simulation Loop ---
for t in range(weeks):
    if t > 0:
        price[t] = price[t-1] * (1 + price_change_rate)

    price_effect = (price[t] / base_price) ** price_elasticity
    random_factor = np.random.uniform(1 - demand_volatility, 1 + demand_volatility)
    demand[t] = base_demand * price_effect * random_factor

    overprice_churn = max(0, price[t] - price_limit) / price_limit
    volatility_churn = abs(price[t] - price[t-1]) / price[t] if t > 0 else 0
    churn[t] = churn_sensitivity * (overprice_churn + volatility_churn)

    customers *= (1 - churn[t])
    lost_customers[t] = initial_customers - customers
    revenue[t] = demand[t] * price[t]

    # Review Logic
    negative_influence = review_sensitivity * (overprice_churn + volatility_churn)
    review_score[t] = max(1.0, (review_score[t-1] - negative_influence + review_decay))

# --- KPIs ---
df = pd.DataFrame({
    "Woche": time,
    "Preis (€)": price,
    "Nachfrage": demand,
    "Umsatz (€)": revenue,
    "Churn Rate": churn,
    "Verlorene Kunden": lost_customers,
    "Ø Google-Bewertung": review_score
})

# --- KPI Dashboard ---
st.subheader("Kennzahlenübersicht")

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Gesamtumsatz (€)", f"{revenue.sum():,.2f}")
kpi2.metric("Ø Preis (€)", f"{np.mean(price):.2f}")
kpi3.metric("Ø Bewertung", f"{np.mean(review_score):.2f}")
kpi4.metric("Verlorene Kunden", f"{int(lost_customers[-1]):,}")

# --- Visualisierung ---
st.line_chart(df.set_index("Woche")[["Preis (€)", "Nachfrage", "Umsatz (€)"]])
st.line_chart(df.set_index("Woche")[["Churn Rate", "Ø Google-Bewertung"]])
st.dataframe(df.style.format({"Preis (€)": "{:.2f}", "Umsatz (€)": "{:.2f}", "Churn Rate": "{:.2%}", "Ø Google-Bewertung": "{:.2f}"}))
