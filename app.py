import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Advanced Dynamic Pricing Simulation", layout="wide")
st.title("👗 Dynamic Pricing Simulation – Modebranche")

# --- Sidebar Inputs with Explanations ---
st.sidebar.header("🔧 Parameter-Einstellungen")

with st.sidebar.expander("Preissetzung"):
    base_price = st.slider("Basispreis (€)", 10, 100, 30, help="Ausgangspunkt des Preises pro Artikel.")
    price_elasticity_sensitive = st.slider("Preiselastizität (preissensibel)", -5.0, -0.5, -2.5, step=0.1, help="Reagieren stark auf Preisänderungen.")
    price_elasticity_loyal = st.slider("Preiselastizität (loyal)", -2.0, -0.1, -0.5, step=0.1, help="Kaum Preisreaktion.")

with st.sidebar.expander("Nachfrage"):
    base_demand = st.slider("Basisnachfrage (pro Woche)", 100, 2000, 1000, step=50, help="Marktnachfrage ohne Preiseffekt.")
    random_fluctuation = st.slider("Zufallseinfluss auf Nachfrage", 0.0, 0.5, 0.1, help="Zufällige Nachfrageschwankungen.")

with st.sidebar.expander("Preisdynamik"):
    price_change_rate = st.slider("Preisänderungsrate", 0.0, 0.5, 0.1, step=0.05, help="Wie stark wird der Preis pro Woche angepasst?")
    volatility_penalty = st.slider("Reaktion auf Preisschwankungen", 0.0, 1.0, 0.3, help="Sanktion durch Kunden bei zu häufigen Änderungen.")

with st.sidebar.expander("Churn-Faktoren"):
    churn_sensitivity = st.slider("Churn-Sensitivität", 0.0, 1.0, 0.4, help="Wie schnell Kunden bei Preisstress abspringen.")
    price_limit = st.slider("Preisakzeptanzgrenze (€)", 20, 120, 70, help="Obergrenze, ab der Kunden systematisch abspringen.")

# --- Simulation Setup ---
weeks = 12
time = np.arange(1, weeks + 1)
price = np.zeros(weeks)
demand_sensitive = np.zeros(weeks)
demand_loyal = np.zeros(weeks)
revenue = np.zeros(weeks)
churn = np.zeros(weeks)
price[0] = base_price
customers = 1000
lost_customers = np.zeros(weeks)

# --- Simulation Loop ---
for t in range(weeks):
    if t > 0:
        # Dynamisch steigender Preis
        price[t] = price[t-1] * (1 + price_change_rate)

    # Reaktion der Segmente
    sensitive_factor = (price[t] / base_price) ** price_elasticity_sensitive
    loyal_factor = (price[t] / base_price) ** price_elasticity_loyal

    fluctuation = np.random.uniform(1 - random_fluctuation, 1 + random_fluctuation)
    demand_sensitive[t] = base_demand * 0.6 * sensitive_factor * fluctuation
    demand_loyal[t] = base_demand * 0.4 * loyal_factor * fluctuation
    total_demand = demand_sensitive[t] + demand_loyal[t]

    # Churn durch Preis oder Volatilität
    overprice_churn = max(0, price[t] - price_limit) / price_limit
    change_churn = volatility_penalty * abs(price[t] - price[t-1]) / price[t] if t > 0 else 0
    churn[t] = churn_sensitivity * (overprice_churn + change_churn)
    customers *= (1 - churn[t])
    lost_customers[t] = 1000 - customers

    revenue[t] = price[t] * total_demand

# --- KPIs ---
df = pd.DataFrame({
    "Woche": time,
    "Preis (€)": price,
    "Nachfrage Preissensibel": demand_sensitive,
    "Nachfrage Loyal": demand_loyal,
    "Umsatz (€)": revenue,
    "Churn Rate": churn,
    "Kundenverlust (kumuliert)": lost_customers
})

# --- KPI Dashboard ---
st.subheader("📊 KPIs über 12 Wochen")

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Gesamtumsatz (€)", f"{revenue.sum():,.2f}")
kpi2.metric("Durchschnittlicher Preis (€)", f"{np.mean(price):.2f}")
kpi3.metric("Durchschn. Churn Rate", f"{np.mean(churn)*100:.2f}%")
kpi4.metric("Verlorene Kunden", f"{int(lost_customers[-1]):,}")

# --- Plots ---
st.line_chart(df.set_index("Woche")[["Preis (€)", "Umsatz (€)", "Churn Rate"]])
st.line_chart(df.set_index("Woche")[["Nachfrage Preissensibel", "Nachfrage Loyal"]])
st.dataframe(df.style.format({"Preis (€)": "{:.2f}", "Umsatz (€)": "{:.2f}", "Churn Rate": "{:.2%}"}))
