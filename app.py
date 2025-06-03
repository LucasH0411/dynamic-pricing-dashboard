import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Dynamic Pricing Dashboard", layout="wide")
st.title("📈 Dynamic Pricing Simulation")

# --- Sidebar Parameters ---
st.sidebar.header("🔧 Parameter einstellen")
base_price = st.sidebar.slider("Basispreis (€)", 5, 50, 20)
price_elasticity = st.sidebar.slider("Preissensitivität (ε)", -5.0, -0.1, -2.0)
time_factor = st.sidebar.slider("Zeitabhängigkeit", 0.0, 1.0, 0.2)
demand_level = st.sidebar.slider("Basisnachfrage", 100, 1000, 500)

# --- Simulation Function ---
def simulate_demand(price, time, base_demand, elasticity, time_effect):
    time_decay = 1 - time_effect * time
    return base_demand * (price / base_price) ** elasticity * time_decay

# --- Run Simulation ---
time_steps = np.arange(0, 10)
prices = base_price * (1 + time_factor * time_steps / 10)
demands = simulate_demand(prices, time_steps, demand_level, price_elasticity, time_factor)
revenues = prices * demands

df = pd.DataFrame({
    "Zeit": time_steps,
    "Preis (€)": prices,
    "Nachfrage": demands,
    "Umsatz (€)": revenues
})

# --- Plots ---
st.subheader("📊 Ergebnisse der Simulation")

col1, col2 = st.columns(2)
with col1:
    st.line_chart(df.set_index("Zeit")[["Preis (€)", "Nachfrage"]])
with col2:
    st.line_chart(df.set_index("Zeit")[["Umsatz (€)"]])

st.dataframe(df.style.format({"Preis (€)": "{:.2f}", "Umsatz (€)": "{:.2f}"}))
