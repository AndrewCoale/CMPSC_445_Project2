import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# Load trained models
gdp_model = joblib.load("gdp_model.pkl")
inflation_model = joblib.load("inflation_model.pkl")

# Extract categories from the model's encoder
encoder = gdp_model.named_steps["preprocessor"].named_transformers_["cat"]
countries = encoder.categories_[0]
tariffs = encoder.categories_[1]

# Streamlit app
st.set_page_config(page_title="Tariff Economic Impact Predictor", layout="wide")
st.title("\U0001F4B0 Tariff Economic Impact Predictor")

with st.sidebar:
    st.header("Input Parameters")
    country = st.selectbox("Select Country", countries)
    tariff_policy = st.selectbox("Select Trump Tariff Policy", tariffs)
    us_exports = st.number_input("US 2024 Exports (in B$)", value=100.0)
    us_imports = st.number_input("US 2024 Imports (Customs Basis) (in B$)", value=100.0)
    population = st.number_input("Population (in millions)", value=50.0)

input_df = pd.DataFrame([[country, tariff_policy, us_exports, us_imports, population * 1_000_000]],
                         columns=["Country", "Trump Tariffs Alleged", "US 2024 Exports", "US 2024 Imports (Customs Basis)", "Population"])

if st.button("\u2705 Predict Impact"):
    gdp_pred = gdp_model.predict(input_df)[0]
    inflation_pred = inflation_model.predict(input_df)[0]

    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Predicted Trade Balance Impact (B$)", value=f"{gdp_pred:.2f}")
    with col2:
        st.metric(label="Predicted Deficit per Capita Impact ($)", value=f"{inflation_pred:.4f}")

    # Add a simple bar chart
    fig = go.Figure(data=[
        go.Bar(name='GDP Impact', x=['Trade Balance'], y=[gdp_pred]),
        go.Bar(name='Inflation Impact', x=['Deficit per Capita'], y=[inflation_pred])
    ])
    fig.update_layout(barmode='group', title_text='Predicted Economic Impacts')
    st.plotly_chart(fig, use_container_width=True)
