import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from model import load_model, predict_price

st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A5F;
        text-align: center;
        margin-bottom: 0.25rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #6B7280;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background: linear-gradient(135deg, #1E3A5F 0%, #2563EB 100%);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        color: white;
        margin: 1rem 0;
    }
    .prediction-label {
        font-size: 1rem;
        opacity: 0.85;
        margin-bottom: 0.5rem;
    }
    .prediction-value {
        font-size: 2.8rem;
        font-weight: 800;
        letter-spacing: -1px;
    }
    .prediction-unit {
        font-size: 1rem;
        opacity: 0.75;
        margin-top: 0.25rem;
    }
    .info-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1.25rem;
        margin: 0.5rem 0;
    }
    .section-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: #1E3A5F;
        margin-bottom: 1rem;
        border-left: 4px solid #2563EB;
        padding-left: 0.75rem;
    }
    .stButton>button {
        background: linear-gradient(135deg, #1E3A5F 0%, #2563EB 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        width: 100%;
        transition: opacity 0.2s;
    }
    .stButton>button:hover {
        opacity: 0.9;
    }
    .metric-card {
        background: white;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    </style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def get_model():
    return load_model()


@st.cache_data
def load_dataset():
    try:
        return pd.read_csv("car_data.csv")
    except FileNotFoundError:
        return None


model_data = get_model()
df = load_dataset()

st.markdown('<p class="main-header">🚗 Car Price Predictor</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Get an instant AI-powered estimate for your car\'s selling price</p>',
    unsafe_allow_html=True,
)

st.sidebar.markdown("## 📋 Car Details")
st.sidebar.markdown("Fill in the details below to get a price prediction.")

with st.sidebar:
    st.markdown("### 🏷️ Basic Information")
    car_name = st.text_input("Car Model Name", placeholder="e.g. Swift, Innova, Creta")
    current_year = 2024
    year = st.slider(
        "Year of Purchase", min_value=2000, max_value=current_year, value=2018, step=1
    )
    present_price = st.number_input(
        "Showroom Price (Lakhs ₹)",
        min_value=0.5,
        max_value=100.0,
        value=8.0,
        step=0.1,
        format="%.2f",
        help="Current ex-showroom price of the car",
    )

    st.markdown("### ⚙️ Usage & Condition")
    kms_driven = st.number_input(
        "Kilometres Driven",
        min_value=0,
        max_value=500000,
        value=25000,
        step=500,
        help="Total distance the car has been driven",
    )
    owner = st.selectbox(
        "Number of Previous Owners",
        options=[0, 1, 2, 3],
        format_func=lambda x: {
            0: "0 – First Owner",
            1: "1 – Second Owner",
            2: "2 – Third Owner",
            3: "3+ – Fourth Owner or More",
        }[x],
    )

    st.markdown("### 🔧 Vehicle Specifications")
    fuel_type = st.selectbox(
        "Fuel Type", options=["Petrol", "Diesel", "CNG"], index=0
    )
    transmission = st.selectbox(
        "Transmission", options=["Manual", "Automatic"], index=0
    )
    seller_type = st.selectbox(
        "Seller Type", options=["Dealer", "Individual"], index=0
    )

    predict_btn = st.button("🔍 Predict Selling Price")

car_age = current_year - year

col1, col2 = st.columns([3, 2], gap="large")

with col1:
    st.markdown('<p class="section-title">📊 Prediction Result</p>', unsafe_allow_html=True)

    if predict_btn:
        try:
            predicted = predict_price(
                model_data,
                car_age,
                present_price,
                kms_driven,
                fuel_type,
                seller_type,
                transmission,
                owner,
            )

            depreciation_pct = ((present_price - predicted) / present_price) * 100

            st.markdown(
                f"""
                <div class="prediction-box">
                    <p class="prediction-label">Estimated Selling Price</p>
                    <p class="prediction-value">₹ {predicted:.2f} Lakhs</p>
                    <p class="prediction-unit">± 10% margin of error</p>
                </div>
            """,
                unsafe_allow_html=True,
            )

            mc1, mc2, mc3 = st.columns(3)
            with mc1:
                st.metric(
                    "Showroom Price",
                    f"₹ {present_price:.2f}L",
                    help="Original ex-showroom price",
                )
            with mc2:
                st.metric(
                    "Predicted Price",
                    f"₹ {predicted:.2f}L",
                    delta=f"-₹{present_price - predicted:.2f}L",
                    delta_color="inverse",
                )
            with mc3:
                st.metric(
                    "Depreciation",
                    f"{depreciation_pct:.1f}%",
                    help="Total value lost from original price",
                )

            st.markdown("#### 💡 Price Breakdown")
            fig = go.Figure(
                go.Bar(
                    x=["Showroom Price", "Predicted Selling Price"],
                    y=[present_price, predicted],
                    marker_color=["#2563EB", "#10B981"],
                    text=[f"₹{present_price:.2f}L", f"₹{predicted:.2f}L"],
                    textposition="outside",
                    width=0.4,
                )
            )
            fig.update_layout(
                yaxis_title="Price (Lakhs ₹)",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                height=320,
                margin=dict(t=20, b=20),
                yaxis=dict(gridcolor="#F1F5F9"),
            )
            st.plotly_chart(fig, use_container_width=True)

        except ValueError as e:
            st.error("⚠️ Please check that all input values are valid and try again.")
        except Exception:
            st.error("⚠️ An unexpected error occurred. Please try again later.")
    else:
        st.info("👈 Fill in your car details in the sidebar and click **Predict Selling Price**.")

        if df is not None:
            st.markdown("#### 📈 Sample Prices from Our Dataset")
            sample = df[["Car_Name", "Year", "Present_Price", "Selling_Price", "Fuel_Type"]].head(8)
            sample.columns = ["Car", "Year", "Showroom Price (L)", "Selling Price (L)", "Fuel"]
            st.dataframe(sample, use_container_width=True, hide_index=True)

with col2:
    st.markdown('<p class="section-title">ℹ️ Your Car Summary</p>', unsafe_allow_html=True)

    summary_data = {
        "🚗 Car Model": car_name if car_name else "Not specified",
        "📅 Year of Purchase": year,
        "⏳ Car Age": f"{car_age} year{'s' if car_age != 1 else ''}",
        "💰 Showroom Price": f"₹ {present_price:.2f} Lakhs",
        "🛣️ Kms Driven": f"{kms_driven:,} km",
        "⛽ Fuel Type": fuel_type,
        "⚙️ Transmission": transmission,
        "🏪 Seller Type": seller_type,
        "👤 Previous Owners": owner,
    }

    for label, value in summary_data.items():
        st.markdown(
            f"""
            <div class="info-card">
                <strong>{label}</strong><br>
                <span style="color:#4B5563;">{value}</span>
            </div>
        """,
            unsafe_allow_html=True,
        )

    if df is not None:
        st.markdown("---")
        st.markdown('<p class="section-title">📊 Market Insights</p>', unsafe_allow_html=True)

        fuel_counts = df["Fuel_Type"].value_counts()
        fig_pie = px.pie(
            values=fuel_counts.values,
            names=fuel_counts.index,
            color_discrete_sequence=["#2563EB", "#10B981", "#F59E0B"],
            hole=0.4,
        )
        fig_pie.update_layout(
            height=220,
            margin=dict(t=10, b=10, l=10, r=10),
            paper_bgcolor="rgba(0,0,0,0)",
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.2),
        )
        fig_pie.update_traces(textinfo="percent+label")
        st.markdown("**Fuel Type Distribution**")
        st.plotly_chart(fig_pie, use_container_width=True)

st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#9CA3AF; font-size:0.85rem;'>"
    "🚗 Car Price Predictor — Powered by Random Forest ML | "
    "Prices are estimates only and may vary based on actual condition and market.</p>",
    unsafe_allow_html=True,
)
