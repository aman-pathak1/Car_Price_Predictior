import streamlit as st
import pandas as pd
import joblib

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="wide"
)

# ---------------- DARK YELLOW THEME ----------------

st.markdown("""
<style>

.stApp {
background-color:#1e1e1e;
color:white;
}

h1{
color:#ffcc00;
text-align:center;
}

.stButton>button{
background-color:#ffcc00;
color:black;
font-weight:bold;
border-radius:8px;
height:45px;
width:200px;
}

.stButton>button:hover{
background-color:#e6b800;
}

.block-container{
animation: fadeIn 1s ease-in;
}

@keyframes fadeIn{
0%{opacity:0; transform:translateY(20px);}
100%{opacity:1; transform:translateY(0);}
}

</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------

st.title("🚗 Car Price Prediction App")

# ---------------- LOAD MODEL ----------------

model = joblib.load("pipe.pkl")

# ---------------- INPUT SECTION ----------------

col1, col2, col3 = st.columns(3)

with col1:

    name = st.text_input("Car Name", "Maruti Swift")

    year = st.number_input("Year", 2000, 2025, 2015)

    km_driven = st.number_input("KM Driven", value=50000)

    fuel = st.selectbox(
        "Fuel",
        ["Petrol","Diesel","CNG","LPG"]
    )

with col2:

    seller_type = st.selectbox(
        "Seller Type",
        ["Dealer","Individual","Trustmark Dealer"]
    )

    transmission = st.selectbox(
        "Transmission",
        ["Manual","Automatic"]
    )

    # -------- OWNER (NUMERICAL MAPPING) --------

    owner_dict = {
        "Test Drive Car":0,
        "First Owner":1,
        "Second Owner":2,
        "Third Owner":3,
        "Fourth & Above Owner":4
    }

    owner_text = st.selectbox(
        "Owner",
        list(owner_dict.keys())
    )

    owner = owner_dict[owner_text]

with col3:

    mileage = st.number_input(
        "Mileage (km/ltr/kg)",
        value=20.0
    )

    engine = st.number_input(
        "Engine (CC)",
        value=1200
    )

    max_power = st.number_input(
        "Max Power",
        value=80.0
    )

    seats = st.number_input(
        "Seats",
        min_value=2,
        max_value=10,
        value=5
    )

# ---------------- PREDICTION ----------------

if st.button("Predict Price 🚀"):

    input_data = pd.DataFrame({
        'name':[name],
        'year':[year],
        'km_driven':[km_driven],
        'fuel':[fuel],
        'seller_type':[seller_type],
        'transmission':[transmission],
        'owner':[owner],
        'mileage(km/ltr/kg)':[mileage],
        'engine':[engine],
        'max_power':[max_power],
        'seats':[seats]
    })

    prediction = model.predict(input_data)[0]

    st.success(f"💰 Predicted Car Price: ₹ {round(prediction,2)}")