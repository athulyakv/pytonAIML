import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

# -----------------------------
# Safe model loader
# -----------------------------
@st.cache_resource
def load_model():
    try:
        return joblib.load("C:\\Users\\ASUS\\OneDrive\\supermarket\\demand_predictor.pkl")
    except AttributeError as e:
        st.error(
            "⚠️ Model could not be loaded. "
            "This usually happens because of a scikit-learn version mismatch.\n\n"
            "👉 Try installing scikit-learn==1.2.2 inside your virtual environment:\n"
            "`pip install scikit-learn==1.2.2`\n\n"
            f"Error details: {e}"
        )
        return None

model = load_model()

# -----------------------------
# Define valid product and store IDs
# -----------------------------
product_ids = [f"P{i:03d}" for i in range(1, 11)]
store_ids = [f"S{i:03d}" for i in range(1, 6)]

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🛒 Product Demand Prediction App")
st.write("Predict daily demand based on selected product, store, date, and promotions.")

# User inputs
product_id = st.selectbox("Select Product", product_ids)
store_id = st.selectbox("Select Store", store_ids)
selected_date = st.date_input("Select Date", datetime.today())
promo = st.checkbox("Promotion Active?")
is_holiday = st.checkbox("Is it a Holiday?")

# Extract weekday (0 = Monday, 6 = Sunday)
day_of_week = selected_date.weekday()

# -----------------------------
# Predict button
# -----------------------------
if st.button("🔍 Predict Demand"):
    if model is None:
        st.warning("🚫 Prediction unavailable until model is loaded correctly.")
    else:
        # Convert IDs like "P001" → 1, "S002" → 2
        product_num = int(product_id.replace("P", ""))
        store_num = int(store_id.replace("S", ""))

        # Build input DataFrame
        input_data = pd.DataFrame([{
            "product_id": product_num,
            "store_id": store_num,
            "promo": int(promo),
            "is_holiday": int(is_holiday),
            "day_of_week": int(day_of_week)
        }])

        # Debug print so you can verify
        st.write("🔎 Debug Input DataFrame:", input_data)
