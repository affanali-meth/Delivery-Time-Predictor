import streamlit as st
import pickle
import pandas as pd
import numpy as np

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Delivery Intelligence Dashboard",
    page_icon="🚚",
    layout="wide"
)

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_model():
    with open("optimized_rf_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("label_encoders.pkl", "rb") as f:
        encoders = pickle.load(f)
    return model, encoders

model, label_encoders = load_model()

# -----------------------------
# HEADER
# -----------------------------
st.markdown("""
# 🚚 Delivery Intelligence Dashboard
### AI-powered delivery time prediction & insights
""")

st.markdown("---")

# -----------------------------
# LAYOUT
# -----------------------------
col1, col2 = st.columns([1, 1.3])

# -----------------------------
# INPUT PANEL
# -----------------------------
with col1:
    st.subheader("📥 Input Parameters")

    weather = st.selectbox("Weather", label_encoders["Weather"].classes_)
    traffic = st.selectbox("Traffic Level", label_encoders["Traffic_Level"].classes_)
    time_of_day = st.selectbox("Time of Day", label_encoders["Time_of_Day"].classes_)
    vehicle = st.selectbox("Vehicle Type", label_encoders["Vehicle_Type"].classes_)

    distance = st.slider("Distance (km)", 0.0, 50.0, 5.0)
    prep_time = st.slider("Preparation Time (min)", 0.0, 60.0, 15.0)
    experience = st.slider("Courier Experience (yrs)", 0.0, 20.0, 2.0)

    predict_btn = st.button("🚀 Predict Delivery Time")

# -----------------------------
# PREPROCESS
# -----------------------------
def preprocess():
    data = {
        "Distance_km": distance,
        "Weather": label_encoders["Weather"].transform([weather])[0],
        "Traffic_Level": label_encoders["Traffic_Level"].transform([traffic])[0],
        "Time_of_Day": label_encoders["Time_of_Day"].transform([time_of_day])[0],
        "Vehicle_Type": label_encoders["Vehicle_Type"].transform([vehicle])[0],
        "Preparation_Time_min": prep_time,
        "Courier_Experience_yrs": experience
    }

    df = pd.DataFrame([data])

    df = df[[
        'Distance_km',
        'Weather',
        'Traffic_Level',
        'Time_of_Day',
        'Vehicle_Type',
        'Preparation_Time_min',
        'Courier_Experience_yrs'
    ]]

    return df

# -----------------------------
# OUTPUT PANEL
# -----------------------------
with col2:
    st.subheader("📊 Prediction & Insights")

    if predict_btn:
        try:
            input_df = preprocess()
            prediction = model.predict(input_df)[0]

            c1, c2, c3 = st.columns(3)

            with c1:
                st.metric("⏱ Delivery Time", f"{round(prediction, 2)} min")

            with c2:
                if prediction < 30:
                    st.metric("⚡ Speed", "Fast")
                elif prediction < 60:
                    st.metric("⏳ Speed", "Moderate")
                else:
                    st.metric("🚨 Speed", "Slow")

            with c3:
                st.metric("📍 Distance", f"{distance} km")

            st.progress(min(int(prediction), 100))

            # Feature Importance
            st.markdown("### 📊 Feature Importance")
            try:
                importance = model.feature_importances_
                feature_names = [
                    'Distance_km',
                    'Weather',
                    'Traffic_Level',
                    'Time_of_Day',
                    'Vehicle_Type',
                    'Preparation_Time_min',
                    'Courier_Experience_yrs'
                ]

                importance_df = pd.DataFrame({
                    "Feature": feature_names,
                    "Importance": importance
                }).sort_values(by="Importance", ascending=False)

                st.bar_chart(importance_df.set_index("Feature"))

            except:
                st.warning("Feature importance not available.")

            # Map
            st.markdown("### 🌍 Delivery Map")

            lat = 17.3850 + np.random.uniform(-0.01, 0.01)
            lon = 78.4867 + np.random.uniform(-0.01, 0.01)

            map_df = pd.DataFrame({
                'lat': [lat],
                'lon': [lon]
            })

            st.map(map_df)

        except Exception as e:
            st.error(f"Error: {e}")

    else:
        st.info("Enter inputs and click Predict to see results.")

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.markdown(
    "🚀 Built with Streamlit | Random Forest ML Model | Production-ready AI Project"
)