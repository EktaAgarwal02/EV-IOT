import streamlit as st
import numpy as np
import joblib

# Load trained model
model = joblib.load("xgboost.pkl")

# Page title
st.set_page_config(page_title="⚡ EV Failure Prediction")
st.title("⚡ EV Failure Prediction System")
st.subheader("Predict EV Component Failure based on sensor data")

# Sidebar inputs
st.sidebar.header("🔧 Enter Sensor Readings")

# Define sensible ranges / options for each feature
feature_inputs = {
    "SoC": st.sidebar.slider("SoC (%)", 0, 100, 50),
    "SoH": st.sidebar.slider("SoH (%)", 0, 100, 90),
    "Battery_Voltage": st.sidebar.slider("Battery Voltage (V)", 100, 500, 300),
    "Battery_Current": st.sidebar.slider("Battery Current (A)", -200, 200, 0),
    "Battery_Temperature": st.sidebar.slider("Battery Temp (°C)", -20, 80, 25),
    "Charge_Cycles": st.sidebar.slider("Charge Cycles", 0, 2000, 500),
    "Motor_Temperature": st.sidebar.slider("Motor Temp (°C)", -10, 150, 60),
    "Motor_Vibration": st.sidebar.slider("Motor Vibration (m/s²)", 0, 50, 5),
    "Motor_Torque": st.sidebar.slider("Motor Torque (Nm)", 0, 500, 200),
    "Motor_RPM": st.sidebar.slider("Motor RPM", 0, 10000, 3000),
    "Power_Consumption": st.sidebar.slider("Power Consumption (kW)", 0, 500, 100),
    "Brake_Pad_Wear": st.sidebar.slider("Brake Pad Wear (%)", 0, 100, 20),
    "Brake_Pressure": st.sidebar.slider("Brake Pressure (bar)", 0, 200, 50),
    "Reg_Brake_Efficiency": st.sidebar.slider("Regen Brake Efficiency (%)", 0, 100, 70),
    "Tire_Pressure": st.sidebar.slider("Tire Pressure (psi)", 20, 50, 32),
    "Tire_Temperature": st.sidebar.slider("Tire Temp (°C)", -10, 100, 30),
    "Suspension_Load": st.sidebar.slider("Suspension Load (kg)", 0, 1000, 200),
    "Ambient_Temperature": st.sidebar.slider("Ambient Temp (°C)", -20, 50, 25),
    "Ambient_Humidity": st.sidebar.slider("Ambient Humidity (%)", 0, 100, 50),
    "Load_Weight": st.sidebar.slider("Load Weight (kg)", 0, 5000, 500),
    "Driving_Speed": st.sidebar.slider("Driving Speed (km/h)", 0, 200, 60),
    "Distance_Traveled": st.sidebar.slider("Distance Traveled (km)", 0, 1000, 100),
    "Idle_Time": st.sidebar.slider("Idle Time (min)", 0, 600, 10),
    "Route_Roughness": st.sidebar.slider("Route Roughness (0=smooth, 10=rough)", 0, 10, 3),
    "RUL": st.sidebar.slider("Remaining Useful Life (hrs)", 0, 2000, 1000),
    "Maintenance_Type": st.sidebar.selectbox("Maintenance Type", ["None", "Preventive", "Corrective"]),
    "TTF": st.sidebar.slider("Time to Failure (hrs)", 0, 2000, 500),
    "Component_Health_Score": st.sidebar.slider("Component Health Score (0-100)", 0, 100, 80),
}

# Collect values in correct order
features = [
    'SoC', 'SoH', 'Battery_Voltage', 'Battery_Current',
    'Battery_Temperature', 'Charge_Cycles', 'Motor_Temperature',
    'Motor_Vibration', 'Motor_Torque', 'Motor_RPM', 'Power_Consumption',
    'Brake_Pad_Wear', 'Brake_Pressure', 'Reg_Brake_Efficiency',
    'Tire_Pressure', 'Tire_Temperature', 'Suspension_Load',
    'Ambient_Temperature', 'Ambient_Humidity', 'Load_Weight',
    'Driving_Speed', 'Distance_Traveled', 'Idle_Time', 'Route_Roughness',
    'RUL', 'Maintenance_Type', 'TTF',
    'Component_Health_Score'
]

input_data = []
for col in features:
    val = feature_inputs[col]
    # Encode categorical (Maintenance_Type)
    if col == "Maintenance_Type":
        val = {"None": 0, "Preventive": 1, "Corrective": 2}[val]
    input_data.append(val)

# Predict button
if st.sidebar.button("🚀 Predict Failure"):
    input_array = np.array([input_data])
    prediction = model.predict(input_array)[0]

    if prediction == 1:
        st.error("⚠️ Failure Predicted! Maintenance Required.")
    else:
        st.success("✅ No Failure Predicted. System is Healthy.")

# Footer
st.markdown("---")
st.caption("Made with ❤️ using Streamlit")
