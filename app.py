import streamlit as st
import pandas as pd
import joblib

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Vehicle Machine Fault Prediction",
    page_icon="🚗",
    layout="wide"
)

# =====================================================
# LOAD MODEL
# =====================================================

MODEL_PATH = "models/vehicle_failure_model.pkl"
FEATURE_PATH = "models/feature_columns.pkl"

try:
    model = joblib.load(MODEL_PATH)
    features = joblib.load(FEATURE_PATH)

except Exception:
    st.error("Model files not found.")
    st.info("Please run: python src/train_models.py")
    st.stop()


# =====================================================
# HEADER
# =====================================================

st.title("🚗 Vehicle Machine Fault Prediction System")

st.write(
    "An intelligent Data Mining and Machine Learning "
    "system for predicting vehicle machine failure."
)

st.divider()


# =====================================================
# PROJECT INFORMATION
# =====================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Dataset Size", "100,000")

with col2:
    st.metric("Features Used", "14")

with col3:
    st.metric("ML Models", "3")

with col4:
    st.metric("Data Mining", "K-Means + IF")


st.divider()


# =====================================================
# SIDEBAR INPUT
# =====================================================

st.sidebar.title("🚗 Vehicle Parameters")

st.sidebar.write(
    "Enter the vehicle operating and maintenance details."
)

vehicle_age = st.sidebar.number_input(
    "Vehicle Age (Years)",
    1.0,
    15.0,
    8.0
)

mileage = st.sidebar.number_input(
    "Mileage (KM)",
    5000.0,
    300000.0,
    85000.0
)

usage_hours = st.sidebar.number_input(
    "Usage Hours",
    0.0,
    5000.0,
    2000.0
)

engine_temperature = st.sidebar.number_input(
    "Engine Temperature (°C)",
    0.0,
    150.0,
    80.0
)

engine_rpm = st.sidebar.number_input(
    "Engine RPM",
    500.0,
    5000.0,
    2000.0
)

oil_pressure = st.sidebar.number_input(
    "Oil Pressure (PSI)",
    5.0,
    60.0,
    35.0
)

coolant_temperature = st.sidebar.number_input(
    "Coolant Temperature (°C)",
    0.0,
    120.0,
    70.0
)

battery_voltage = st.sidebar.number_input(
    "Battery Voltage (V)",
    8.0,
    15.0,
    12.5
)

vibration = st.sidebar.number_input(
    "Vibration Level",
    0.0,
    10.0,
    2.0
)

fuel_consumption = st.sidebar.number_input(
    "Fuel Consumption (LPH)",
    1.0,
    30.0,
    10.0
)

tire_pressure = st.sidebar.number_input(
    "Tire Pressure (PSI)",
    15.0,
    50.0,
    32.0
)

brake_wear = st.sidebar.number_input(
    "Brake Wear (MM)",
    0.0,
    15.0,
    4.0
)

maintenance_count = st.sidebar.number_input(
    "Maintenance Count",
    0,
    30,
    5
)

days_since_maintenance = st.sidebar.number_input(
    "Days Since Last Maintenance",
    0,
    1000,
    90
)


# =====================================================
# INPUT DATA
# =====================================================

input_data = pd.DataFrame({

    "Vehicle_Age_Years": [vehicle_age],

    "Mileage_KM": [mileage],

    "Usage_Hours": [usage_hours],

    "Engine_Temperature_C": [engine_temperature],

    "Engine_RPM": [engine_rpm],

    "Oil_Pressure_PSI": [oil_pressure],

    "Coolant_Temperature_C": [coolant_temperature],

    "Battery_Voltage_V": [battery_voltage],

    "Vibration_Level": [vibration],

    "Fuel_Consumption_LPH": [fuel_consumption],

    "Tire_Pressure_PSI": [tire_pressure],

    "Brake_Wear_MM": [brake_wear],

    "Maintenance_Count": [maintenance_count],

    "Days_Since_Last_Maintenance": [
        days_since_maintenance
    ]
})


# =====================================================
# MAIN SECTION
# =====================================================

st.header("🔍 Vehicle Failure Analysis")

st.write(
    "Use the sidebar to enter vehicle parameters."
)

predict_button = st.button(
    "🚀 Predict Vehicle Failure",
    use_container_width=True
)


# =====================================================
# PREDICTION
# =====================================================

if predict_button:

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(
        input_data
    )[0][1]

    probability_percentage = probability * 100


    # =================================================
    # RISK LEVEL
    # =================================================

    if probability >= 0.60:

        risk_level = "HIGH"

    elif probability >= 0.30:

        risk_level = "MEDIUM"

    else:

        risk_level = "LOW"


    # =================================================
    # RESULT
    # =================================================

    st.divider()

    st.header("📊 Prediction Result")

    result1, result2, result3 = st.columns(3)

    with result1:

        if prediction == 1:

            st.error(
                "⚠️ VEHICLE FAILURE PREDICTED"
            )

        else:

            st.success(
                "✅ NO FAILURE PREDICTED"
            )


    with result2:

        st.metric(
            "Failure Probability",
            f"{probability_percentage:.2f}%"
        )


    with result3:

        st.metric(
            "Risk Level",
            risk_level
        )


    # =================================================
    # PROBABILITY
    # =================================================

    st.subheader("Failure Probability")

    st.progress(
        float(probability)
    )


    # =================================================
    # MAINTENANCE RECOMMENDATIONS
    # =================================================

    st.subheader(
        "🤖 AI-Based Maintenance Recommendations"
    )

    recommendations = []


    if engine_temperature > 100:

        recommendations.append(
            "Check engine cooling system and coolant level."
        )


    if oil_pressure < 25:

        recommendations.append(
            "Inspect engine oil level, oil filter and oil pump."
        )


    if vibration > 5:

        recommendations.append(
            "Inspect engine mounts, wheels and rotating components."
        )


    if battery_voltage < 11.8:

        recommendations.append(
            "Check battery condition and charging system."
        )


    if tire_pressure < 28 or tire_pressure > 38:

        recommendations.append(
            "Adjust tire pressure to the recommended level."
        )


    if brake_wear > 7:

        recommendations.append(
            "Inspect brake pads/discs and consider replacement."
        )


    if days_since_maintenance > 180:

        recommendations.append(
            "Vehicle maintenance is overdue. Schedule servicing."
        )


    if mileage > 200000:

        recommendations.append(
            "High mileage detected. Perform comprehensive inspection."
        )


    if vehicle_age > 12:

        recommendations.append(
            "Older vehicle detected. Increase preventive maintenance."
        )


    # =================================================
    # SHOW RECOMMENDATIONS
    # =================================================

    if len(recommendations) == 0:

        st.success(
            "✅ No major maintenance warning detected "
            "from the entered vehicle parameters."
        )

    else:

        for item in recommendations:

            st.warning(
                "🔧 " + item
            )


    # =================================================
    # VEHICLE DATA
    # =================================================

    st.subheader("📋 Vehicle Input Summary")

    st.dataframe(
        input_data,
        use_container_width=True
    )


# =====================================================
# FOOTER
# =====================================================

st.divider()

st.caption(
    "Vehicle Machine Fault Prediction | "
    "Data Mining + Machine Learning + Predictive Analytics"
)

st.caption(
    "Predictive results are intended for analytical purposes "
    "and should not replace professional vehicle inspection."
)