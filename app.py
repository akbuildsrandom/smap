# SmartMaintain - Simple Streamlit Web Application
import os
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'

import numpy as np
import pandas as pd
import joblib
import streamlit as st

st.set_page_config(page_title="SmartMaintain", page_icon="⚙️", layout="wide")

# 1. Load Trained Pipeline and Dataset
@st.cache_resource
def get_model():
    return joblib.load('models/machine_failure_model.pkl')

@st.cache_data
def get_data():
    return pd.read_csv('data/ai4i2020.csv')

model = get_model()
df = get_data()

# 2. Helper Functions
def add_features(data):
    """Calculates the 2 physical engineered features required by the model."""
    d = data.copy()
    d['Temperature difference [K]'] = d['Process temperature [K]'] - d['Air temperature [K]']
    d['Mechanical power [W]'] = d['Torque [Nm]'] * (2 * np.pi * d['Rotational speed [rpm]'] / 60.0)
    return d

def get_risk_category(prob):
    """Assigns risk tier based on application thresholds (<10%, 10-30%, >=30%)."""
    if prob < 0.10:
        return "Low Risk", "green"
    elif prob < 0.30:
        return "Moderate Risk", "orange"
    return "High Risk", "red"

# 3. Sidebar Navigation
st.sidebar.title("⚙️ SmartMaintain")
st.sidebar.caption("Machine Failure Prediction System")
section = st.sidebar.radio("Navigation", ["Dashboard", "Single Prediction", "Batch Prediction"])

# ==============================================================================
# SECTION A: DASHBOARD
# ==============================================================================
if section == "Dashboard":
    st.title("📊 Operational Fleet Dashboard")
    
    total = len(df)
    failures = int(df['Machine failure'].sum())
    normals = total - failures
    failure_pct = (failures / total) * 100

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records", f"{total:,}")
    col2.metric("Normal Machines", f"{normals:,}")
    col3.metric("Failure Records", f"{failures:,}")
    col4.metric("Failure Rate", f"{failure_pct:.2f}%")

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Normal vs. Failure Distribution")
        target_counts = df['Machine failure'].value_counts()
        target_counts.index = ["Normal (0)", "Failure (1)"]
        st.bar_chart(target_counts)
    
    with c2:
        st.subheader("Failures by Product Type (L, M, H)")
        type_fails = df.groupby('Type')['Machine failure'].sum()
        st.bar_chart(type_fails)

# ==============================================================================
# SECTION B: SINGLE MACHINE PREDICTION
# ==============================================================================
elif section == "Single Prediction":
    st.title("🔍 Single Machine Failure Prediction")
    st.write("Enter operational sensor values to predict failure probability and risk category.")

    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)
        p_type = c1.selectbox("Product Type", ['L', 'M', 'H'], index=1)
        air_temp = c1.number_input("Air Temperature [K]", value=300.0, step=0.1)
        proc_temp = c2.number_input("Process Temperature [K]", value=310.0, step=0.1)
        rot_speed = c2.number_input("Rotational Speed [rpm]", value=1500, step=10)
        torque = c3.number_input("Torque [Nm]", value=40.0, step=0.5)
        tool_wear = c3.number_input("Tool Wear [min]", value=60, step=5)
        
        submitted = st.form_submit_button("⚡ Predict Failure")

    if submitted:
        input_df = pd.DataFrame([{
            'Type': p_type,
            'Air temperature [K]': air_temp,
            'Process temperature [K]': proc_temp,
            'Rotational speed [rpm]': rot_speed,
            'Torque [Nm]': torque,
            'Tool wear [min]': tool_wear
        }])
        
        # Calculate engineered features & run inference
        X_eval = add_features(input_df)
        pred_class = int(model.predict(X_eval)[0])
        pred_prob = float(model.predict_proba(X_eval)[0, 1])
        risk_label, risk_color = get_risk_category(pred_prob)

        st.markdown("---")
        st.subheader("📋 Prediction Results")
        r1, r2, r3 = st.columns(3)
        status_text = "🚨 FAILURE PREDICTED" if pred_class == 1 else "✅ NORMAL OPERATION"
        r1.metric("Predicted Status", status_text)
        r2.metric("Failure Probability", f"{pred_prob * 100:.2f}%")
        r3.markdown(f"**Assigned Risk Category:**\n### :{risk_color}[{risk_label}]")

        # Basic condition-based maintenance recommendations
        st.subheader("🛠️ Maintenance Recommendations")
        suggestions = []
        if tool_wear >= 200:
            suggestions.append("⚠️ **Tool Wear Critical:** Tool wear exceeds 200 minutes. Replace cutting tool insert immediately.")
        if (proc_temp - air_temp) < 8.6:
            suggestions.append("⚠️ **Cooling Degradation:** Temperature gradient is low (<8.6 K). Inspect coolant fluid and clean heat dissipation vents.")
        if torque > 60:
            suggestions.append("⚠️ **High Torque Strain:** Torque exceeds 60 Nm. Check spindle bearings and mechanical gear engagement.")
        if pred_prob < 0.10 and not suggestions:
            suggestions.append("✅ Machine is operating within safe nominal parameters. Continue standard routine inspection schedule.")
        
        for s in suggestions:
            st.write(s)

        st.caption("ℹ️ *Disclaimer: Risk categories and predictions are illustrative engineering estimations from a synthetic ML model.*")

# ==============================================================================
# SECTION C: BATCH CSV PREDICTION
# ==============================================================================
elif section == "Batch Prediction":
    st.title("📁 Batch Machine Records Prediction")
    st.write("Upload a CSV file containing machine operating conditions to predict fleet-wide failure risk.")

    uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        
        # Add engineered features and run predictions
        X_batch = add_features(batch_df)
        preds = model.predict(X_batch)
        probs = model.predict_proba(X_batch)[:, 1]

        batch_df['Predicted_Class'] = preds
        batch_df['Failure_Probability'] = np.round(probs, 4)
        batch_df['Risk_Category'] = [get_risk_category(p)[0] for p in probs]

        # Display Summary Cards
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Uploaded", len(batch_df))
        c2.metric("Normal Predictions", int((preds == 0).sum()))
        c3.metric("Predicted Failures", int((preds == 1).sum()))
        c4.metric("High Risk Units (≥30%)", sum(batch_df['Risk_Category'] == 'High Risk'))

        st.subheader("Prediction Results Table")
        st.dataframe(batch_df, use_container_width=True)

        # CSV Download Button
        csv_data = batch_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="💾 Download Predictions as CSV",
            data=csv_data,
            file_name="batch_machine_predictions.csv",
            mime="text/csv"
        )
