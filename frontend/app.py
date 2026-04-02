import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, max_error

# --- Configuration ---
st.set_page_config(page_title="SMA Informatics Platform", layout="wide")
API_URL = "http://127.0.0.1:8000/predict"

# --- UI Header ---
st.title("🧪 Shape Memory Alloy (SMA) Prediction Platform")
st.markdown("Enter the elemental composition and processing parameters to predict thermodynamic properties.")

# --- Layout: Sidebar for Inputs ---
with st.sidebar:
    st.header("1. Input Parameters")
    
    st.subheader("Base Elements (Atomic %)")
    ni = st.number_input("Nickel (Ni)", min_value=0.0, max_value=100.0, value=50.5, step=0.1)
    ti = st.number_input("Titanium (Ti)", min_value=0.0, max_value=100.0, value=49.5, step=0.1)
    
    with st.expander("Alloying Elements (Ternary/Quaternary)"):
        cu = st.number_input("Copper (Cu)", 0.0, 100.0, 0.0)
        hf = st.number_input("Hafnium (Hf)", 0.0, 100.0, 0.0)
        pd_el = st.number_input("Palladium (Pd)", 0.0, 100.0, 0.0)
        # We set the rest to 0 automatically for the API payload
    
    st.subheader("Processing Conditions")
    cool_rate = st.number_input("Cooling Rate (°C/min)", min_value=0.1, value=10.0)
    heat_rate = st.number_input("Heating Rate (°C/min)", min_value=0.1, value=10.0)
    density = st.number_input("Density (g/cm³)", min_value=0.1, value=6.45)
    
    predict_button = st.button("Predict Properties", type="primary", use_container_width=True)

# --- Layout: Main Body Tabs ---
tab1, tab2 = st.tabs(["📊 Prediction Results", "⚙️ Model Diagnostics"])

# --- TAB 1: Predictions ---
with tab1:
    if predict_button:
        # 1. Build Payload
        payload = {
            "Ni": ni, "Ti": ti, "Cu": cu, "Hf": hf, "Pd": pd_el,
            "Cooling_Rate": cool_rate, "Heating_Rate": heat_rate, "Density": density,
            "Ag": 0, "Al": 0, "Au": 0, "Cd": 0, "Co": 0, "Fe": 0, "Mn": 0, 
            "Nb": 0, "Pt": 0, "Ru": 0, "Si": 0, "Ta": 0, "Zn": 0, "Zr": 0
        }
        
        # 2. Call the FastAPI Backend
        with st.spinner("Calculating thermodynamics..."):
            try:
                response = requests.post(API_URL, json=payload)
                response.raise_for_status()
                data = response.json()
                
                # 3. Display Results safely
                system_name = data.get('alloy_system', data.get('routed_system', 'Unknown'))
                st.success(f"Routed successfully to: **{system_name}**")
                
                if data.get('warnings'):
                    st.warning(data['warnings'])
                
                preds = data.get('predictions', {})
                conf = data.get('confidence_std_dev', data.get('confidence_intervals', {}))
                
                # Big Number Display
                st.subheader("Predicted Transformation Temperatures")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Austenite Finish (AF)", f"{preds.get('AF', 0)} °C", f"± {conf.get('AF_std', 0)} °C")
                col2.metric("Austenite Start (AS)", f"{preds.get('AS', 0)} °C", f"± {conf.get('AS_std', 0)} °C")
                col3.metric("Martensite Start (MS)", f"{preds.get('MS', 0)} °C", f"± {conf.get('MS_std', 0)} °C")
                col4.metric("Martensite Finish (MF)", f"{preds.get('MF', 0)} °C", f"± {conf.get('MF_std', 0)} °C")
                
                st.info(f"**Thermal Transformation Span (TSPAN):** {preds.get('TSPAN', 0)} °C")
                
            except requests.exceptions.RequestException as e:
                st.error(f"Failed to connect to the backend API. Is FastAPI running? Error: {e}")
    else:
        st.info("👈 Enter parameters in the sidebar and click 'Predict' to see results.")

# --- TAB 2: Model Performance Graphs & Metrics ---
with tab2:
    st.header("Global Model Performance Diagnostics")
    
    # Check if we have data to plot (Adjust paths if running from a different directory)
    model_path = "../models/registry/NiTi_Family.pkl"
    data_path = "../data/niti_dummy_data.csv"
    
    # Fallback paths just in case Streamlit is run from the root directory instead of the frontend folder
    if not os.path.exists(model_path):
        model_path = "models/registry/NiTi_Family.pkl"
        data_path = "data/niti_dummy_data.csv"

    if os.path.exists(model_path) and os.path.exists(data_path):
        try:
            # Load model and data
            model = joblib.load(model_path)
            df = pd.read_csv(data_path)
            
            # Predict on the dataset
            X = df.drop(columns=['AF', 'AS', 'MF', 'MS', 'TSPAN'], errors='ignore')
            y_actual = df[['AF', 'AS', 'MF', 'MS']]
            y_pred = model.predict(X)
            
            # --- 1. Model Architecture Metadata ---
            st.subheader("1. Architecture & Training Data")
            rf_estimator = model.named_steps['regressor'].estimators_[0]
            
            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            m_col1.metric("Algorithm", "Random Forest")
            m_col2.metric("Decision Trees", rf_estimator.n_estimators)
            m_col3.metric("Input Features", rf_estimator.n_features_in_)
            m_col4.metric("Dataset Size", f"{len(df)} rows")
            
            st.divider()

            # --- 2. Advanced Error Metrics ---
            st.subheader("2. Austenite Finish (AF) Accuracy Metrics")
            
            # Calculate metrics
            mae_af = mean_absolute_error(y_actual['AF'], y_pred[:, 0])
            rmse_af = np.sqrt(mean_squared_error(y_actual['AF'], y_pred[:, 0]))
            max_err_af = max_error(y_actual['AF'], y_pred[:, 0])
            r2_af = r2_score(y_actual['AF'], y_pred[:, 0])
            
            e_col1, e_col2, e_col3, e_col4 = st.columns(4)
            e_col1.metric("Mean Absolute Error (MAE)", f"± {mae_af:.2f} °C", help="Average error magnitude")
            e_col2.metric("Root Mean Squared Error", f"± {rmse_af:.2f} °C", help="Penalizes larger errors more heavily")
            e_col3.metric("Max Error (Worst Case)", f"{max_err_af:.2f} °C", help="The single most inaccurate prediction in the dataset")
            e_col4.metric("R² Score", f"{r2_af:.3f}", help="1.0 is a perfect prediction model")
            
            st.divider()
            
            # --- 3. Visualizations ---
            st.subheader("3. Predictive Visualizations")
            col1, col2 = st.columns(2)
            
            # Graph 1: Parity Plot (AF)
            with col1:
                fig1, ax1 = plt.subplots(figsize=(6, 4))
                sns.scatterplot(x=y_actual['AF'], y=y_pred[:, 0], alpha=0.6, ax=ax1)
                ax1.plot([y_actual['AF'].min(), y_actual['AF'].max()], [y_actual['AF'].min(), y_actual['AF'].max()], 'r--')
                ax1.set_title("Parity Plot: Austenite Finish (AF)")
                ax1.set_xlabel("Actual AF (°C)")
                ax1.set_ylabel("Predicted AF (°C)")
                st.pyplot(fig1)

            # Graph 2: Parity Plot (MF)
            with col2:
                fig2, ax2 = plt.subplots(figsize=(6, 4))
                sns.scatterplot(x=y_actual['MF'], y=y_pred[:, 2], alpha=0.6, color='orange', ax=ax2)
                ax2.plot([y_actual['MF'].min(), y_actual['MF'].max()], [y_actual['MF'].min(), y_actual['MF'].max()], 'r--')
                ax2.set_title("Parity Plot: Martensite Finish (MF)")
                ax2.set_xlabel("Actual MF (°C)")
                ax2.set_ylabel("Predicted MF (°C)")
                st.pyplot(fig2)

            # Graph 3: Feature Importance
            with col1:
                importances = rf_estimator.feature_importances_
                feat_names = X.columns
                importance_df = pd.DataFrame({"Feature": feat_names, "Importance": importances}).sort_values(by="Importance", ascending=False).head(10)
                
                fig3, ax3 = plt.subplots(figsize=(6, 4))
                sns.barplot(data=importance_df, x="Importance", y="Feature", hue="Feature", legend=False, palette="viridis", ax=ax3)
                ax3.set_title("Top 10 Feature Importances")
                ax3.set_xlabel("Relative Importance")
                st.pyplot(fig3)

            # Graph 4: Residual Distribution
            with col2:
                residuals = y_actual['AF'] - y_pred[:, 0]
                fig4, ax4 = plt.subplots(figsize=(6, 4))
                sns.histplot(residuals, kde=True, color='purple', ax=ax4)
                ax4.set_title("Residual Distribution (AF Error Spread)")
                ax4.set_xlabel("Prediction Error (°C)")
                ax4.set_ylabel("Frequency")
                st.pyplot(fig4)

        except Exception as e:
            st.warning(f"Could not generate graphs. Error: {e}")
    else:
        st.warning("Model or data files not found. Ensure you have run train_initial.py.")