# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# 🚨 MUST BE THE FIRST STREAMLIT COMMAND 🚨
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📉",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': "# Customer Churn Prediction App\nPredict if a customer will churn using ML models."
    }
)

# --------------------------
# 🚀 LOAD MODELS & PREPROCESSORS
# --------------------------
@st.cache_resource
def load_artifacts():
    artifacts = {}
    
    # Load models
    model_names = ["RandomForest", "XGBoost"]  # Add more if you trained them
    for name in model_names:
        path = f"models/{name}.pkl"
        if os.path.exists(path):
            with open(path, 'rb') as f:
                artifacts[name] = pickle.load(f)
        else:
            st.warning(f"⚠️ Model {name} not found at {path}")
    
    # Load scaler
    with open('models/scaler.pkl', 'rb') as f:
        artifacts['scaler'] = pickle.load(f)
    
    # Load label encoders
    with open('models/label_encoders.pkl', 'rb') as f:
        artifacts['label_encoders'] = pickle.load(f)
    
    # Load feature column order (CRITICAL)
    with open('models/feature_columns.pkl', 'rb') as f:
        artifacts['feature_columns'] = pickle.load(f)
    
    return artifacts

artifacts = load_artifacts()
final_model = artifacts.get("XGBoost", None) or artifacts.get("RandomForest", None)
if not final_model:
    st.error("❌ No model loaded. Check models folder.")
    st.stop()

scaler = artifacts['scaler']
label_encoders = artifacts['label_encoders']
feature_columns = artifacts['feature_columns']
# --------------------------
# 🎯 FEATURE ENGINEERING FUNCTION (MUST MATCH TRAINING) - DEBUGGED
# --------------------------
def engineer_features(df):
    st.write("🛠️ Inside engineer_features, df columns before:", df.columns.tolist()) # Debug
    required_cols = ['Usage Frequency', 'Support Calls', 'Payment Delay', 'Total Spend', 'Tenure']
    missing_base_cols = [col for col in required_cols if col not in df.columns]
    if missing_base_cols:
        raise KeyError(f"Missing base columns for feature engineering: {missing_base_cols}")

    # Ensure numerical types for calculations
    for col in required_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce') # Convert, handle non-numeric gracefully

    # Check for NaNs after conversion
    if df[required_cols].isnull().any().any():
        st.warning("⚠️ NaNs found in base columns after conversion. This might cause issues.")
        st.write(df[required_cols].isnull().sum())

    df['Engagement_Score'] = df['Usage Frequency'] / (df['Support Calls'] + 1)
    df['Delay_Ratio'] = df['Payment Delay'] / df['Tenure'].clip(lower=1)
    df['Spend_per_Month'] = df['Total Spend'] / df['Tenure'].clip(lower=1)
    st.write("🛠️ Inside engineer_features, df columns after:", df.columns.tolist()) # Debug
    return df

# --------------------------
# 🧮 PREPROCESS FUNCTION (MUST MATCH TRAINING PIPELINE) - DEBUGGED
# --------------------------
def preprocess_input(df_raw):
    df = df_raw.copy()
    st.write("🔍 Input columns received by preprocess_input:", df.columns.tolist())  # 👈 DEBUG

    # Encode categorical columns
    categorical_cols = ['Gender', 'Subscription Type', 'Contract Length']
    for col in categorical_cols:
        if col in df.columns: # Check if column exists before processing
            le = label_encoders[col]
            if df[col].dtype == 'object':
                df[col] = df[col].map(lambda s: le.transform([s])[0] if s in le.classes_ else -1)
            # else: column is already numeric, assume encoded or handle appropriately
        else:
            st.warning(f"⚠️ Categorical column '{col}' not found in input data.")

    # Feature engineering - This should create the missing columns
    df = engineer_features(df)

    # ✅ Validate required columns AFTER engineering
    st.write("🔍 Feature columns expected by model:", feature_columns) # Debug
    missing_cols = set(feature_columns) - set(df.columns)
    if missing_cols:
        raise ValueError(f"❌ Still missing required columns after engineering: {missing_cols}. Available columns: {df.columns.tolist()}")

    # Ensure correct column order and selection
    try:
        df = df[feature_columns] # Select only the columns the model expects, in the right order
    except KeyError as e:
        st.error(f"KeyError during column selection: {e}")
        st.write("Available columns in df:", df.columns.tolist())
        st.write("Expected feature columns:", feature_columns)
        raise e

    # Scale
    df_scaled = scaler.transform(df)
    df_processed = pd.DataFrame(df_scaled, columns=feature_columns)

    return df_processed
# --------------------------
# 🖥️ STREAMLIT UI — TITLE & INTRO
# --------------------------
st.title("📉 Customer Churn Prediction System")
st.markdown("""
> Predict whether a customer will churn using our trained ML model.  
> Upload a CSV or enter details manually below.
""")

# Model Selector
model_choice = st.selectbox("Select Model", [k for k in artifacts.keys() if k not in ['scaler', 'label_encoders', 'feature_columns']])
current_model = artifacts[model_choice]
st.info(f"✅ Using model: **{model_choice}**")

tab1, tab2 = st.tabs(["📝 Single Prediction", "📁 Batch Prediction (CSV)"])

# --------------------------
# TAB 1: SINGLE PREDICTION
# --------------------------
with tab1:
    st.subheader("Enter Customer Details")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.number_input("Age", min_value=18, max_value=100, value=35)
        tenure = st.number_input("Tenure (months)", min_value=1, max_value=120, value=12)
    
    with col2:
        usage_freq = st.number_input("Usage Frequency", min_value=0, max_value=50, value=10)
        support_calls = st.number_input("Support Calls", min_value=0, max_value=20, value=2)
        payment_delay = st.number_input("Payment Delay (days)", min_value=0, max_value=100, value=0)
    
    with col3:
        subscription = st.selectbox("Subscription Type", ["Basic", "Standard", "Premium"])
        contract = st.selectbox("Contract Length", ["Monthly", "Quarterly", "Annual"])
        total_spend = st.number_input("Total Spend ($)", min_value=0.0, value=500.0, step=10.0)
        last_interaction = st.number_input("Last Interaction (days ago)", min_value=0, max_value=365, value=30)  # 👈 NEW
    
    if st.button("🔮 Predict Churn", type="primary", use_container_width=True):
        # Create DataFrame
        input_data = pd.DataFrame([{
            'Gender': gender,
            'Age': age,
            'Tenure': tenure,
            'Usage Frequency': usage_freq,          # ✅ Must match exactly
            'Support Calls': support_calls,         # ✅
            'Payment Delay': payment_delay,         # ✅
            'Subscription Type': subscription,
            'Contract Length': contract,
            'Total Spend': total_spend,             # ✅
            'Last Interaction': last_interaction    # ✅ if required
        }])
        
        try:
            # Preprocess
            X_processed = preprocess_input(input_data)
            
            # Predict
            proba = current_model.predict_proba(X_processed)[0, 1]
            pred = current_model.predict(X_processed)[0]
            
            # Display
            st.divider()
            st.subheader("📊 Prediction Results")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Churn Probability", f"{proba:.2%}")
            with col2:
                status = "🚨 WILL CHURN" if pred == 1 else "✅ WILL NOT CHURN"
                st.metric("Prediction", status, delta=None)
            
            # Risk gauge
            st.progress(proba, text="Churn Risk Level")
            
            if proba > 0.7:
                st.error("🔴 High Risk Customer — Consider retention strategies!")
            elif proba > 0.4:
                st.warning("🟠 Medium Risk — Monitor usage and satisfaction.")
            else:
                st.success("🟢 Low Risk — Customer is likely to stay.")
                
        except Exception as e:
            st.error(f"❌ Error during prediction: {e}")

# --------------------------
# TAB 2: BATCH PREDICTION
# --------------------------
with tab2:
    st.subheader("Upload Customer Data (CSV)")
    st.markdown("""
    > CSV must contain these columns (case-sensitive):  
    > `CustomerID`, `Gender`, `Age`, `Tenure`, `Usage Frequency`, `Support Calls`,  
    > `Payment Delay`, `Subscription Type`, `Contract Length`, `Total Spend`, `Last Interaction`
    """)
    
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    
    if uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file)
            st.write("✅ Preview of uploaded data:")
            st.dataframe(df_upload.head())
            
            if st.button("🚀 Run Batch Prediction", type="primary"):
                with st.spinner("Processing..."):
                    # Store CustomerID for output
                    if 'CustomerID' not in df_upload.columns:
                        st.error("❌ CSV must contain 'CustomerID' column.")
                        st.stop()
                    
                    customer_ids = df_upload['CustomerID'].copy()
                    
                    # Preprocess
                    X_batch_processed = preprocess_input(df_upload)
                    
                    # Predict
                    proba_batch = current_model.predict_proba(X_batch_processed)[:, 1]
                    pred_batch = current_model.predict(X_batch_processed)
                    
                    # Create results
                    results_df = pd.DataFrame({
                        'CustomerID': customer_ids,
                        'Churn_Probability': proba_batch,
                        'Churn_Prediction': pred_batch,
                        'Risk_Level': pd.cut(proba_batch, 
                                           bins=[-0.1, 0.4, 0.7, 1.1], 
                                           labels=['Low', 'Medium', 'High'])
                    })
                    
                    st.divider()
                    st.subheader("📈 Batch Prediction Results")
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Summary
                    total = len(results_df)
                    churned = results_df['Churn_Prediction'].sum()
                    st.info(f"Out of {total} customers, **{churned} ({churned/total:.1%})** are predicted to churn.")
                    
                    # Download
                    csv = results_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "📥 Download Predictions as CSV",
                        csv,
                        "churn_predictions_batch.csv",
                        "text/csv",
                        key='download-csv'
                    )
                    
                    # Visualization
                    st.subheader("📉 Risk Distribution")
                    risk_counts = results_df['Risk_Level'].value_counts().reindex(['Low', 'Medium', 'High']).fillna(0)
                    st.bar_chart(risk_counts)
                    
        except Exception as e:
            st.error(f"❌ Error processing file: {e}")

# --------------------------
# ℹ️ FOOTER
# --------------------------
st.divider()
st.caption("Built with ❤️ using Streamlit | Trained on Customer Churn Dataset")