import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Set page configuration for an attractive layout
st.set_page_config(
    page_title="Transaction Classifier",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .predict-box {
        background-color: #ffffff;
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-top: 20px;
    }
    .result-title {
        font-weight: bold;
        font-size: 24px;
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Load the KNN Model safely
@st.cache_resource
def load_model():
    with open("KNN_Mode.pkl", "rb") as f:
        model = pickle.load(f)
    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading the model file: {e}")
    st.stop()

# App Header
st.title("🛡️ KNN Model Prediction Dashboard")
st.markdown("Enter the transaction attributes below to evaluate the classification status using the trained KNN model.")

st.hr()

# Form Layout structured into columns for clean design
with st.form("prediction_form"):
    st.subheader("📊 Input Features")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 💰 Transaction Basics")
        transaction_amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=50.0, step=0.5)
        num_items = st.number_input("Number of Items", min_value=1, value=1, step=1)
        velocity_score = st.number_input("Velocity Score", min_value=0.0, value=1.0, step=0.1)
        store_type = st.number_input("Store Type (ID/Code)", min_value=0, value=1, step=1)

    with col2:
        st.markdown("### 👤 Customer Profile")
        customer_age = st.number_input("Customer Age", min_value=0, max_value=120, value=30, step=1)
        prev_transactions = st.number_input("Previous Transactions Count", min_value=0, value=5, step=1)
        distance_from_home = st.number_input("Distance From Home (miles/km)", min_value=0.0, value=5.0, step=0.1)

    with col3:
        st.markdown("### ⏰ Context & Technicals")
        hour_of_day = st.slider("Hour of Day", min_value=0, max_value=23, value=12)
        device_type = st.number_input("Device Type (ID/Code)", min_value=0, value=1, step=1)
        network_quality = st.number_input("Network Quality (ID/Code)", min_value=0, value=1, step=1)
        
        st.markdown("**Flags:**")
        is_weekend = st.checkbox("Is Weekend?", value=False)
        is_first_transaction = st.checkbox("Is First Transaction?", value=False)

    # Submit Button
    submit_button = st.form_submit_button(label="⚡ Run KNN Prediction")

# Handle Prediction Execution
if submit_button:
    # Construct the array in the exact feature order expected by your model
    input_features = np.array([[
        transaction_amount,
        hour_of_day,
        1 if is_weekend else 0,
        num_items,
        customer_age,
        prev_transactions,
        distance_from_home,
        device_type,
        network_quality,
        1 if is_first_transaction else 0,
        store_type,
        velocity_score
    ]])

    # Make the prediction
    try:
        prediction = model.predict(input_features)
        
        # Check if probability estimation is supported by your KNN setup
        has_proba = hasattr(model, "predict_proba")
        if has_proba:
            prediction_proba = model.predict_proba(input_features)[0]

        # Display Result Section styled gracefully
        st.markdown('<div class="predict-box">', unsafe_allow_html=True)
        st.markdown('<p class="result-title">🎯 Model Assessment Result</p>', unsafe_allow_html=True)
        
        # Assuming binary categorization (e.g., 0 for Normal/Approved, 1 for Suspicious/Flagged)
        # Tweak labels and metrics to perfectly match your specific training target definition
        if prediction[0] == 1:
            st.error(f"### ⚠️ Class Flagged: {prediction[0]}")
            if has_proba:
                st.metric(label="Confidence Level", value=f"{prediction_proba[1] * 100:.2f}%")
        else:
            st.success(f"### ✅ Class Verified: {prediction[0]}")
            if has_proba:
                st.metric(label="Confidence Level", value=f"{prediction_proba[0] * 100:.2f}%")
                
        st.markdown('</div>', unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"An error occurred during prediction mapping: {e}")
