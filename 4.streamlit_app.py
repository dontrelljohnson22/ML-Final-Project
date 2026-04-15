import streamlit as st
import pandas as pd
import mlflow
import os
import datetime

# --- 1. Load the Model & Data Structure ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
mlflow.set_tracking_uri(f"file://{os.path.join(BASE_DIR, 'mlruns')}")
DATA_PATH = os.path.join(BASE_DIR, "Processed_Data", "cleaned_used_car_sales.csv")

@st.cache_resource 
def load_model_and_template():
    experiment = mlflow.get_experiment_by_name("Used_Car_Pricing_MVP")
    df_runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    
    # Grab the winning Gradient Boosting model
    best_run = df_runs[df_runs['tags.mlflow.runName'] == 'gradient_boosting_tuned'].iloc[0]
    model_uri = f"runs:/{best_run.run_id}/gradient_boost_model_tuned"
    model = mlflow.sklearn.load_model(model_uri)
    
    # Learn the expected columns
    df_template = pd.read_csv(DATA_PATH)
    template_encoded = pd.get_dummies(df_template, columns=['make', 'model'], drop_first=True)
    expected_columns = template_encoded.drop(columns=['price']).columns
    
    makes = sorted(df_template['make'].unique().tolist())
    models = sorted(df_template['model'].unique().tolist())
    
    return model, expected_columns, makes, models

model, expected_columns, available_makes, available_models = load_model_and_template()

# --- 2. Build the User Interface ---
st.title("🚗 Used Car Price Estimator")
st.write("Welcome to our Final Project! Enter vehicle details below to get an instant, data-driven price estimate.")

st.header("Vehicle Specifications")
col1, col2 = st.columns(2)

with col1:
    user_make = st.selectbox("Make (Manufacturer)", available_makes)
    user_year = st.slider("Manufacture Year", min_value=2000, max_value=2024, value=2018)

with col2:
    user_model = st.selectbox("Model", available_models)
    user_transmission = st.radio("Transmission", ["Automatic", "Manual"])

user_mileage = st.number_input("Mileage (KM)", min_value=0, value=50000, step=5000)

# --- 3. The Prediction Engine & Usage Tracker ---
if st.button("Estimate Price", type="primary"):
    with st.spinner("Calculating market value using Gradient Boosting..."):
        
        trans_val = 1 if user_transmission == "Automatic" else 0
        input_data = pd.DataFrame([{
            'make': user_make,
            'model': user_model,
            'year': user_year,
            'mileage': user_mileage,
            'transmission': trans_val
        }])

        input_encoded = pd.get_dummies(input_data, columns=['make', 'model'])
        input_encoded = input_encoded.reindex(columns=expected_columns, fill_value=0)

        prediction = model.predict(input_encoded)[0]

        # --- USAGE TRACKER (Saves to CSV) ---
        log_file = os.path.join(BASE_DIR, "usage_logs.csv")
        log_data = pd.DataFrame([{
            "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Make": user_make,
            "Model": user_model,
            "Year": user_year,
            "Mileage": user_mileage,
            "Transmission": user_transmission,
            "Predicted_Price": round(prediction, 2)
        }])
        
        # This creates the file if it doesn't exist, or adds a new row if it does
        log_data.to_csv(log_file, mode='a', header=not os.path.exists(log_file), index=False)

        st.success("Analysis Complete!")
        st.metric(label="Estimated Fair Market Value", value=f"${prediction:,.2f}")
