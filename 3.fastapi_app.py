from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import mlflow
import os

# 1. Setup paths to find your database and data
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
mlflow.set_tracking_uri(f"file://{os.path.join(BASE_DIR, 'mlruns')}")
DATA_PATH = os.path.join(BASE_DIR, "Processed_Data", "cleaned_used_car_sales.csv")

# 2. Automatically find and load your EXACT model!
print("Waking up the MLflow model...")
experiment = mlflow.get_experiment_by_name("Used_Car_Pricing_MVP")
df_runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

# Tell it to specifically grab the Tuned Gradient Boosting model run
best_run = df_runs[df_runs['tags.mlflow.runName'] == 'gradient_boosting_tuned'].iloc[0]

# Point it to the NEW folder name where the model actually lives!
model_uri = f"runs:/{best_run.run_id}/gradient_boost_model_tuned"
model = mlflow.sklearn.load_model(model_uri)

# 3. Learn the exact column structure the model expects
df_template = pd.read_csv(DATA_PATH)
template_encoded = pd.get_dummies(df_template, columns=['make', 'model'], drop_first=True)
expected_columns = template_encoded.drop(columns=['price']).columns

# 4. Initialize the API
app = FastAPI(title="Used Car Price Estimator API", version="1.0")

# 5. Define the exact input we expect from users
class CarFeatures(BaseModel):
    make: str
    model: str
    year: int
    mileage: int
    transmission: int  # 1 for Auto, 0 for Manual

@app.get("/")
def home():
    return {"message": "Welcome to the Used Car Pricing API! Go to /docs to test it."}

@app.post("/predict")
def predict_price(car: CarFeatures):
    # Convert user input into a dataframe
    input_data = pd.DataFrame([{
        'make': car.make,
        'model': car.model,
        'year': car.year,
        'mileage': car.mileage,
        'transmission': car.transmission
    }])

    # Convert their text into the 1s and 0s
    input_encoded = pd.get_dummies(input_data, columns=['make', 'model'])
    
    # Force the dataframe to match the exact columns the model expects
    input_encoded = input_encoded.reindex(columns=expected_columns, fill_value=0)

    # Ask the model for the price
    prediction = model.predict(input_encoded)[0]
    
    return {
        "status": "success",
        "predicted_price": round(prediction, 2),
        "currency": "USD"
    }

# --- The Play Button Hack ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)