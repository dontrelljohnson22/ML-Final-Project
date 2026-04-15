import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn

# Dynamically find the exact folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "Processed_Data", "cleaned_used_car_sales.csv")

def train_model():
    print("Loading clean data...")
    df = pd.read_csv(DATA_PATH)

    # 1. Feature Engineering (Keeping this exactly the same so Streamlit doesn't break!)
    df_encoded = pd.get_dummies(df, columns=['make', 'model'], drop_first=True)

    # 2. Split into Features (X) and Target (y)
    X = df_encoded.drop(columns=['price'])
    y = df_encoded['price']

    # 3. Train/Test Split 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Set up MLflow Database
    mlflow.set_tracking_uri(f"file://{os.path.join(BASE_DIR, 'mlruns')}")
    mlflow.set_experiment("Used_Car_Pricing_MVP")

    # Name this run specifically so you can spot the upgrade in MLflow!
    with mlflow.start_run(run_name="RandomForest_Tuned_v1"):
        print("Starting Hyperparameter Tuning... (This might take 10-20 seconds)")
        
        # 5. Define the base model and the settings we want to test
        base_rf = RandomForestRegressor(random_state=42)
        
        # We are forcing the computer to test every combination of these settings
        param_grid = {
            'n_estimators': [100, 200, 300],   # How many trees to build
            'max_depth': [None, 10, 20],       # How deep the trees can grow
            'min_samples_split': [2, 5]        # Strictness of the math
        }
        
        # Run the grid search
        grid_search = GridSearchCV(
            estimator=base_rf, 
            param_grid=param_grid, 
            cv=3, 
            n_jobs=-1, # This tells your Mac to use all of its processing cores!
            scoring='neg_mean_squared_error'
        )
        
        grid_search.fit(X_train, y_train)
        
        # Extract the absolute smartest model from the search
        best_rf_model = grid_search.best_estimator_
        print(f"\n✅ Optimal settings found: {grid_search.best_params_}")

        # 6. Make Predictions using the new, smarter brain
        predictions = best_rf_model.predict(X_test)

        # 7. Calculate Technical Performance Metrics
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        print(f"\nModel Performance (Tuned):")
        print(f"  RMSE: ${rmse:.2f}")
        print(f"  MAE:  ${mae:.2f}")
        print(f"  R^2:  {r2:.4f}\n")

        # 8. Log Everything to MLflow Registry
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        
        # Save the tuned brain so Streamlit can use it!
        mlflow.sklearn.log_model(best_rf_model, "random_forest_model")
        print("✅ Tuned Random Forest Model successfully logged to MLflow!")

if __name__ == "__main__":
    train_model()