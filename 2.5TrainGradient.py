import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn
import os
import math

# --- 1. Setup Paths & MLflow ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "Processed_Data", "cleaned_used_car_sales.csv")
mlflow.set_tracking_uri(f"file://{os.path.join(BASE_DIR, 'mlruns')}")

# --- 2. Load & Prepare Data ---
print("Loading cleaned data...")
df = pd.read_csv(DATA_PATH)

df_encoded = pd.get_dummies(df, columns=['make', 'model'], drop_first=True)
X = df_encoded.drop('price', axis=1)
y = df_encoded['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 3. Initialize MLflow Experiment ---
mlflow.set_experiment("Used_Car_Pricing_MVP")

print("Tuning Gradient Boosting Model... This will test multiple combinations!")

# --- 4. Hyperparameter Tuning (GridSearchCV) ---
with mlflow.start_run(run_name="gradient_boosting_tuned"):
    
    # The grid of parameters we want the computer to test
    param_grid = {
        "max_iter": [100, 200, 300],         # Number of sequential trees
        "max_depth": [5, 10, None],          # How deep the trees can go
        "learning_rate": [0.05, 0.1, 0.2]    # Step size for learning
    }
    
    # Initialize the base model
    base_model = HistGradientBoostingRegressor(random_state=42)
    
    # Set up the Grid Search (Testing 27 different combinations)
    grid_search = GridSearchCV(
        estimator=base_model, 
        param_grid=param_grid, 
        cv=3, 
        scoring='neg_mean_absolute_error', 
        n_jobs=-1 # Uses all cores of your Mac to speed this up!
    )
    
    # Run the tuning process
    grid_search.fit(X_train, y_train)
    
    # Extract the absolute best model from the search
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    # Make Predictions with the optimized model
    predictions = best_model.predict(X_test)
    
    # Calculate Metrics
    rmse = math.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    # Log the winning parameters and metrics to MLflow
    mlflow.log_params(best_params)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2", r2)
    
    # Log the model artifact
    mlflow.sklearn.log_model(best_model, "gradient_boost_model_tuned")
    
    # --- 5. Print Output to Terminal ---
    print("\n" + "="*45)
    print("🏆 TUNED GRADIENT BOOSTING COMPLETE 🏆")
    print("="*45)
    print("Best Parameters Found:")
    for key, value in best_params.items():
        print(f"  - {key}: {value}")
    print("-" * 45)
    print("Evaluation Metrics:")
    print(f"  - RMSE: ${rmse:,.2f}")
    print(f"  - MAE:  ${mae:,.2f}")
    print(f"  - R2:   {r2:.4f}")
    print("="*45)
