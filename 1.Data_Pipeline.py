import pandas as pd
import os

def clean_data(input_path, output_path):
    print("Starting data pipeline...")
    
    # 1. Ingestion
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} rows from raw data.")

    # 2. Rename messy Kaggle columns to clean, simple names
    df = df.rename(columns={
        'Manufacturer Name': 'make',
        'Car Name': 'model',
        'Manufactured Year': 'year',
        'Sold Price-$': 'price',
        'Mileage-KM': 'mileage',
        'Gearbox': 'transmission'
    })

    # Keep only the essential columns we need for predicting price
    columns_to_keep = ['make', 'model', 'year', 'price', 'mileage', 'transmission']
    df = df[columns_to_keep]

    # --- DIAGNOSTIC GUARDRAILS ---
    # Test 1: Impossible Years
    starting_rows = len(df)
    df = df[(df['year'] >= 2000) & (df['year'] <= 2024)]
    print(f"Rows lost to Year rule (not 2000-2024): {starting_rows - len(df)}")
    
    # Test 2: Negative/Zero Math
    starting_rows = len(df)
    df = df[(df['mileage'] >= 0) & (df['price'] > 0)]
    print(f"Rows lost to Negative Math rule (Mileage/Price): {starting_rows - len(df)}")
    
    # Test 3: Missing Blank Data
    starting_rows = len(df)
    df = df.dropna(subset=['make', 'model', 'year', 'price', 'mileage'])
    print(f"Rows lost to Missing Data (Blank cells): {starting_rows - len(df)}")
    # ------------------------------

    # 4. Processing
    # Convert categorical values to binary (e.g., Automatic = 1, Manual = 0)
    if 'transmission' in df.columns:
        df['transmission_is_auto'] = df['transmission'].apply(lambda x: 1 if str(x).lower() == 'automatic' else 0)
        df = df.drop(columns=['transmission'])

    print(f"Data cleaned. {len(df)} rows remaining after validation.")

    # 5. Storage (Save to Processed_Data folder)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Clean data saved to {output_path}")

if __name__ == "__main__":
    # This magically finds the exact folder where this script lives
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # We safely glue that exact folder path to your data folders
    RAW_DATA_PATH = os.path.join(BASE_DIR, "Raw_Data", "used_car_sales.csv")
    PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "Processed_Data", "cleaned_used_car_sales.csv")
    
    clean_data(RAW_DATA_PATH, PROCESSED_DATA_PATH)