import pandas as pd
import numpy as np
import os
from models.pipeline import train_alloy_model, FEATURES

def create_dummy_data():
    """Generates 100 rows of synthetic NiTi data for testing."""
    print("Generating dummy dataset...")
    np.random.seed(42)
    n_samples = 100
    
    # Initialize all features to 0
    data = {feat: np.zeros(n_samples) for feat in FEATURES}
    
    # Populate dominant elements (Ni and Ti) and properties
    data['Ni'] = np.random.uniform(49.0, 51.0, n_samples)
    data['Ti'] = 100.0 - data['Ni']
    data['Cooling_Rate'] = np.random.uniform(5.0, 20.0, n_samples)
    data['Heating_Rate'] = np.random.uniform(5.0, 20.0, n_samples)
    data['Density'] = np.random.uniform(6.4, 6.5, n_samples)
    
    df = pd.DataFrame(data)
    
    # Create fake logical thermodynamic targets
    df['AF'] = np.random.uniform(50, 100, n_samples)
    df['AS'] = df['AF'] - 10
    df['MS'] = df['AS'] - 20
    df['MF'] = df['MS'] - 15
    
    # Save the dummy data to the data folder
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/niti_dummy_data.csv", index=False)
    return df

if __name__ == "__main__":
    # 1. Create Data
    df = create_dummy_data()
    
    # 2. Train Model
    print("Training the NiTi_Family model...")
    os.makedirs("models/registry", exist_ok=True)
    
    # This matches the model name expected by core/router.py
    train_alloy_model(df, model_name="NiTi_Family", save_path="models/registry")
    print("Success! Model saved to models/registry/NiTi_Family.pkl")