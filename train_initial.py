import pandas as pd
import os
from models.pipeline import train_alloy_model

# Map the target model names to their respective datasets
DATA_MAPPING = {
    "NiTi_Family": "data/niti_data.csv",
    "NiAl_Family": "data/ni_al_data.csv",
    "NiCo_Family": "data/ni_co_data.csv"
}

def train_all_models():
    """Reads real CSV data and trains distinct models for each alloy system."""
    print("Starting model training pipeline...")
    os.makedirs("models/registry", exist_ok=True)
    
    for model_name, file_path in DATA_MAPPING.items():
        if os.path.exists(file_path):
            print(f"Loading data for {model_name} from {file_path}...")
            df = pd.read_csv(file_path)
            
            print(f"Training {model_name}...")
            # This calls your pipeline function and saves it as an individual .pkl
            train_alloy_model(df, model_name=model_name, save_path="models/registry")
            print(f"Success! Model saved to models/registry/{model_name}.pkl\n")
        else:
            print(f"Error: Could not find {file_path}. Skipping {model_name}.\n")

if __name__ == "__main__":
    train_all_models()
    print("All available models have been trained and registered.")