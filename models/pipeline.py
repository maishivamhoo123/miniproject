import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
import joblib

# Define feature columns strictly (No TSPAN)
FEATURES = ['Ag', 'Al', 'Au', 'Cd', 'Co', 'Cu', 'Fe', 'Hf', 'Mn', 'Nb', 
            'Ni', 'Pd', 'Pt', 'Ru', 'Si', 'Ta', 'Ti', 'Zn', 'Zr', 
            'Cooling_Rate', 'Heating_Rate', 'Density']
TARGETS = ['AF', 'AS', 'MF', 'MS']

def train_alloy_model(df: pd.DataFrame, model_name: str, save_path: str):
    X = df[FEATURES]
    y = df[TARGETS]

    # Preprocessing & Model Pipeline
    # MultiOutputRegressor handles the 4 thermodynamic targets simultaneously
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', MultiOutputRegressor(RandomForestRegressor(n_estimators=200, random_state=42)))
    ])

    pipeline.fit(X, y)
    
    # Save model
    joblib.dump(pipeline, f"{save_path}/{model_name}.pkl")
    return pipeline

def predict_with_confidence(model, X_input: pd.DataFrame):
    """Calculates point predictions and uncertainty via Tree Variance."""
    rf_multi = model.named_steps['regressor']
    scaler = model.named_steps['scaler']
    X_scaled = scaler.transform(X_input)
    
    predictions = model.predict(X_input)
    
    # Calculate variance across trees for confidence estimation
    variances = []
    for i in range(len(TARGETS)): # For each target
        rf = rf_multi.estimators_[i]
        tree_preds = np.array([tree.predict(X_scaled) for tree in rf.estimators_])
        std_dev = np.std(tree_preds, axis=0)
        variances.append(std_dev[0])
        
    return predictions[0], variances