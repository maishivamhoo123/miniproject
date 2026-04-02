from fastapi import FastAPI, HTTPException
import joblib
from api.schemas import AlloyInput
from core.router import AlloyRouter
from core.physics import enforce_thermodynamics
from models.pipeline import FEATURES, TARGETS, predict_with_confidence
import pandas as pd

app = FastAPI(title="SMA Informatics Platform", version="1.0")

# In-memory model cache to prevent reloading from disk on every request
MODEL_CACHE = {}

def get_model(model_name: str):
    if model_name not in MODEL_CACHE:
        try:
            MODEL_CACHE[model_name] = joblib.load(f"models/registry/{model_name}.pkl")
        except FileNotFoundError:
            raise HTTPException(status_code=500, detail=f"Model {model_name} not found.")
    return MODEL_CACHE[model_name]

@app.post("/predict")
async def predict_transformation(data: AlloyInput):
    # 1. Convert input to dictionary
    input_dict = data.model_dump()
    
    # 2. Route to correct model system
    alloy_system = AlloyRouter.identify_system(input_dict)
    model = get_model(alloy_system)
    
    # 3. Format input for pipeline
    X_df = pd.DataFrame([[input_dict.get(feat, 0.0) for feat in FEATURES]], columns=FEATURES)
    
    # 4. Predict with confidence
    preds_array, std_devs = predict_with_confidence(model, X_df)
    
    # 5. Format results
    raw_predictions = {TARGETS[i]: float(preds_array[i]) for i in range(len(TARGETS))}
    confidence_intervals = {f"{TARGETS[i]}_std": float(std_devs[i]) for i in range(len(TARGETS))}
    
    # 6. Apply Physics logic (TSPAN & validity)
    final_predictions = enforce_thermodynamics(raw_predictions)
    
    return {
        "alloy_system": alloy_system,
        "predictions": final_predictions,
        "confidence_std_dev": confidence_intervals,
        "warnings": "Extrapolation warning: Input composition heavily deviates from training bounds." if std_devs[0] > 15.0 else None
    }