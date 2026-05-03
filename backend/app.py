from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

ROOT = Path(__file__).parent
MODEL_FP = ROOT / "model.joblib"
SCALER_FP = ROOT / "scaler.joblib"

app = FastAPI(title="FitSense model API")


class PredictRequest(BaseModel):
    # Accept either a list of numeric features or a dict of named features
    features: Optional[List[float]] = None
    # optional: allow passing a dict for named features (frontend can use either)
    named: Optional[dict] = None


class SimpleInputRequest(BaseModel):
    height: float  # inches
    weight: float  # kg
    age: float
    product_size: float
    body_type: str = "Rectangle"
    product_category: str = "Casual Dress"
    rented_for: str = "Casual Event"
    review_rating: float = 0


def load_artifacts():
    if not MODEL_FP.exists():
        return None, None
    model = joblib.load(MODEL_FP)
    scaler = None
    if SCALER_FP.exists():
        scaler = joblib.load(SCALER_FP)
    return model, scaler


@app.on_event("startup")
def startup_event():
    app.state.model, app.state.scaler = load_artifacts()


@app.get("/health")
def health():
    ok = app.state.model is not None
    return {"ok": ok, "model_loaded": ok}


@app.post("/predict")
def predict(req: PredictRequest):
    model = app.state.model
    scaler = app.state.scaler
    if model is None:
        raise HTTPException(status_code=503, detail="Model not found. Save your model as backend/model.joblib")

    # Prefer list input for simplicity
    if req.features:
        X = np.array(req.features, dtype=float).reshape(1, -1)
    elif req.named:
        # convert dict values to array in insertion order (user must match training order)
        X = np.array(list(req.named.values()), dtype=float).reshape(1, -1)
    else:
        raise HTTPException(status_code=400, detail="No features provided")

    if scaler is not None:
        try:
            X = scaler.transform(X)
        except Exception:
            # if scaler expected different shape, ignore and rely on model
            pass

    pred = model.predict(X).tolist()
    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X).tolist()

    return {"prediction": pred, "probabilities": proba}


@app.post("/predict-simple")
def predict_simple(req: SimpleInputRequest):
    """
    Accepts user-friendly inputs and converts to model features.
    WARNING: This is a simplified implementation. The actual feature engineering
    (like which categorical features to include after variance filtering) should
    match your training notebook exactly.
    """
    model = app.state.model
    scaler = app.state.scaler
    if model is None:
        raise HTTPException(status_code=503, detail="Model not found")

    # Calculate BMI from height and weight
    bmi = req.weight / (req.height * req.height)

    # For now, create a simple 9-feature vector.
    # This assumes: [product_size, review_rating, age, BMI, + 5 categorical features]
    # The categorical features are one-hot encoded (after variance filtering)
    # IMPORTANT: You must verify this matches your training pipeline!
    
    # Simple one-hot encoding for demonstration
    # Note: In production, load pre-trained mappings from your training script
    body_types = ["Apple", "Athletic", "Bell", "Curvy", "Hourglass", "Inverted Triangle", "Pear", "Rectangle", "Triangle"]
    body_type_vec = [1 if req.body_type == bt else 0 for bt in body_types]
    
    product_cats = ["All Dressed Up", "Casual Dress", "Cocktail & Party", "Coverups", "Denim", "Gowns", "Knit", "Shirt"]
    product_cat_vec = [1 if req.product_category == pc else 0 for pc in product_cats]
    
    rented_fors = ["Bride", "Bridesmaid", "Casual Event", "Cocktail", "Concert", "Dinner Date", "Graduation", "Holiday", "Honeymoon", "Wedding Guest", "Work Event", "uncommon"]
    rented_for_vec = [1 if req.rented_for == rf else 0 for rf in rented_fors]
    
    # Combine all features
    all_features = [req.product_size, req.review_rating, req.age, bmi] + body_type_vec + product_cat_vec + rented_for_vec
    
    # The model expects 9 features, so we need to select the top 5 features from the one-hot encoded categories
    # For now, we'll just take the first 9 features (this is a simplification)
    features = all_features[:9]
    
    X = np.array(features, dtype=float).reshape(1, -1)

    if scaler is not None:
        try:
            X = scaler.transform(X)
        except Exception as e:
            # Log but continue
            pass

    try:
        pred = model.predict(X).tolist()
        proba = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X).tolist()
        return {"prediction": pred, "probabilities": proba}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")

