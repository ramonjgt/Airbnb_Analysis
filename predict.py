import joblib
import pandas as pd
from fastapi import FastAPI
import uvicorn
from typing import Dict, Any

app = FastAPI(title="Predicting number of booked/reserved days in trailing twelve months")

model_file = 'gradient_boosting_model.bin'
def cast_to_float(x):
    return x.astype(float)
pipeline = joblib.load(model_file)

# List of features expected by the model
FEATURES = [
    'listing_type', 'room_type', 'cancellation_policy',
    'superhost', 'instant_book', 'photos_count', 'guests', 'bedrooms',
    'beds', 'baths', 'min_nights', 'cleaning_fee', 'extra_guest_fee',
    'num_reviews', 'rating_overall', 'amenities_count'
]

def predict_single(client: Dict[str, Any]):
    # Ensure input is a DataFrame with one row
    df = pd.DataFrame([client], columns=FEATURES)
    result = pipeline.predict(df)[0]
    return float(result)

@app.post("/predict")
def predict(client: Dict[str, Any]):
    prediction = predict_single(client)
    return {
        "ttm_reserved_days": prediction
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)