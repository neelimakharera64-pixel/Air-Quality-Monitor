# utils/predictor.py
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "best_model.pkl"


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}\n"
            "Place best_model.pkl in the model/ folder."
        )
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def predict_co(model, df_processed: pd.DataFrame) -> dict:
    """Predict CO concentration and return result dict."""
    pred = float(np.clip(model.predict(df_processed)[0], 0, None))

    # Air quality tier based on WHO / EU standards for CO (mg/m³)
    if pred < 2.0:
        level = "Good"
        color = "green"
        icon  = "🟢"
        advice = "Air quality is good. No health concerns."
    elif pred < 4.0:
        level = "Moderate"
        color = "orange"
        icon  = "🟡"
        advice = "Acceptable air quality. Sensitive groups should limit exposure."
    elif pred < 7.0:
        level = "Poor"
        color = "red"
        icon  = "🔴"
        advice = "Unhealthy for sensitive groups. Consider reducing outdoor activity."
    else:
        level = "Hazardous"
        color = "darkred"
        icon  = "⛔"
        advice = "Very unhealthy. Avoid outdoor exposure."

    return {
        "co_pred" : round(pred, 3),
        "level"   : level,
        "color"   : color,
        "icon"    : icon,
        "advice"  : advice,
    }