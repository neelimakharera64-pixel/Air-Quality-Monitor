# utils/preprocessor.py
import numpy as np
import pandas as pd
import json
import pickle
from pathlib import Path
from sklearn.preprocessing import StandardScaler

MODEL_DIR  = Path(__file__).resolve().parent.parent / "model"
FEAT_PATH  = MODEL_DIR / "feature_names.json"
MODEL_PATH = MODEL_DIR / "best_model.pkl"

# ── Fallback feature list (from your training notebook) ───────
FALLBACK_FEATURES = [
    "PT08_S1_CO", "C6H6_GT", "PT08_S2_NMHC", "NOx_GT",
    "PT08_S3_NOx", "NO2_GT", "PT08_S4_NO2", "PT08_S5_O3",
    "T", "RH", "AH",
    "IsWeekend", "Season",
    "Hour_sin", "Hour_cos", "Month_sin", "Month_cos",
    "S1_S2_ratio", "S3_S4_ratio",
    "CO_rolling_mean_3h", "CO_rolling_std_3h",
    "CO_lag_1h", "CO_lag_2h",
]

# Cache
_FEATURE_NAMES = None
_SCALER        = None


def get_feature_names() -> list:
    """Load feature names from JSON or fallback list."""
    global _FEATURE_NAMES
    if _FEATURE_NAMES is not None:
        return _FEATURE_NAMES

    if FEAT_PATH.exists():
        with open(FEAT_PATH) as f:
            _FEATURE_NAMES = json.load(f)
        return _FEATURE_NAMES

    _FEATURE_NAMES = FALLBACK_FEATURES
    return _FEATURE_NAMES


def build_features(raw: dict) -> pd.DataFrame:
    """
    Convert raw user inputs into the full feature vector
    the model was trained on.

    raw keys expected:
        PT08_S1_CO, C6H6_GT, PT08_S2_NMHC, NOx_GT,
        PT08_S3_NOx, NO2_GT, PT08_S4_NO2, PT08_S5_O3,
        T, RH, AH,
        hour (int 0-23), month (int 1-12),
        is_weekend (0/1), season (0-3),
        co_lag1, co_lag2   (recent CO readings)
    """
    hour    = int(raw.get("hour",    12))
    month   = int(raw.get("month",   6))
    is_wknd = int(raw.get("is_weekend", 0))
    season  = int(raw.get("season",  1))
    co_lag1 = float(raw.get("co_lag1", 2.0))
    co_lag2 = float(raw.get("co_lag2", 2.0))

    s1  = float(raw["PT08_S1_CO"])
    s2  = float(raw["PT08_S2_NMHC"])
    s3  = float(raw["PT08_S3_NOx"])
    s4  = float(raw["PT08_S4_NO2"])
    s5  = float(raw["PT08_S5_O3"])

    feat = {
        # Raw sensor / environmental readings
        "PT08_S1_CO"         : s1,
        "C6H6_GT"            : float(raw.get("C6H6_GT", 5.0)),
        "PT08_S2_NMHC"       : s2,
        "NOx_GT"             : float(raw.get("NOx_GT", 200.0)),
        "PT08_S3_NOx"        : s3,
        "NO2_GT"             : float(raw.get("NO2_GT", 100.0)),
        "PT08_S4_NO2"        : s4,
        "PT08_S5_O3"         : s5,
        "T"                  : float(raw.get("T", 18.0)),
        "RH"                 : float(raw.get("RH", 50.0)),
        "AH"                 : float(raw.get("AH", 0.7)),
        # Time / calendar
        "IsWeekend"          : is_wknd,
        "Season"             : season,
        # Cyclic encodings
        "Hour_sin"           : np.sin(2 * np.pi * hour / 24),
        "Hour_cos"           : np.cos(2 * np.pi * hour / 24),
        "Month_sin"          : np.sin(2 * np.pi * month / 12),
        "Month_cos"          : np.cos(2 * np.pi * month / 12),
        # Sensor ratios
        "S1_S2_ratio"        : s1 / (s2 + 1e-9),
        "S3_S4_ratio"        : s3 / (s4 + 1e-9),
        # Lag / rolling (user provides recent CO values)
        "CO_rolling_mean_3h" : (co_lag1 + co_lag2) / 2,
        "CO_rolling_std_3h"  : float(np.std([co_lag1, co_lag2])),
        "CO_lag_1h"          : co_lag1,
        "CO_lag_2h"          : co_lag2,
    }

    feature_names = get_feature_names()
    df = pd.DataFrame([feat])

    # Align to exact training column order
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0.0

    extra = [c for c in df.columns if c not in feature_names]
    if extra:
        df = df.drop(columns=extra)

    df = df[feature_names].astype(float)

    # Scale using a fresh scaler fitted on this single row
    # (tree models are scale invariant, but we keep it for consistency)
    scaler = StandardScaler()
    df_sc  = pd.DataFrame(
        scaler.fit_transform(df),
        columns=feature_names,
    )
    return df_sc