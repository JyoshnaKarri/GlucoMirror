"""
Fuses static EHR features with dynamic wearable windows.
EHR → vector (one per patient)
Wearable → last-N-min window → LSTM/MLP input
Fusion = [EHR embedding ⊕ sensor embedding]
"""
import numpy as np
import pandas as pd

EHR_FEATURES = ["age", "sex_m", "bmi", "hba1c", "diabetes_years", "on_insulin", "on_metformin"]

def load_ehr(path: str) -> pd.DataFrame:
    """One row per patient. Columns must include EHR_FEATURES."""
    df = pd.read_csv(path)
    missing = [c for c in EHR_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"EHR missing columns: {missing}")
    return df.set_index("patient_id")

def ehr_to_vector(ehr_row: pd.Series) -> np.ndarray:
    return ehr_row[EHR_FEATURES].astype(float).values

def wearable_window(df: pd.DataFrame, patient_id: str, at_time, window_min: int = 240,
                    features=("cgm_mgdl", "heart_rate", "steps", "sleep_stage")):
    """
    Returns a (window_len, n_features) array for one patient ending at at_time.
    Missing timestamps are forward-filled; leading NaNs are zero-filled.
    """
    end = pd.to_datetime(at_time)
    start = end - pd.Timedelta(minutes=window_min)
    sl = df[(df.patient_id == patient_id) & (df.timestamp >= start) & (df.timestamp <= end)]
    sl = sl.sort_values("timestamp")[list(features)].ffill().fillna(0.0)
    return sl.values

def fuse(ehr_vec: np.ndarray, sensor_window: np.ndarray,
         sensor_encoder) -> np.ndarray:
    """
    sensor_encoder: callable(window)->1D embedding (LSTM hidden or mean-pool).
    Returns concatenated fused vector.
    """
    sensor_emb = sensor_encoder(sensor_window)
    return np.concatenate([ehr_vec, sensor_emb], axis=-1)