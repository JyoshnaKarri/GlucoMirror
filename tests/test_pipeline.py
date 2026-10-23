import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
import pandas as pd
import numpy as np
from labeling import label_spike, SPIKE_THRESHOLD


def test_label_has_patient_id():
    df = pd.DataFrame({"patient_id": ["P1"]*60,
                       "timestamp": pd.date_range("2026-01-01", periods=60, freq="5min"),
                       "cgm_mgdl": [100.0]*30 + [200.0]*30})
    out = label_spike(df)
    assert "patient_id" in out.columns
    assert "label" in out.columns


def test_label_is_correct():
    df = pd.DataFrame({"patient_id": ["P1"]*60,
                       "timestamp": pd.date_range("2026-01-01", periods=60, freq="5min"),
                       "cgm_mgdl": [100.0]*30 + [200.0]*30})
    out = label_spike(df).sort_values("timestamp").reset_index(drop=True)
    assert out.loc[0, "label"] == 0
    assert out.loc[10, "label"] == 1


def test_data_file_exists():
    assert os.path.exists("data/wearable_ts/synthetic_cgm.parquet")


def test_data_has_severity_column():
    df = pd.read_parquet("data/wearable_ts/synthetic_cgm.parquet")
    assert "severity" in df.columns


def test_spike_rate_is_realistic():
    df = pd.read_parquet("data/wearable_ts/synthetic_cgm.parquet")
    rate = (df["cgm_mgdl"] > SPIKE_THRESHOLD).mean()
    assert 0.02 <= rate <= 0.40


def test_models_exist():
    for f in ["fusion.pt", "lstm_sensor.pt",
              "fusion_mu_s.npy", "fusion_sd_s.npy",
              "fusion_mu_e.npy", "fusion_sd_e.npy"]:
        assert os.path.exists(f"models/{f}"), f"Missing models/{f}"