"""
Inference API with caching. Explainer does 7 forward passes ? cache per (pid, ts).
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import sys, os, functools

sys.path.insert(0, os.path.dirname(__file__))
from feature_fusion import EHR_FEATURES

WINDOW = 48
FEATURES = ["cgm_mgdl", "heart_rate", "steps"]
RNG = np.random.default_rng(42)


class FusionModel(nn.Module):
    def __init__(self, n_sensor=3, n_ehr=7, hidden=32, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(n_sensor, hidden, batch_first=True)
        self.ehr_mlp = nn.Sequential(
            nn.Linear(n_ehr, 16), nn.ReLU(), nn.Linear(16, 8), nn.ReLU()
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden + 8, 16), nn.ReLU(),
            nn.Linear(16, 1),
        )
    def forward(self, x_sensor, x_ehr):
        _, (h, _) = self.lstm(x_sensor)
        e = self.ehr_mlp(x_ehr)
        return self.head(torch.cat([h[-1], e], dim=-1)).squeeze(-1)


def synth_ehr_for(patient_severity):
    rows = []
    for pid, sev in patient_severity.items():
        rows.append({
            "patient_id": pid,
            "age": int(RNG.integers(35, 70)),
            "sex_m": int(RNG.integers(0, 2)),
            "bmi": round(22 + (sev - 0.4) * 5 + float(RNG.normal(0, 2)), 1),
            "hba1c": round(5.5 + (sev - 0.4) * 2.0 + float(RNG.normal(0, 0.3)), 1),
            "diabetes_years": int(RNG.integers(1, 15)),
            "on_insulin": int(RNG.random() < (sev - 0.3) * 0.6),
            "on_metformin": 1,
        })
    return pd.DataFrame(rows).set_index("patient_id")


class DigitalTwin:
    def __init__(self, models_dir="models", data_path="data/wearable_ts/synthetic_cgm.parquet"):
        self.data = pd.read_parquet(data_path)
        self.data["timestamp"] = pd.to_datetime(self.data["timestamp"])
        self.sev_map = self.data.groupby("patient_id")["severity"].first().to_dict()
        self.ehr = synth_ehr_for(self.sev_map)
        self.model = FusionModel()
        self.model.load_state_dict(torch.load(f"{models_dir}/fusion.pt", map_location="cpu"))
        self.model.eval()
        self.mu_s = np.load(f"{models_dir}/fusion_mu_s.npy")
        self.sd_s = np.load(f"{models_dir}/fusion_sd_s.npy")
        self.mu_e = np.load(f"{models_dir}/fusion_mu_e.npy")
        self.sd_e = np.load(f"{models_dir}/fusion_sd_e.npy")

    def patient_history(self, pid, end_time, window_min=240):
        end = pd.to_datetime(end_time)
        start = end - pd.Timedelta(minutes=window_min)
        g = self.data[(self.data.patient_id == pid) &
                      (self.data.timestamp >= start) &
                      (self.data.timestamp <= end)]
        return g.sort_values("timestamp").reset_index(drop=True)

    def _raw_forward(self, x_s_n, e_n):
        with torch.no_grad():
            logit = self.model(torch.tensor(x_s_n), torch.tensor(e_n))
        return float(torch.sigmoid(logit).item())

    @functools.lru_cache(maxsize=256)
    def _predict_cached(self, pid, ts_str):
        ts = pd.to_datetime(ts_str)
        hist = self.patient_history(pid, ts, window_min=5 * WINDOW)
        if len(hist) < WINDOW:
            return (None, ())
        x_s = hist[FEATURES].values.astype(np.float32)[-WINDOW:]
        e = self.ehr.loc[pid, EHR_FEATURES].values.astype(np.float32)
        x_s_n = (x_s[None] - self.mu_s) / self.sd_s
        e_n = (e[None] - self.mu_e) / self.sd_e
        base = self._raw_forward(x_s_n, e_n)

        drivers = []
        for i, name in enumerate(EHR_FEATURES):
            e_alt = e.copy(); e_alt[i] = 0.0
            e_alt_n = (e_alt[None] - self.mu_e) / self.sd_e
            r2 = self._raw_forward(x_s_n, e_alt_n)
            drivers.append((name, base - r2))
        cgm = x_s[:, 0]
        slope = float(cgm[-1] - cgm[-12]) if len(cgm) >= 12 else 0.0
        drivers.append(("cgm_slope_1h", slope / 100.0))
        drivers.sort(key=lambda x: abs(x[1]), reverse=True)
        return (base, tuple(drivers[:5]))

    def predict(self, pid, timestamp):
        ts_str = pd.to_datetime(timestamp).isoformat()
        risk, drivers = self._predict_cached(pid, ts_str)
        hist = self.patient_history(pid, timestamp, window_min=5 * WINDOW)
        return {"risk": risk, "drivers": list(drivers), "window": hist}


_INSTANCE = None

def get_twin():
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = DigitalTwin()
    return _INSTANCE


def predict(pid, timestamp):
    return get_twin().predict(pid, timestamp)


if __name__ == "__main__":
    twin = get_twin()
    pid = twin.data.patient_id.iloc[0]
    ts = twin.data[twin.data.patient_id == pid].timestamp.iloc[-1]
    out = predict(pid, ts)
    print(f"Patient: {pid} @ {ts}")
    print(f"Risk: {out['risk']:.3f}")
    for name, val in out["drivers"]:
        print(f"  {name:20s} {val:+.3f}")
    print("\nSecond call (should be instant ? cached):")
    import time
    t0 = time.time(); predict(pid, ts); print(f"  {time.time()-t0:.3f}s")
