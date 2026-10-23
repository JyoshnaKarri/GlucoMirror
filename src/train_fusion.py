"""
Fusion model (Row 3 - Digital Twin headline).
EHR now correlated with severity -> fusion should beat sensor-only.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

from labeling import label_spike
from feature_fusion import EHR_FEATURES

WINDOW = 48
FEATURES = ["cgm_mgdl", "heart_rate", "steps"]
BATCH = 128
EPOCHS = 15
LR = 1e-3
RNG = np.random.default_rng(42)


def synth_ehr_for(patient_severity: dict):
    rows = []
    for pid, sev in patient_severity.items():
        hba1c = round(5.5 + (sev - 0.4) * 2.0 + float(RNG.normal(0, 0.3)), 1)
        bmi = round(22 + (sev - 0.4) * 5 + float(RNG.normal(0, 2)), 1)
        insulin = int(RNG.random() < (sev - 0.3) * 0.6)
        rows.append({
            "patient_id": pid,
            "age": int(RNG.integers(35, 70)),
            "sex_m": int(RNG.integers(0, 2)),
            "bmi": bmi,
            "hba1c": hba1c,
            "diabetes_years": int(RNG.integers(1, 15)),
            "on_insulin": insulin,
            "on_metformin": 1,
        })
    return pd.DataFrame(rows).set_index("patient_id")


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
        fused = torch.cat([h[-1], e], dim=-1)
        return self.head(fused).squeeze(-1)


class DS(Dataset):
    def __init__(self, xs, xe, y):
        self.xs, self.xe, self.y = xs, xe, y
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.xs[i], self.xe[i], self.y[i]


def build(df, ehr):
    Xs, Xe, ys, pids = [], [], [], []
    for pid, g in df.groupby("patient_id"):
        g = g.sort_values("timestamp").reset_index(drop=True)
        feats = g[FEATURES].values.astype(np.float32)
        labels = g["label"].values.astype(np.float32)
        ehr_vec = ehr.loc[pid, EHR_FEATURES].values.astype(np.float32)
        for end in range(WINDOW, len(g)):
            Xs.append(feats[end - WINDOW:end])
            Xe.append(ehr_vec)
            ys.append(labels[end])
            pids.append(pid)
    return (np.array(Xs), np.array(Xe), np.array(ys), np.array(pids))


def main():
    torch.manual_seed(42); np.random.seed(42)
    df = pd.read_parquet("data/wearable_ts/synthetic_cgm.parquet")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = label_spike(df)
    severity_map = df.groupby("patient_id")["severity"].first().to_dict()
    ehr = synth_ehr_for(severity_map)
    print(f"Loaded {len(df):,} rows, {df.patient_id.nunique()} patients")

    Xs, Xe, y, pids = build(df, ehr)
    print(f"Windows: {len(Xs):,}  positive rate: {y.mean():.2%}")

    tr_pids, te_pids = train_test_split(np.unique(pids), test_size=0.3, random_state=42)
    tr_m = np.isin(pids, tr_pids); te_m = np.isin(pids, te_pids)

    mu_s = Xs[tr_m].mean(axis=(0,1), keepdims=True)
    sd_s = Xs[tr_m].std(axis=(0,1), keepdims=True) + 1e-6
    mu_e = Xe[tr_m].mean(axis=0, keepdims=True)
    sd_e = Xe[tr_m].std(axis=0, keepdims=True) + 1e-6
    Xs_n = (Xs - mu_s) / sd_s
    Xe_n = (Xe - mu_e) / sd_e

    tr = DS(torch.tensor(Xs_n[tr_m], dtype=torch.float32),
            torch.tensor(Xe_n[tr_m], dtype=torch.float32),
            torch.tensor(y[tr_m], dtype=torch.float32))
    te = DS(torch.tensor(Xs_n[te_m], dtype=torch.float32),
            torch.tensor(Xe_n[te_m], dtype=torch.float32),
            torch.tensor(y[te_m], dtype=torch.float32))
    tr_dl = DataLoader(tr, batch_size=BATCH, shuffle=True)
    te_dl = DataLoader(te, batch_size=BATCH, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = FusionModel().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(EPOCHS):
        model.train(); losses = []
        for xs, xe, yb in tr_dl:
            xs, xe, yb = xs.to(device), xe.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xs, xe)
            loss = loss_fn(logits, yb)
            loss.backward(); opt.step()
            losses.append(loss.item())
        print(f"Epoch {epoch+1:02d}/{EPOCHS}  loss {np.mean(losses):.4f}")

    model.eval(); probs, ys = [], []
    with torch.no_grad():
        for xs, xe, yb in te_dl:
            logits = model(xs.to(device), xe.to(device))
            probs.append(torch.sigmoid(logits).cpu().numpy())
            ys.append(yb.numpy())
    probs = np.concatenate(probs); ys = np.concatenate(ys)
    pred = (probs > 0.5).astype(int)

    print("\n=== FUSION MODEL (Row 3 - Digital Twin headline) ===")
    print(f"AUC        : {roc_auc_score(ys, probs):.3f}")
    print(f"Precision  : {precision_score(ys, pred, zero_division=0):.3f}")
    print(f"Recall     : {recall_score(ys, pred, zero_division=0):.3f}")
    print(f"F1         : {f1_score(ys, pred, zero_division=0):.3f}")

    torch.save(model.state_dict(), "models/fusion.pt")
    np.save("models/fusion_mu_s.npy", mu_s); np.save("models/fusion_sd_s.npy", sd_s)
    np.save("models/fusion_mu_e.npy", mu_e); np.save("models/fusion_sd_e.npy", sd_e)
    print("\nSaved model + normalization stats -> models/")


if __name__ == "__main__":
    main()
