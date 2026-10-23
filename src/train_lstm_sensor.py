"""
LSTM sensor-only baseline. Uses past 4h of CGM + HR + steps to predict
whether CGM crosses 180 in the next 2h. No EHR input. No leaks.
This is Row 2 of the ablation: sensor-only.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

from labeling import label_spike

WINDOW = 48        # past 48 steps * 5 min = 4 hours
HORIZON = 24       # next 24 steps * 5 min = 2 hours
FEATURES = ["cgm_mgdl", "heart_rate", "steps"]
BATCH = 128
EPOCHS = 15
LR = 1e-3


class SpikeDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


class LSTMSensor(nn.Module):
    def __init__(self, n_features=3, hidden=32, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, batch_first=True)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.head(h[-1]).squeeze(-1)


def build_windows(df):
    """Returns (X, y, patient_ids) arrays. One sample per (patient, end_timestamp)."""
    Xs, ys, pids = [], [], []
    for pid, g in df.groupby("patient_id"):
        g = g.sort_values("timestamp").reset_index(drop=True)
        feats = g[FEATURES].values.astype(np.float32)
        labels = g["label"].values.astype(np.float32)
        for end in range(WINDOW, len(g)):
            Xs.append(feats[end - WINDOW:end])
            ys.append(labels[end])
            pids.append(pid)
    return (np.array(Xs), np.array(ys), np.array(pids))


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    df = pd.read_parquet("data/wearable_ts/synthetic_cgm.parquet")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = label_spike(df)
    print(f"Loaded {len(df):,} rows, {df.patient_id.nunique()} patients")

    X, y, pids = build_windows(df)
    print(f"Windows: {len(X):,}  positive rate: {y.mean():.2%}")

    # Per-patient split (avoid leakage between train/test)
    unique_pids = np.unique(pids)
    tr_pids, te_pids = train_test_split(unique_pids, test_size=0.3, random_state=42)
    tr_mask = np.isin(pids, tr_pids)
    te_mask = np.isin(pids, te_pids)

    # Normalize using training stats
    mu = X[tr_mask].mean(axis=(0, 1), keepdims=True)
    sd = X[tr_mask].std(axis=(0, 1), keepdims=True) + 1e-6
    Xn = (X - mu) / sd

    tr_ds = SpikeDataset(torch.tensor(Xn[tr_mask], dtype=torch.float32),
                        torch.tensor(y[tr_mask], dtype=torch.float32))
    te_ds = SpikeDataset(torch.tensor(Xn[te_mask], dtype=torch.float32),
                        torch.tensor(y[te_mask], dtype=torch.float32))
    tr_dl = DataLoader(tr_ds, batch_size=BATCH, shuffle=True)
    te_dl = DataLoader(te_ds, batch_size=BATCH, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LSTMSensor().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(EPOCHS):
        model.train()
        losses = []
        for xb, yb in tr_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        print(f"Epoch {epoch+1:02d}/{EPOCHS}  loss {np.mean(losses):.4f}")

    model.eval()
    probs, ys = [], []
    with torch.no_grad():
        for xb, yb in te_dl:
            logits = model(xb.to(device))
            probs.append(torch.sigmoid(logits).cpu().numpy())
            ys.append(yb.numpy())
    probs = np.concatenate(probs)
    ys = np.concatenate(ys)

    pred = (probs > 0.5).astype(int)
    print("\n=== LSTM Sensor-Only (Row 2 of ablation) ===")
    print(f"AUC        : {roc_auc_score(ys, probs):.3f}")
    print(f"Precision  : {precision_score(ys, pred, zero_division=0):.3f}")
    print(f"Recall     : {recall_score(ys, pred, zero_division=0):.3f}")
    print(f"F1         : {f1_score(ys, pred, zero_division=0):.3f}")

    torch.save(model.state_dict(), "models/lstm_sensor.pt")
    print("\nSaved model -> models/lstm_sensor.pt")


if __name__ == "__main__":
    main()
