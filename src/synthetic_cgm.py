"""
Synthetic CGM generator for GlucoMirror Digital Twin.
Heterogeneous patients: each has a clinical severity that modulates meal response.
Severity is SAVED in the output so downstream EHR generation can correlate.
"""
import argparse, os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def bergman_glucose(duration_min=1440, dt=5, seed=None, severity=1.0):
    rng = np.random.default_rng(seed)
    n = int(duration_min / dt)
    t = np.arange(n) * dt
    G = np.zeros(n)
    G[0] = rng.uniform(80, 110)
    meals_min = rng.choice([420, 780, 1140], size=3, replace=False)
    meal_carbs = rng.uniform(20, 55, size=3)
    for i in range(1, n):
        dG = 0.0
        for m, c in zip(meals_min, meal_carbs):
            if 0 <= t[i] - m <= 120:
                dG += c * 0.55 * severity * np.exp(-(t[i] - m) / 45)
        dG -= 0.06 * (G[i - 1] - 90)
        G[i] = np.clip(G[i - 1] + dG * dt / 5 + rng.normal(0, 1.5), 60, 320)
    return G


def build_patient_row(pid, day_offset, G, severity=1.0):
    ts = datetime(2026, 1, 1) + timedelta(days=day_offset)
    rows = []
    for i, g in enumerate(G):
        rows.append({
            "patient_id": pid,
            "timestamp": ts + timedelta(minutes=5 * i),
            "cgm_mgdl": round(float(g), 1),
            "heart_rate": int(np.clip(np.random.normal(72, 8), 50, 130)),
            "steps": int(np.random.poisson(60)),
            "sleep_stage": int(np.random.choice([0, 1, 2, 3, 4]))
                if (ts + timedelta(minutes=5 * i)).hour < 7 else 5,
            "severity": round(float(severity), 3),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patients", type=int, default=200)
    ap.add_argument("--days", type=int, default=3)
    ap.add_argument("--out", type=str, default="data/wearable_ts/")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    all_rows = []
    for pid in range(args.patients):
        pid_str = f"P{pid:04d}"
        severity = np.random.default_rng(pid).uniform(0.4, 1.6)
        for d in range(args.days):
            G = bergman_glucose(seed=pid * 1000 + d, severity=severity)
            all_rows.extend(build_patient_row(pid_str, d, G, severity=severity))

    df = pd.DataFrame(all_rows)
    out_path = os.path.join(args.out, "synthetic_cgm.parquet")
    df.to_parquet(out_path, index=False)
    print(f"Wrote {len(df):,} rows for {args.patients} patients x {args.days} days -> {out_path}")
    print(f"Spike rate (>180 mg/dL): {(df.cgm_mgdl > 180).mean():.2%}")
    print(f"Severity range: {df.severity.min():.2f} - {df.severity.max():.2f}")


if __name__ == "__main__":
    main()
