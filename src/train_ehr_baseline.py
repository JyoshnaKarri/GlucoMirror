"""
EHR-only baseline (Row 1). EHR now correlated with patient severity,
so hba1c/bmi/insulin carry real signal. Still leak-free (no CGM summaries).
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import xgboost as xgb

from labeling import label_spike

RNG = np.random.default_rng(42)


def synth_ehr_for(patient_severity: dict):
    """EHR where hba1c, bmi, insulin depend on patient severity."""
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


def main():
    cgm = pd.read_parquet("data/wearable_ts/synthetic_cgm.parquet")
    cgm["timestamp"] = pd.to_datetime(cgm["timestamp"])
    print(f"Loaded {len(cgm):,} CGM rows, {cgm.patient_id.nunique()} patients")

    labeled = label_spike(cgm)
    severity_map = labeled.groupby("patient_id")["severity"].first().to_dict()
    ehr = synth_ehr_for(severity_map)

    agg = labeled.groupby("patient_id").agg(
        mean_hr=("heart_rate", "mean"),
        std_hr=("heart_rate", "std"),
        mean_steps=("steps", "mean"),
        label=("label", "max"),
    ).join(ehr)

    print(f"\nPatient-level dataset: {agg.shape[0]} rows, {agg.shape[1]} cols")
    print(f"Positive rate: {agg.label.mean():.1%}")

    X = agg.drop(columns=["label"])
    y = agg["label"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    model = xgb.XGBClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        eval_metric="logloss", random_state=42,
    )
    model.fit(X_tr, y_tr)

    proba = model.predict_proba(X_te)[:, 1]
    pred = (proba > 0.5).astype(int)

    print("\n=== EHR-Only Baseline (Row 1) ===")
    print(f"AUC        : {roc_auc_score(y_te, proba):.3f}")
    print(f"Precision  : {precision_score(y_te, pred, zero_division=0):.3f}")
    print(f"Recall     : {recall_score(y_te, pred, zero_division=0):.3f}")
    print(f"F1         : {f1_score(y_te, pred, zero_division=0):.3f}")

    imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\nTop features:")
    for k, v in imp.head(5).items():
        print(f"  {k:20s} {v:.3f}")


if __name__ == "__main__":
    main()
