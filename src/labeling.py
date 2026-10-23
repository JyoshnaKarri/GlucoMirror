"""
Creates the label for spike prediction.
label(t) = 1 iff CGM crosses 180 mg/dL in (t, t + horizon_min]
"""
import pandas as pd

SPIKE_THRESHOLD = 180.0
HORIZON_MIN = 120


def label_spike(df: pd.DataFrame, horizon_min: int = HORIZON_MIN,
                threshold: float = SPIKE_THRESHOLD) -> pd.DataFrame:
    """
    df columns: patient_id, timestamp, cgm_mgdl
    Adds: label (0/1) - whether a spike occurs within horizon_min after this row.
    """
    df = df.sort_values(["patient_id", "timestamp"]).copy()
    df["label"] = 0

    def _label_group(g):
        t = g["timestamp"].values
        v = g["cgm_mgdl"].values
        labels = []
        for i in range(len(g)):
            end_t = t[i] + pd.Timedelta(minutes=horizon_min).to_timedelta64()
            mask = (t > t[i]) & (t <= end_t) & (v > threshold)
            labels.append(1 if mask.any() else 0)
        return g.assign(label=labels)

    df = df.groupby("patient_id", group_keys=True).apply(_label_group)
    df = df.reset_index(level=0)
    df = df.reset_index(drop=True)
    if "patient_id" not in df.columns and "level_0" in df.columns:
        df = df.rename(columns={"level_0": "patient_id"})

    pos = df["label"].mean()
    print(f"Label created. Positive rate (spike in next {horizon_min} min): {pos:.2%}")
    if pos < 0.02:
        print("WARNING: very few positives - lower threshold or widen horizon.")
    if pos > 0.40:
        print("WARNING: too many positives - tighten horizon.")
    return df