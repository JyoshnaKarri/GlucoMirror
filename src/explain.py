"""
Standalone explainability: occlusion-based per-prediction drivers.
Reads fusion model + produces human-readable explanation.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from model_infer import get_twin, predict


def explain(pid, timestamp):
    out = predict(pid, timestamp)
    if out["risk"] is None:
        return {"error": "insufficient history"}
    return {
        "patient": pid,
        "timestamp": str(timestamp),
        "risk": out["risk"],
        "drivers": out["drivers"],
        "narrative": (
            f"Risk {out['risk']*100:.0f}%. "
            f"Top driver: {out['drivers'][0][0] if out['drivers'] else 'n/a'} "
            f"({out['drivers'][0][1]:+.2f}). "
            + ("Above threshold ? recommend intervention." if out["risk"] > 0.7
               else "Below threshold ? continue monitoring.")
        ),
    }


if __name__ == "__main__":
    twin = get_twin()
    pid = twin.data.patient_id.iloc[500]
    ts = twin.data[twin.data.patient_id == pid].timestamp.iloc[-1]
    r = explain(pid, ts)
    print(f"Patient: {r['patient']} @ {r['timestamp']}")
    print(f"Risk: {r['risk']:.3f}")
    print(r["narrative"])
    for name, val in r["drivers"]:
        print(f"  {name:20s} {val:+.3f}")
