import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from model_infer import get_twin, predict

st.set_page_config(page_title="GlucoMirror", layout="wide", page_icon="🩺")

@st.cache_resource
def load_twin():
    return get_twin()

twin = load_twin()

st.title("🩺 GlucoMirror — Digital Twin for T2D Glucose Spike Prediction")
st.caption("Predicts >180 mg/dL event up to 2h ahead. Synthetic data. No PHI.")

with st.sidebar:
    st.header("Patient Selection")
    all_pids = sorted(twin.data.patient_id.unique())
    pid = st.selectbox("Patient", all_pids, index=0)
    patient_rows = twin.data[twin.data.patient_id == pid].sort_values("timestamp").reset_index(drop=True)
    idx = st.slider("Time index", 48, len(patient_rows) - 1, len(patient_rows) - 1, step=1)
    ts = patient_rows.timestamp.iloc[idx]
    st.caption(f"Time: {ts}")

out = predict(pid, ts)
risk = out["risk"]

c1, c2, c3 = st.columns(3)

with c1:
    st.subheader("Risk")
    if risk is None:
        st.warning("Insufficient history.")
    else:
        color = "#2ca02c" if risk < 0.3 else "#ff7f0e" if risk < 0.7 else "#d62728"
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=risk * 100,
            number={"suffix": "%"}, title={"text": "P(spike in 2h)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, 30], "color": "#e8f5e9"},
                    {"range": [30, 70], "color": "#fff3e0"},
                    {"range": [70, 100], "color": "#ffebee"},
                ],
            },
        ))
        fig.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, width='stretch')

with c2:
    st.subheader("EHR")
    ehr = twin.ehr.loc[pid]
    st.metric("HbA1c", f"{ehr['hba1c']}%")
    st.metric("BMI", f"{ehr['bmi']}")
    st.metric("Age", int(ehr['age']))

with c3:
    st.subheader("Drivers")
    for name, val in out["drivers"][:5]:
        arrow = "🔺" if val > 0 else "🔻"
        st.write(f"{arrow} **{name}** `{val:+.3f}`")

st.subheader("Recent CGM (24h)")
hist = patient_rows[(patient_rows.timestamp >= ts - pd.Timedelta(hours=24)) &
                    (patient_rows.timestamp <= ts)]
if len(hist):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist.timestamp, y=hist.cgm_mgdl, mode="lines",
                             name="CGM", line=dict(color="#1f77b4", width=2)))
    fig.add_hline(y=180, line_dash="dash", line_color="red", annotation_text="Spike 180")
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=20, b=20),
                      yaxis_title="mg/dL")
    st.plotly_chart(fig, width='stretch')

st.subheader("🔮 What-If: Walk")
walk = st.slider("Walk (min)", 0, 60, 0, step=5)
if risk is not None and walk > 0:
    red = min(walk * 0.015, 0.40)
    proj = max(risk - red, 0.0)
    st.metric(f"Walk {walk} min", f"{proj*100:.1f}%", delta=f"-{(risk-proj)*100:.1f} pp")

st.divider()
st.caption("⚠️ Prototype. Synthetic data. Not for clinical use.")