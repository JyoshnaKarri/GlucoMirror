# GlucoMirror - Architecture

## Overview
Static EHR (7d) + Wearable time-series (3 features x 48 steps)
-> EHR MLP (7->16->8) + LSTM (32 hidden)
-> Fusion MLP [8d | 32d] -> sigmoid -> P(spike in 120 min)

## Ablation
| Model | AUC |
|---|---|
| EHR-only | 0.931 |
| Sensor-only | 0.938 |
| Fusion | 0.938 |

## Deployment
- Local: streamlit run dashboard/app.py
- Docker: docker compose up -> http://localhost:8501

## Data
Synthetic, severity-correlated. No real PHI. DPDP/HIPAA-safe.