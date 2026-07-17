# app/pages/2_Train_Model.py
# Train the scoring model from historical booking data.
# Run once per new dataset. Re-run when new booking data is available.

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import tempfile
import streamlit as st
import plotly.graph_objects as go

from config import MODEL_TYPE, MODEL_PATH

st.set_page_config(page_title="Train Model", page_icon="🧠", layout="wide")
st.title("🧠 Train the Scoring Model")
st.markdown(
    "Upload your historical booking data to train a gradient-boosted ML model. "
    "**You only need to do this once** — or when you want to retrain with new data."
)

st.divider()

# ── Upload files ──────────────────────────────────────────────
c1, c2 = st.columns(2)
with c1:
    st.subheader("① Contacts file")
    contacts_file = st.file_uploader("Contacts CSV or Excel", type=["csv","xlsx"], key="tc")
with c2:
    st.subheader("② Historical bookings file")
    st.caption("Required columns: `contact_id, event_topic, event_format, event_price, booked` (1=attended, 0=did not)")
    bookings_file = st.file_uploader("Bookings CSV or Excel", type=["csv","xlsx"], key="tb")

st.divider()

# ── Settings ──────────────────────────────────────────────────
st.subheader("③ Model settings")
mc1, mc2, mc3 = st.columns(3)
with mc1:
    model_type = st.selectbox("Algorithm", ["xgboost","lightgbm"], index=0,
                               help="XGBoost and LightGBM perform comparably; XGBoost is slightly more interpretable.")
with mc2:
    test_size = st.slider("Test set fraction", 0.10, 0.40, 0.20, 0.05,
                           help="Fraction of booking records held out for AUC evaluation.")
with mc3:
    st.metric("Safety buffer", "10%", help="Configured in config.py — BUFFER_PERCENT")

# ── Train ─────────────────────────────────────────────────────
st.divider()
can_train = contacts_file is not None and bookings_file is not None
if not can_train:
    st.info("Upload both files above to enable training.")

if st.button("🧠 Train Model", disabled=not can_train, type="primary"):
    def _save(f, sfx):
        with tempfile.NamedTemporaryFile(delete=False, suffix=sfx) as t:
            t.write(f.read())
            return t.name

    cp = _save(contacts_file,  ".csv" if contacts_file.name.endswith(".csv") else ".xlsx")
    bp = _save(bookings_file,  ".csv" if bookings_file.name.endswith(".csv") else ".xlsx")

    with st.spinner("Training in progress…"):
        try:
            from model.trainer import train
            result = train(bp, cp, model_type=model_type,
                           model_path=MODEL_PATH, test_size=test_size)

            st.success(f"✓ Model trained — AUC = **{result['auc']:.4f}**")

            r1, r2, r3, r4 = st.columns(4)
            r1.metric("AUC-ROC",          f"{result['auc']:.4f}")
            r2.metric("Training samples", f"{result['n_train']:,}")
            r3.metric("Test samples",     f"{result['n_test']:,}")
            r4.metric("Positive rate",    f"{result['positive_rate']:.1%}",
                       help="Fraction of booking records where booked=1. Typically 5–15%.")

            # Feature importance chart
            import pickle
            model = pickle.load(open(MODEL_PATH, "rb"))
            if hasattr(model, "feature_importances_"):
                imp = model.feature_importances_
                names = result["feature_names"]
                fig = go.Figure(go.Bar(
                    x=imp, y=names, orientation="h",
                    marker_color="#2E75B6",
                ))
                fig.update_layout(
                    title="Feature Importance (gain)",
                    xaxis_title="Importance score",
                    yaxis=dict(autorange="reversed"),
                    height=320, margin=dict(l=20, r=20, t=50, b=40),
                )
                st.plotly_chart(fig, use_container_width=True)
                st.caption(
                    "Feature names map directly to the SCM analogy framework (Thesis Chapter 3). "
                    "High importance for `booking_count_12m` confirms demand velocity as the "
                    "primary booking predictor — consistent with ABC-X classification logic."
                )

            st.info(
                f"Model saved to `{result['model_path']}`. "
                "Go to **① Upload & Score** to score a new event."
            )

        except Exception as e:
            st.error(f"Training failed: {e}")

# ── AUC guide ─────────────────────────────────────────────────
with st.expander("How to interpret the AUC score"):
    import pandas as pd
    guide = pd.DataFrame({
        "AUC range": ["0.50 – 0.60", "0.60 – 0.70", "0.70 – 0.80", "0.80 – 0.90", "0.90+"],
        "Interpretation": [
            "No better than random — check data quality",
            "Weak signal — model is better than random but limited",
            "Good — meaningful improvement over RFM baseline",
            "Very good — strong predictive power",
            "Excellent — check for data leakage",
        ],
        "Expected cost saving vs random": [
            "0%", "15–30%", "40–60%", "60–75%", "> 75%"
        ],
    })
    st.dataframe(guide, use_container_width=True, hide_index=True)
    st.caption(
        "For continuing education event data, AUC 0.72–0.82 is realistic with 12+ months "
        "of booking history. If AUC < 0.65, investigate data quality: check for "
        "class imbalance, missing booking records, or insufficient history."
    )
