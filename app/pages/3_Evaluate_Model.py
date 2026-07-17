# app/pages/3_Evaluate_Model.py
# Full evaluation suite: lift curve, ROC, precision@K, ROI table, ABC distribution.
# All metrics presented using SCM terminology (fill rate, cost per booking, ABC tiers).

import sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import pandas as pd

from config import MODEL_PATH, COST_PER_CONTACT_EUR

st.set_page_config(page_title="Evaluate Model", page_icon="📈", layout="wide")
st.title("📈 Evaluate Model Performance")
st.markdown(
    "Understand how much better the ML model performs compared to "
    "SCM-inspired baselines (RFM heuristic, recency-only rule). "
    "All metrics are expressed in supply chain–familiar terms."
)

if not Path(MODEL_PATH).exists():
    st.error(f"No trained model at `{MODEL_PATH}`. Go to **② Train Model** first.")
    st.stop()

st.divider()

# ── Upload data ───────────────────────────────────────────────
ec1, ec2 = st.columns(2)
with ec1:
    contacts_f = st.file_uploader("Contacts CSV or Excel", type=["csv","xlsx"], key="ec")
with ec2:
    bookings_f = st.file_uploader("Historical bookings CSV or Excel", type=["csv","xlsx"], key="eb")

# ── Reference event for features ─────────────────────────────
st.subheader("Reference event (used to build evaluation feature matrix)")
f1, f2, f3 = st.columns(3)
with f1:
    topic    = st.text_input("Topic",    "HR Management")
    fmt      = st.selectbox("Format",   ["Seminar","Congress","E-Learning","Workshop","Other"])
with f2:
    price    = st.number_input("Price (€)", value=890)
    location = st.text_input("Location", "Frankfurt")
with f3:
    audience = st.text_input("Audience", "HR Manager, Team Lead, L&D Manager")
    online   = st.checkbox("Online event")

event = {
    "topic": topic, "format": fmt, "price": float(price),
    "location": "" if online else location,
    "audience": [a.strip() for a in audience.split(",") if a.strip()],
    "online": online,
}

st.divider()

can_eval = contacts_f is not None and bookings_f is not None
if not can_eval:
    st.info("Upload both files to enable evaluation.")

if st.button("📈 Run Evaluation", disabled=not can_eval, type="primary"):
    def _save(f, sfx):
        with tempfile.NamedTemporaryFile(delete=False, suffix=sfx) as t:
            t.write(f.read()); return t.name

    cp = _save(contacts_f, ".csv" if contacts_f.name.endswith(".csv") else ".xlsx")
    bp = _save(bookings_f, ".csv" if bookings_f.name.endswith(".csv") else ".xlsx")

    with st.spinner("Evaluating…"):
        try:
            from model.evaluator import evaluate
            res = evaluate(MODEL_PATH, cp, bp, event)

            # ── AUC summary ───────────────────────────────────
            st.success(f"✓ Evaluation complete — AUC = **{res['auc']:.4f}**")
            st.divider()

            # ── Lift curve ────────────────────────────────────
            st.subheader("Lift Curve — ML Model vs. SCM-Inspired Baselines")
            st.plotly_chart(res["lift_fig"], use_container_width=True)
            st.caption(
                "**How to read this:** The further the ML model line is above the "
                "baselines and the random line, the better. "
                "A lift of 3× at 10% means: by contacting only the top 10% of contacts "
                "(as ranked by the model), you reach 30% of all actual bookers — "
                "equivalent to a 100% fill-rate improvement over random selection at the same budget. "
                "This is the *demand forecasting accuracy* metric expressed in marketing terms."
            )
            st.divider()

            # ── Precision@K table ─────────────────────────────
            st.subheader("Precision @ K — How Many Bookers Are in the Top List?")
            st.dataframe(res["precision_at_k"], use_container_width=True, hide_index=True)
            st.caption(
                "**SCM interpretation:** Precision@K is the *fill rate* of the marketing campaign. "
                "If precision@10% = 15% and the random baseline is 5%, "
                "the model delivers a 3× fill-rate improvement at the same outreach cost."
            )
            st.divider()

            # ── ROI table ─────────────────────────────────────
            st.subheader(f"ROI Analysis — Cost Per Booking (at €{COST_PER_CONTACT_EUR:.0f}/contact)")
            st.dataframe(res["roi_table"], use_container_width=True, hide_index=True)
            st.caption(
                f"Outreach cost assumed at €{COST_PER_CONTACT_EUR:.2f} per contact "
                f"(configurable in config.py → COST_PER_CONTACT_EUR). "
                "Cost saving is the % reduction in cost-per-booking vs. random selection — "
                "the primary business case for the scoring system."
            )
            st.divider()

            # ── ROC curve ─────────────────────────────────────
            st.subheader("ROC Curve")
            st.plotly_chart(res["roc_fig"], use_container_width=True)
            st.divider()

            # ── ABC distribution ──────────────────────────────
            st.subheader("ABC Tier Distribution of Scored Contacts")
            abc = res["abc_dist"]
            ac1, ac2, ac3 = st.columns(3)
            for col, (tier, vals) in zip([ac1, ac2, ac3], abc.items()):
                col.metric(
                    tier,
                    f"{vals['count']:,} contacts",
                    f"Avg score: {vals['avg_score']:.3f}",
                )
            st.caption(
                "**A-tier** contacts (top 20% by score) are your highest-demand segment — "
                "equivalent to A-items in inventory: high value, reliable, priority outreach. "
                "**C-tier** contacts have low booking probability for this specific event and "
                "should be excluded from the campaign to reduce wasted outreach cost."
            )

        except Exception as e:
            st.error(f"Evaluation failed: {e}")
            import traceback
            with st.expander("Error detail"):
                st.code(traceback.format_exc())
