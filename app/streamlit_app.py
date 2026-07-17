# app/streamlit_app.py
# ─────────────────────────────────────────────────────────────
# Main Streamlit entry point.
# Run: streamlit run app/streamlit_app.py
# Deploy free: share.streamlit.io → point to this file.
# ─────────────────────────────────────────────────────────────

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

st.set_page_config(
    page_title="SCM Contact Scoring Agent",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📦 SCM Scoring Agent")
    st.markdown(
        "AI-driven demand forecasting for continuing education marketing. "
        "Identifies your highest-probability attendees for any new event."
    )
    st.divider()
    st.markdown("**Navigation**")
    st.page_link("pages/1_Upload_and_Score.py", label="① Upload & Score",   icon="📊")
    st.page_link("pages/2_Train_Model.py",       label="② Train Model",     icon="🧠")
    st.page_link("pages/3_Evaluate_Model.py",    label="③ Evaluate Model",  icon="📈")
    st.divider()
    st.caption(
        "Thesis: *AI-Driven Demand Forecasting and Predictive Contact Scoring "
        "in Continuing Education Supply Chains*\n\n"
        "Student: Afaan Irfan Siddiquee\n"
        "Supervisor: Dr. Dany Djeudeu\n"
        "SRH Hochschule — MSc SCM 2024–2026"
    )

# ── Home page ─────────────────────────────────────────────────
st.title("📦 SCM Predictive Contact Scoring Agent")
st.markdown(
    "**Supply chain thinking applied to marketing logistics.** "
    "Score your entire contact database against any new event in seconds — "
    "ranked by individual booking probability, segmented by ABC tier, explained by SHAP."
)

st.divider()

col1, col2, col3 = st.columns(3)
with col1:
    st.info(
        "**① Train**\n\n"
        "Upload historical booking data. "
        "Train XGBoost or LightGBM once. "
        "Takes ~30 seconds."
    )
with col2:
    st.success(
        "**② Score**\n\n"
        "Describe a new event in plain language or fill the form. "
        "The agent scores all contacts and assigns ABC tiers."
    )
with col3:
    st.warning(
        "**③ Evaluate**\n\n"
        "Review lift curves, precision@K, ROI tables, "
        "and ABC-XYZ segment distributions."
    )

st.divider()
st.markdown("### SCM Framework at a Glance")
st.markdown(
    "This system translates established supply chain management concepts "
    "directly into ML features and system parameters:"
)

data = {
    "SCM Concept": [
        "Demand Signal Recency",
        "Demand Velocity (ABC-X)",
        "Value Tier Match (ABC)",
        "Product Family Affinity",
        "Market Segment Targeting",
        "Regional Distribution",
        "Safety Stock Buffer",
        "Fill Rate",
    ],
    "Contact Scoring Implementation": [
        "Days since last booking → recency feature",
        "Booking count (12 months) → frequency feature",
        "Normalised price distance → price alignment feature",
        "Topic overlap: event topic ∩ contact industry",
        "Industry match: contact profile ∈ event audience",
        "Location match: contact city = event city",
        "BUFFER_PERCENT: return TOP_N × 1.10 contacts",
        "Recall@K: % of actual bookers in top-N list",
    ],
}
import pandas as pd
st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)

with st.expander("What data do you need?"):
    st.markdown(
        """
**contacts.csv** — one row per contact:
`contact_id, name, email, job_title, industry, location, last_booking_date, booking_count_12m, avg_booking_price`

**historical_bookings.csv** — one row per contact–event pair:
`contact_id, event_topic, event_format, event_price, booked` (1=attended, 0=did not)

Column names are remapped in `config.py` — no code changes needed for a new dataset.

**Example files** are provided in the `data/` folder.
"""
    )

with st.expander("How to deploy on Streamlit Community Cloud (free)"):
    st.markdown(
        """
1. Push this repository to a **public GitHub repo**.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in.
3. Click **New app** → select your repo → set main file to `app/streamlit_app.py`.
4. Under **Advanced settings → Secrets**, add:
   ```toml
   ANTHROPIC_API_KEY = "sk-ant-..."
   ```
5. Click **Deploy**. Live in under 2 minutes.

The Anthropic API key is only needed for free-text event parsing and SHAP narration.
The scoring pipeline runs without it.
"""
    )
