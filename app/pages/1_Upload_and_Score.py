# app/pages/1_Upload_and_Score.py
# ─────────────────────────────────────────────────────────────
# Primary user flow:
#   ① Upload contacts → ② Describe event → ③ Run agent → ④ Download results
#
# Dual mode:
#   LLM Agent mode  : Claude parses free text, orchestrates tools, narrates results.
#   Pipeline mode   : Pure Python, no API key required, identical output.
# ─────────────────────────────────────────────────────────────

import sys, os, json, tempfile, pickle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from config import TOP_N, BUFFER_PERCENT, USE_LLM_EXPLANATIONS, MODEL_PATH, TOP_N_EXPLANATIONS

st.set_page_config(page_title="Upload & Score", page_icon="📊", layout="wide")
st.title("📊 Upload & Score")
st.markdown("Score your entire contact database against a new event.")

# ── Mode detection ────────────────────────────────────────────
has_key = bool(
    os.getenv("ANTHROPIC_API_KEY") or
    (hasattr(st, "secrets") and st.secrets.get("ANTHROPIC_API_KEY", ""))
)
model_ok = Path(MODEL_PATH).exists()

c1, c2 = st.columns(2)
c1.info(f"**Mode:** {'🤖 LLM Agent (Claude)' if has_key else '⚙️ Pipeline only (no API key)'}")
if model_ok:
    c2.success("**Model:** Trained model found ✓")
else:
    c2.error("**Model:** Not found — go to ② Train Model first")

st.divider()

# ── Session state ─────────────────────────────────────────────
for k, default in [("contacts_path", None), ("scored_path", None),
                   ("output_path", None), ("explanations", []),
                   ("parsed_event", {}), ("event", {})]:
    if k not in st.session_state:
        st.session_state[k] = default

# ═══════════════════════════════════════════════════════════
# STEP 1 — Upload contacts
# ═══════════════════════════════════════════════════════════
st.subheader("① Upload your contact database")
uploaded = st.file_uploader(
    "Contacts CSV or Excel",
    type=["csv", "xlsx"],
    help=(
        "Required columns (rename in config.py if needed): "
        "contact_id, name, email, job_title, industry, location, "
        "last_booking_date, booking_count_12m, avg_booking_price"
    ),
)
if uploaded:
    sfx = ".csv" if uploaded.name.endswith(".csv") else ".xlsx"
    with tempfile.NamedTemporaryFile(delete=False, suffix=sfx) as tmp:
        tmp.write(uploaded.read())
        st.session_state.contacts_path = tmp.name
    try:
        prev = pd.read_csv(st.session_state.contacts_path) if sfx == ".csv" \
               else pd.read_excel(st.session_state.contacts_path)
        m1, m2, m3 = st.columns(3)
        m1.metric("Contacts", f"{len(prev):,}")
        m2.metric("Columns",  len(prev.columns))
        m3.metric("File size", f"{uploaded.size / 1024:.0f} KB")
        with st.expander("Preview — first 5 rows"):
            st.dataframe(prev.head(5), use_container_width=True)
    except Exception as e:
        st.error(f"Could not read file: {e}")
        st.session_state.contacts_path = None

st.divider()

# ═══════════════════════════════════════════════════════════
# STEP 2 — Describe the event
# ═══════════════════════════════════════════════════════════
st.subheader("② Describe the new event")
tab_text, tab_form = st.tabs(["Free text  (AI parses)", "Structured form"])

with tab_text:
    st.caption(
        "Type a plain-language description. "
        "The AI agent extracts topic, format, price, location, and audience automatically."
    )
    event_text = st.text_area(
        "Event description",
        placeholder=(
            "e.g. Two-day HR management seminar in Frankfurt, 890 EUR, "
            "September 2025, aimed at HR managers and team leads in mid-sized companies."
        ),
        height=90, label_visibility="collapsed",
    )
    if st.button("🔍 Parse event description", disabled=not (event_text and has_key)):
        with st.spinner("Parsing with Claude..."):
            from agent.llm_client import parse_event_from_text
            parsed = parse_event_from_text(event_text)
        if parsed:
            st.session_state.parsed_event = parsed
            st.session_state.event = parsed
            st.success("Parsed successfully.")
            st.json(parsed)
        else:
            st.warning("Could not parse. Please use the structured form.")
    if not has_key:
        st.caption("⚠️ Free-text parsing requires an Anthropic API key. Use the structured form.")
    elif st.session_state.parsed_event:
        st.session_state.event = st.session_state.parsed_event

with tab_form:
    fc1, fc2 = st.columns(2)
    with fc1:
        topic    = st.text_input("Topic *", placeholder="HR Management")
        fmt      = st.selectbox("Format *", ["Seminar","Congress","E-Learning","Workshop","Other"])
        price    = st.number_input("Price (€) *", min_value=0, value=890, step=10)
        location = st.text_input("Location", placeholder="Frankfurt")
    with fc2:
        audience_raw = st.text_input("Target audience (comma-separated)",
                                     placeholder="HR Manager, Team Lead, L&D Manager")
        date     = st.date_input("Event date")
        duration = st.number_input("Duration (days)", min_value=1, value=2)
        online   = st.checkbox("Online event")
    if topic:
        st.session_state.event = {
            "topic": topic, "format": fmt, "price": float(price),
            "location": "" if online else location,
            "audience": [a.strip() for a in audience_raw.split(",") if a.strip()],
            "date": str(date), "duration_days": int(duration), "online": online,
        }

st.divider()

# ═══════════════════════════════════════════════════════════
# STEP 3 — Run
# ═══════════════════════════════════════════════════════════
st.subheader("③ Run the scoring agent")

cc1, cc2 = st.columns([1, 2])
with cc1:
    top_n = st.number_input("Contacts to return", min_value=100, value=TOP_N, step=500,
                             help=f"A {BUFFER_PERCENT}% safety buffer is added automatically.")
    want_explain = st.checkbox(
        f"Generate SHAP explanations (top {TOP_N_EXPLANATIONS})",
        value=False, disabled=not model_ok,
        help="Shows which SCM-grounded features drove each contact's score.",
    )

missing = []
if not st.session_state.contacts_path:  missing.append("contact file")
if not st.session_state.event.get("topic"): missing.append("event topic")
if not model_ok:                         missing.append("trained model")
if missing:
    st.info(f"Still needed before running: **{', '.join(missing)}**.")

if st.button("🚀 Run Agent", disabled=bool(missing), type="primary"):
    prog = st.progress(0, text="Initialising…")
    log_ph = st.empty()
    logs: list[str] = []

    def _log(msg: str):
        logs.append(msg)
        log_ph.caption("  ·  ".join(logs[-5:]))

    try:
        if has_key:
            # ── LLM Agent path ────────────────────────────────
            from agent.llm_client import run_agent_loop
            prog.progress(10, "Sending to Claude…")
            desc = (
                f"Score contacts for this event and return the top {top_n} contacts. "
                f"Event: {json.dumps(st.session_state.event)}. "
                + (f"Also generate SHAP explanations for the top {TOP_N_EXPLANATIONS}."
                   if want_explain else "")
            )
            result = run_agent_loop(
                user_message=desc,
                contacts_path=st.session_state.contacts_path,
                model_path=MODEL_PATH,
                stream_callback=_log,
            )
            if result["status"] != "ok":
                st.error(f"Agent error: {result.get('error')}")
                prog.empty(); st.stop()
            st.session_state.output_path   = result["output_path"]
            st.session_state.explanations  = result["explanations"]

        else:
            # ── Pure pipeline path ────────────────────────────
            from pipeline.data_loader import load_contacts
            from pipeline.scorer import score_contacts
            from pipeline.explainer import explain_batch

            prog.progress(15, "Loading contacts…")
            _log("Loading contacts…")
            df = load_contacts(st.session_state.contacts_path)
            _log(f"Loaded {len(df):,} contacts")

            prog.progress(40, "Running scoring model…")
            _log("Running scoring model…")
            scored = score_contacts(df, st.session_state.event, MODEL_PATH)
            _log(f"Scored {len(scored):,} contacts | top score: {scored['score'].iloc[0]:.3f}")

            prog.progress(75, "Saving output…")
            Path("outputs").mkdir(exist_ok=True)
            label = (st.session_state.event.get("topic") or "event").replace(" ", "_")[:30]
            buffer_n = int(top_n * (1 + BUFFER_PERCENT / 100))
            out_path = f"outputs/top_{top_n}_{label}.csv"
            scored.head(buffer_n).to_csv(out_path, index=False)
            st.session_state.output_path = out_path
            _log(f"Saved {buffer_n:,} contacts (top {top_n:,} + {BUFFER_PERCENT}% buffer) → {out_path}")

            st.session_state.explanations = []
            if want_explain:
                prog.progress(85, "Generating SHAP explanations…")
                _log("Generating SHAP explanations…")
                model = pickle.load(open(MODEL_PATH, "rb"))
                st.session_state.explanations = explain_batch(scored, st.session_state.event,
                                                               model, n=TOP_N_EXPLANATIONS)

        prog.progress(100, "Complete ✓")
        log_ph.empty()
        st.success(f"✓ Scoring complete — top {top_n:,} contacts ready for download.")

    except Exception as e:
        st.error(f"Error: {e}")
        prog.empty()

st.divider()

# ═══════════════════════════════════════════════════════════
# STEP 4 — Results
# ═══════════════════════════════════════════════════════════
if st.session_state.output_path and Path(st.session_state.output_path).exists():
    st.subheader("④ Results")
    df = pd.read_csv(st.session_state.output_path)

    # Metrics row
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Contacts returned", f"{len(df):,}")
    r2.metric("Top score",  f"{df['score'].max():.3f}" if "score" in df.columns else "—")
    r3.metric("Avg score",  f"{df['score'].mean():.3f}" if "score" in df.columns else "—")
    r4.metric("Score > 0.7", f"{(df['score'] > 0.7).sum():,}" if "score" in df.columns else "—")

    # ABC distribution chart
    if "abc_tier" in df.columns:
        abc_counts = df["abc_tier"].value_counts().reindex(["A","B","C"]).fillna(0)
        fig_abc = go.Figure(go.Bar(
            x=abc_counts.index, y=abc_counts.values,
            marker_color=["#2E75B6", "#ED7D31", "#A5A5A5"],
            text=abc_counts.values.astype(int), textposition="outside",
        ))
        fig_abc.update_layout(
            title="ABC Tier Distribution (analogous to inventory ABC classification)",
            xaxis_title="ABC Tier", yaxis_title="Number of contacts",
            height=280, margin=dict(l=40, r=20, t=50, b=40),
        )
        st.plotly_chart(fig_abc, use_container_width=True)

    # Score distribution
    if "score" in df.columns:
        fig_hist = go.Figure(go.Histogram(
            x=df["score"], nbinsx=40,
            marker_color="#2E75B6", opacity=0.8,
        ))
        fig_hist.update_layout(
            title="Score Distribution",
            xaxis_title="Booking Probability Score", yaxis_title="Count",
            height=240, margin=dict(l=40, r=20, t=50, b=40),
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    # Preview table
    with st.expander("Preview — top 20 contacts"):
        disp = [c for c in ["name","email","job_title","industry","location","score","abc_tier","rank"]
                if c in df.columns]
        st.dataframe(
            df[disp].head(20).style.background_gradient(
                subset=["score"] if "score" in disp else [], cmap="Blues"
            ),
            use_container_width=True,
        )

    # Download
    with open(st.session_state.output_path, "rb") as f:
        label = (st.session_state.event.get("topic") or "event").replace(" ", "_")[:20]
        st.download_button(
            "⬇️ Download scored list (CSV)",
            data=f,
            file_name=f"top_contacts_{label}.csv",
            mime="text/csv",
            type="primary",
        )

    # SHAP explanations
    if st.session_state.explanations:
        st.divider()
        st.subheader(f"SHAP Score Explanations — top {len(st.session_state.explanations)} contacts")
        st.caption("Each bar shows how much each SCM-grounded feature pushed the score up (↑) or down (↓).")
        for item in st.session_state.explanations:
            lbl = (f"**{item.get('name', item.get('contact_id', '—'))}**"
                   f" | Score: {item.get('score', 0):.3f}"
                   f" | ABC: {item.get('abc_tier', '—')}")
            with st.expander(lbl):
                st.markdown(f"_{item['explanation']}_")
                sv = item.get("shap_values", {})
                if sv:
                    sorted_sv = sorted(sv.items(), key=lambda x: x[1], reverse=True)
                    fig = go.Figure(go.Bar(
                        x=[v for _, v in sorted_sv],
                        y=[k.replace("_", " ") for k, _ in sorted_sv],
                        orientation="h",
                        marker_color=["#2E75B6" if v >= 0 else "#C00000" for _, v in sorted_sv],
                    ))
                    fig.update_layout(
                        title="SHAP values — feature contributions",
                        xaxis_title="Impact on booking probability score",
                        height=230, margin=dict(l=10, r=10, t=35, b=20),
                    )
                    st.plotly_chart(fig, use_container_width=True)
