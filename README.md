# AI-Driven Demand Forecasting and Predictive Contact Scoring
### MSc Supply Chain Management Thesis Project
**Student:** Afaan Irfan Siddiquee | **Supervisor:** Dr. Dany Djeudeu | **SRH Hochschule 2024–2026**

---

## What this system does

A continuing education provider has 100,000 contacts and runs dozens of events per year.
For each new event, this system automatically identifies the highest-probability attendees —
ranked by ML-predicted booking probability, segmented by ABC tier, explainable by SHAP.

**Supply chain framing:** Audience targeting is treated as a demand forecasting and
inventory prioritisation problem. Every feature, parameter, and metric maps directly
onto an established SCM concept (see Thesis Chapter 3).

---

## Quickstart — 3 steps

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run locally
```bash
streamlit run app/streamlit_app.py
```

### 3. Train → Score → Evaluate
- Go to **② Train Model** → upload `data/synthetic_contacts.csv` + `data/synthetic_bookings.csv`
- Go to **① Upload & Score** → upload contacts, describe an event, click Run
- Go to **③ Evaluate Model** → view lift curve, precision@K, ROI table

---

## Deploy free on Streamlit Community Cloud

1. Push this repo to a **public GitHub repository**
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app
3. Select your repo, set main file: `app/streamlit_app.py`
4. Under **Advanced → Secrets**, add:
   ```toml
   ANTHROPIC_API_KEY = "sk-ant-..."
   ```
5. Click **Deploy** — live in under 2 minutes

The API key is optional. Without it, the system runs in pipeline-only mode
(structured form input, identical scoring output).

---

## Your data format

### contacts.csv
| Column | Description |
|---|---|
| contact_id | Unique identifier |
| name | Full name |
| email | Email address |
| job_title | Job title |
| industry | Industry / sector |
| location | City or region |
| last_booking_date | Date of last event booking (YYYY-MM-DD) |
| booking_count_12m | Number of bookings in last 12 months |
| avg_booking_price | Average price of past bookings (EUR) |

### historical_bookings.csv
| Column | Description |
|---|---|
| contact_id | Must match contacts.csv |
| event_topic | Topic of the past event |
| event_format | Seminar / Congress / E-Learning / Workshop |
| event_price | Registration price (EUR) |
| booked | 1 = attended, 0 = did not attend |

**Column names are remapped in `config.py`** — no code changes needed for a different dataset.

---

## Project structure

```
code/
├── config.py                    ← Edit this for your dataset
├── requirements.txt
├── .env.example                 → Copy to .env, add API key
├── README.md
│
├── data/
│   ├── contacts_example.csv     10 labelled example contacts
│   ├── synthetic_contacts.csv   500 synthetic contacts for testing
│   ├── synthetic_bookings.csv   2,500 synthetic booking records
│   └── events_example.json      3 example event definitions
│
├── agent/
│   ├── tools.py                 4 Python tool functions
│   ├── tool_definitions.py      JSON schemas for Claude API
│   └── llm_client.py            Tool-calling loop + audit logging
│
├── pipeline/
│   ├── data_loader.py           CSV / Excel / SQLite → DataFrame
│   ├── feature_engineer.py      7 SCM-grounded features
│   ├── scorer.py                ML scoring + ABC tier tagging
│   └── explainer.py             SHAP explanations
│
├── model/
│   ├── trainer.py               Train XGBoost or LightGBM
│   └── evaluator.py             AUC, lift, precision@K, ROI
│
├── notebooks/                   ← Your analysis notebooks go here
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_evaluation.ipynb
│
└── app/
    ├── streamlit_app.py         Home page + navigation
    └── pages/
        ├── 1_Upload_and_Score.py
        ├── 2_Train_Model.py
        └── 3_Evaluate_Model.py
```

---

## SCM Feature Framework

| Feature | SCM Concept | Implementation |
|---|---|---|
| `days_since_last_booking` | Demand signal recency | Days elapsed since last booking |
| `booking_count_12m` | Demand velocity (ABC-X) | Bookings in last 12 months |
| `price_distance_norm` | Value tier match (ABC) | Normalised price distance |
| `topic_overlap` | Product family affinity | Topic keyword ∩ contact profile |
| `format_familiarity` | Channel preference | Booking frequency proxy |
| `location_match` | Regional distribution | Contact city = event city |
| `industry_match` | Market segment targeting | Industry ∈ event audience |

---

## Key configuration parameters

| Parameter | Default | Description |
|---|---|---|
| `TOP_N` | 10,000 | Contacts returned per event |
| `BUFFER_PERCENT` | 10 | Safety buffer above cut-off (= safety stock) |
| `MODEL_TYPE` | xgboost | "xgboost" or "lightgbm" |
| `COST_PER_CONTACT_EUR` | 2.0 | For ROI calculation |
| `USE_LLM_EXPLANATIONS` | True | False = no API key needed |

---

## Thesis milestones

| Weeks | Deliverable |
|---|---|
| 1–2 | Literature review + SCM framework answer |
| 3–4 | Data specification document |
| 5–6 | Feature engineering notebook |
| 7–8 | Model training + baseline comparison |
| 9–10 | Agent test report (10 cases) |
| 11–12 | Live Streamlit app |
| 13–14 | Full evaluation (Track A + B + C) |
| 15–17 | Thesis draft |
| 18–20 | Revision + submission |
