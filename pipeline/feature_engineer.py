# pipeline/feature_engineer.py
# SCM-grounded feature engineering (Thesis Chapter 3 & 5.2)
import pandas as pd
from config import FEATURES, RECENCY_MAX_DAYS

def build_features(df: pd.DataFrame, event: dict) -> pd.DataFrame:
    F = pd.DataFrame(index=df.index)
    if FEATURES.get("days_since_last_booking"):
        now = pd.Timestamp.now()
        dates = pd.to_datetime(df.get("last_booking_date"), errors="coerce")
        F["days_since_last_booking"] = (now - dates).dt.days.fillna(RECENCY_MAX_DAYS).clip(upper=RECENCY_MAX_DAYS*1.5)
    if FEATURES.get("booking_count_12m"):
        F["booking_count_12m"] = pd.to_numeric(df.get("booking_count_12m", 0), errors="coerce").fillna(0)
    if FEATURES.get("price_distance_norm"):
        ep = float(event.get("price", 0))
        avg = pd.to_numeric(df.get("avg_booking_price", 0), errors="coerce").fillna(0)
        F["price_distance_norm"] = ((avg - ep).abs() / ep).clip(upper=3.0) if ep > 0 else 0.0
    if FEATURES.get("topic_overlap"):
        kws = list({str(event.get("topic","")).lower()} | {str(a).lower() for a in event.get("audience",[])} - {""})
        def _m(row):
            t = " ".join([str(row.get("job_title","")), str(row.get("industry",""))]).lower()
            return int(any(k in t for k in kws))
        F["topic_overlap"] = df.apply(_m, axis=1) if kws else 0
    if FEATURES.get("format_familiarity"):
        c = pd.to_numeric(df.get("booking_count_12m", 0), errors="coerce").fillna(0)
        mx = c.max()
        F["format_familiarity"] = (c / mx).clip(0,1) if mx > 0 else 0.0
    if FEATURES.get("location_match"):
        if event.get("online", False):
            F["location_match"] = 0.5
        else:
            el = str(event.get("location","")).lower().strip()
            locs = df.get("location", pd.Series("", index=df.index)).astype(str).str.lower().str.strip()
            F["location_match"] = (locs == el).astype(int) if el else 0
    if FEATURES.get("industry_match"):
        aud = [str(a).lower().strip() for a in event.get("audience",[]) if a]
        if aud:
            inds = df.get("industry", pd.Series("", index=df.index)).astype(str).str.lower().str.strip()
            F["industry_match"] = inds.apply(lambda x: int(any(a in x or x in a for a in aud if x)))
        else:
            F["industry_match"] = 0
    return F.astype(float).fillna(0)

def get_feature_names():
    return [k for k, v in FEATURES.items() if v]

def feature_scm_labels():
    return {
        "days_since_last_booking": "demand signal recency",
        "booking_count_12m":       "booking velocity (12m)",
        "price_distance_norm":     "price tier alignment",
        "topic_overlap":           "topic/industry match",
        "format_familiarity":      "format familiarity",
        "location_match":          "geographic proximity",
        "industry_match":          "target segment match",
    }
