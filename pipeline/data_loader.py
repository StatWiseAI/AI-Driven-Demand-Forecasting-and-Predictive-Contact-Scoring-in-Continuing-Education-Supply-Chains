# pipeline/data_loader.py
from pathlib import Path
import pandas as pd
from config import CONTACT_COLUMNS, BOOKING_COLUMNS, RECENCY_MAX_DAYS

def load_contacts(source) -> pd.DataFrame:
    df = _read(source) if not isinstance(source, pd.DataFrame) else source.copy()
    df = _rename(df, CONTACT_COLUMNS)
    return _clean(df)

def load_bookings(source) -> pd.DataFrame:
    df = _read(source) if not isinstance(source, pd.DataFrame) else source.copy()
    df = _rename(df, BOOKING_COLUMNS)
    df["booked"] = pd.to_numeric(df.get("booked", 0), errors="coerce").fillna(0).astype(int)
    df["event_price"] = pd.to_numeric(df.get("event_price", 0), errors="coerce").fillna(0)
    return df

def _read(path) -> pd.DataFrame:
    p = Path(path); s = p.suffix.lower()
    if s == ".csv": return pd.read_csv(p)
    if s in (".xlsx", ".xls"): return pd.read_excel(p)
    if s == ".db":
        import sqlalchemy as sa
        with sa.create_engine(f"sqlite:///{p}").connect() as c:
            return pd.read_sql("SELECT * FROM contacts", c)
    raise ValueError(f"Unsupported format: {s}")

def _rename(df, col_map):
    return df.rename(columns={v: k for k, v in col_map.items()})

def _clean(df):
    if "last_booking_date" in df.columns:
        df["last_booking_date"] = pd.to_datetime(df["last_booking_date"], errors="coerce")
    for col in ["booking_count_12m", "avg_booking_price"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    for col in df.select_dtypes("object").columns:
        df[col] = df[col].astype(str).str.strip()
    return df
