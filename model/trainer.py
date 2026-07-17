# model/trainer.py
import pickle
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from config import MODEL_TYPE, MODEL_PATH, XGBOOST_PARAMS, LIGHTGBM_PARAMS, BOOKING_COLUMNS
from pipeline.data_loader import load_contacts, load_bookings
from pipeline.feature_engineer import build_features, get_feature_names

def train(bookings_source, contacts_source, model_type=MODEL_TYPE,
          model_path=MODEL_PATH, test_size=0.2, random_state=42) -> dict:
    contacts = load_contacts(contacts_source)
    bookings = load_bookings(bookings_source)
    merged = bookings.merge(contacts, left_on=BOOKING_COLUMNS["contact_id"],
                            right_on="id", how="left", suffixes=("_bk",""))
    rows, labels = [], []
    for _, row in merged.iterrows():
        ev = {"topic": str(row.get("event_topic","")), "format": str(row.get("event_format","")),
              "price": float(row.get("event_price",0)), "location": str(row.get("location","")), "audience": []}
        X_r = build_features(pd.DataFrame([row.to_dict()]), ev)
        rows.append(X_r.iloc[0].values); labels.append(int(row.get("booked",0)))
    fnames = get_feature_names()
    X = pd.DataFrame(rows, columns=fnames); y = pd.Series(labels)
    pos_rate = y.mean()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    n_pos = max(int(pos_rate*len(y)),1); n_neg = len(y)-n_pos; spw = round(n_neg/n_pos,2)
    if model_type == "xgboost":
        from xgboost import XGBClassifier
        p = {**XGBOOST_PARAMS, "random_state": random_state}
        if p.get("scale_pos_weight") is None: p["scale_pos_weight"] = spw
        model = XGBClassifier(**p)
    else:
        from lightgbm import LGBMClassifier
        model = LGBMClassifier(**{**LIGHTGBM_PARAMS, "random_state": random_state})
    model.fit(X_tr, y_tr)
    auc = roc_auc_score(y_te, model.predict_proba(X_te)[:,1])
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(model, open(model_path,"wb"))
    return {"auc": round(auc,4), "n_train": len(X_tr), "n_test": len(X_te),
            "model_path": model_path, "feature_names": fnames,
            "positive_rate": round(float(pos_rate),4), "model_type": model_type}
