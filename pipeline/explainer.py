# pipeline/explainer.py
import shap
import pandas as pd
from pipeline.feature_engineer import build_features, get_feature_names, feature_scm_labels

def explain_batch(df: pd.DataFrame, event: dict, model, n: int = 20) -> list:
    top = df.head(n).copy()
    X = build_features(top, event)
    names = get_feature_names()
    labels = feature_scm_labels()
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X)
    if isinstance(sv, list): sv = sv[1]
    results = []
    for i, (_, row) in enumerate(top.iterrows()):
        contribs = dict(zip(names, sv[i].tolist()))
        top3 = sorted(contribs.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
        parts = [f"{labels.get(f,f)} {'↑' if v>0 else '↓'}" for f, v in top3]
        results.append({
            "contact_id": row.get("id", i), "name": row.get("name",""),
            "email": row.get("email",""), "score": round(float(row.get("score",0)),3),
            "abc_tier": row.get("abc_tier","—"),
            "explanation": "Score driven by: " + "; ".join(parts) + ".",
            "shap_values": contribs,
        })
    return results
