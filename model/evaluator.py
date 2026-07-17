# model/evaluator.py
import pickle, numpy as np, pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import roc_auc_score, roc_curve
from config import EVAL_K_FRACTIONS, COST_PER_CONTACT_EUR, BOOKING_COLUMNS
from pipeline.data_loader import load_contacts, load_bookings
from pipeline.feature_engineer import build_features
from pipeline.scorer import score_contacts

def evaluate(model_path, contacts_source, bookings_source, event) -> dict:
    contacts = load_contacts(contacts_source); bookings = load_bookings(bookings_source)
    booked_ids = set(bookings[bookings[BOOKING_COLUMNS["booked"]]==1][BOOKING_COLUMNS["contact_id"]].astype(str))
    contacts["_label"] = contacts["id"].astype(str).isin(booked_ids).astype(int)
    model = pickle.load(open(model_path,"rb"))
    X = build_features(contacts, event)
    scores = model.predict_proba(X)[:,1]; y = contacts["_label"]
    rfm = _rfm(contacts); rec = _rec(contacts)
    return {"auc": round(roc_auc_score(y,scores),4),
            "lift_fig": _lift(y,scores,rfm,rec), "roc_fig": _roc(y,scores),
            "precision_at_k": _pak(y,scores,rfm,rec), "roi_table": _roi(y,scores),
            "abc_dist": _abc(scores)}

def _rfm(df):
    now = pd.Timestamp.now()
    d = pd.to_datetime(df.get("last_booking_date"), errors="coerce")
    r = 1-(now-d).dt.days.fillna(730).clip(upper=730)/730
    f = pd.to_numeric(df.get("booking_count_12m",0),errors="coerce").fillna(0)
    m = pd.to_numeric(df.get("avg_booking_price",0),errors="coerce").fillna(0)
    fn = f/f.max() if f.max()>0 else f; mn = m/m.max() if m.max()>0 else m
    return (0.4*r+0.4*fn+0.2*mn).values

def _rec(df):
    now = pd.Timestamp.now()
    d = pd.to_datetime(df.get("last_booking_date"), errors="coerce")
    return (1-(now-d).dt.days.fillna(730).clip(upper=730)/730).values

def _lift(y,ml,rfm,rec):
    fig = go.Figure(); total = y.sum(); n = len(y); x = np.arange(1,n+1)/n*100
    for sc,lbl,col,dash in [(ml,"ML Model","#2E75B6","solid"),(rfm,"RFM Baseline","#ED7D31","dot"),(rec,"Recency-only","#70AD47","dash")]:
        cum = np.array(y)[np.argsort(sc)[::-1]].cumsum()/total*100
        fig.add_trace(go.Scatter(x=x,y=cum,name=lbl,line=dict(color=col,width=2,dash=dash)))
    fig.add_trace(go.Scatter(x=[0,100],y=[0,100],name="Random",line=dict(color="#888",width=1,dash="longdash")))
    fig.update_layout(title="Lift Curve — ML vs SCM Baselines",xaxis_title="% contacts contacted",
                      yaxis_title="% bookers reached (fill rate)",height=420,legend=dict(x=0.05,y=0.95),
                      margin=dict(l=50,r=20,t=50,b=50))
    return fig

def _roc(y,scores):
    fpr,tpr,_ = roc_curve(y,scores); auc = roc_auc_score(y,scores)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr,y=tpr,name=f"ML (AUC={auc:.3f})",line=dict(color="#2E75B6",width=2)))
    fig.add_trace(go.Scatter(x=[0,1],y=[0,1],name="Random",line=dict(color="#888",width=1,dash="dash")))
    fig.update_layout(title="ROC Curve",xaxis_title="False Positive Rate",yaxis_title="True Positive Rate",
                      height=380,margin=dict(l=50,r=20,t=50,b=50))
    return fig

def _pak(y,ml,rfm,rec):
    n=len(y); base=float(y.mean()); rows=[]
    for frac in EVAL_K_FRACTIONS:
        k=max(1,int(n*frac))
        def p(sc): return float(np.array(y)[np.argsort(sc)[::-1][:k]].mean())
        rows.append({"Top K":f"Top {int(frac*100)}% ({k:,})","Random":f"{base:.1%}",
                     "Recency-only":f"{p(rec):.1%}","RFM heuristic":f"{p(rfm):.1%}",
                     "ML model":f"{p(ml):.1%}","Lift vs random":f"{p(ml)/base:.1f}×" if base>0 else "—"})
    return pd.DataFrame(rows)

def _roi(y,scores):
    n=len(y); base=float(y.mean()); rows=[]
    for frac in EVAL_K_FRACTIONS:
        k=int(n*frac); idx=np.argsort(scores)[::-1][:k]
        prec=float(np.array(y)[idx].mean())
        cost=k*COST_PER_CONTACT_EUR; bk=k*prec
        cpb_m=cost/bk if bk>0 else float("inf"); cpb_r=COST_PER_CONTACT_EUR/base if base>0 else float("inf")
        rows.append({"Contacts":f"{k:,}","Expected bookings":f"{bk:.0f}",
                     "Outreach cost":f"€{cost:,.0f}","Cost/booking (model)":f"€{cpb_m:.1f}",
                     "Cost/booking (random)":f"€{cpb_r:.1f}","Cost saving":f"{(1-cpb_m/cpb_r)*100:.0f}%"})
    return pd.DataFrame(rows)

def _abc(scores):
    n=len(scores); a=int(n*.2); b=int(n*.5); s=np.sort(scores)[::-1]
    return {"A (top 20%)":{"count":a,"avg_score":float(s[:a].mean())},
            "B (next 30%)":{"count":b-a,"avg_score":float(s[a:b].mean())},
            "C (bottom 50%)":{"count":n-b,"avg_score":float(s[b:].mean())}}
