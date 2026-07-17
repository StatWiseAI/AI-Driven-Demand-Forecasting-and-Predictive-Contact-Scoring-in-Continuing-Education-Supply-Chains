# agent/tools.py
import json, pickle
from pathlib import Path
import pandas as pd
from config import MODEL_PATH, TOP_N, BUFFER_PERCENT, OUTPUT_DIR, OUTPUT_FORMAT, TOP_N_EXPLANATIONS
from pipeline.data_loader import load_contacts
from pipeline.scorer import score_contacts
from pipeline.explainer import explain_batch

def tool_load_contacts(file_path: str) -> dict:
    try:
        df = load_contacts(file_path)
        return {"status":"ok","count":len(df),"columns":df.columns.tolist(),
                "sample":df.head(3).to_dict(orient="records"),
                "message":f"Loaded {len(df):,} contacts successfully."}
    except Exception as e:
        return {"status":"error","message":str(e)}

def tool_run_scoring_model(event: dict, contacts_path: str, model_path: str = MODEL_PATH) -> dict:
    try:
        df = load_contacts(contacts_path)
        scored = score_contacts(df, event, model_path)
        Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
        tmp = Path(OUTPUT_DIR)/"_scored_full.csv"; scored.to_csv(tmp, index=False)
        top5 = scored.head(5).to_dict(orient="records")
        return {"status":"ok","scored_count":len(scored),"top_preview":top5,
                "scored_path":str(tmp),"top_score":round(float(scored["score"].iloc[0]),3),
                "message":f"Scored {len(scored):,} contacts. Top: {scored['score'].iloc[0]:.3f}"}
    except Exception as e:
        return {"status":"error","message":str(e)}

def tool_generate_top_list(scored_path: str, n: int = TOP_N,
                           event_label: str = "event", output_format: str = OUTPUT_FORMAT) -> dict:
    try:
        df = pd.read_csv(scored_path)
        buf_n = int(n*(1+BUFFER_PERCENT/100))
        top = df.head(buf_n)
        Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
        lbl = event_label.replace(" ","_")[:40]
        out = Path(OUTPUT_DIR)/(f"top_{n}_{lbl}.xlsx" if output_format=="excel" else f"top_{n}_{lbl}.csv")
        top.to_excel(out,index=False) if output_format=="excel" else top.to_csv(out,index=False)
        abc = top["abc_tier"].value_counts().to_dict() if "abc_tier" in top.columns else {}
        return {"status":"ok","output_path":str(out),"count":len(top),
                "abc_distribution":abc,
                "message":f"Saved {len(top):,} contacts (top {n:,} + {BUFFER_PERCENT}% buffer) to {out}"}
    except Exception as e:
        return {"status":"error","message":str(e)}

def tool_explain_scores(scored_path: str, event: dict, contacts_path: str,
                        model_path: str = MODEL_PATH, n: int = TOP_N_EXPLANATIONS) -> dict:
    try:
        scored = pd.read_csv(scored_path).head(n)
        contacts = load_contacts(contacts_path)
        model = pickle.load(open(model_path,"rb"))
        id_col = "id"
        if id_col in scored.columns and id_col in contacts.columns:
            extra = [c for c in contacts.columns if c not in scored.columns]
            scored = scored.merge(contacts[[id_col]+extra], on=id_col, how="left")
        exps = explain_batch(scored, event, model, n=n)
        return {"status":"ok","explanations":exps,"message":f"Explained {len(exps)} contacts."}
    except Exception as e:
        return {"status":"error","message":str(e)}
