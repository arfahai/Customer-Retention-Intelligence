from pathlib import Path
import json,pandas as pd,streamlit as st
ROOT=Path(__file__).resolve().parent; ART=ROOT/"artifacts"; path=ART/"customer_scores.csv"
st.set_page_config(page_title="Customer Retention Intelligence",layout="wide"); st.title("Customer Retention Intelligence")
if not path.exists(): st.error("Run the offline pipeline first: train_churn → clv → score_customers."); st.stop()
s=pd.read_csv(path); meta=json.loads((ART/"model_metadata.json").read_text()) if (ART/"model_metadata.json").exists() else {}
a,b,c,d,e=st.tabs(["Executive Overview","Customer Lookup","Retention Queue","Model Validation","Methodology"])
with a:
    x,y,z=st.columns(3); x.metric("Eligible customers",len(s)); y.metric("Priority customers",(s.RetentionPriority=="Priority").sum()); z.metric("Total EVaR",f"£{s.ExpectedValueAtRisk.sum():,.0f}"); st.dataframe(s.nlargest(20,"ExpectedValueAtRisk"),use_container_width=True)
with b:
    cid=st.number_input("Customer ID",min_value=0,value=int(s["Customer ID"].iloc[0])); st.dataframe(s[s["Customer ID"].eq(cid)].T,use_container_width=True)
with c:
    n=st.slider("Queue size",1,min(500,len(s)),min(100,len(s))); q=s.nlargest(n,"ExpectedValueAtRisk"); st.dataframe(q,use_container_width=True); st.download_button("Download queue",q.to_csv(index=False).encode(),"retention_queue.csv","text/csv")
with d: st.write("Final temporal-test metrics are documented in notebooks/analysis_final.ipynb.")
with e:
    st.write("Target:",meta.get("target")); st.write("Production refit:",meta.get("production_refit")); st.write("CLV: 12-month PredictedGrossPurchaseCLV; one-time customers use historical value fallback."); st.info("EVaR is a prioritization score, not a direct revenue forecast.")
