from pathlib import Path
import json,joblib,numpy as np,pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from common import load_transactions,build_snapshot,attach_cancellation_features
ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/"data"/"online_retail_II.xlsx"
r=load_transactions(DATA); ct=r[r["Customer ID"].notna()].copy()
cuts=[pd.Timestamp("2010-06-09"),pd.Timestamp("2010-09-09"),pd.Timestamp("2010-12-09")]
s=pd.concat([build_snapshot(c,ct) for c in cuts],ignore_index=True); tr=s[s.SnapshotCutoff.isin(cuts[:2])].copy(); va=s[s.SnapshotCutoff.eq(cuts[2])].copy()
tr=attach_cancellation_features(tr,ct); va=attach_cancellation_features(va,ct)
for d in [tr,va]: d[["MedianPurchaseGap","MeanPurchaseGap","PurchaseGapCount"]]=d[["MedianPurchaseGap","MeanPurchaseGap","PurchaseGapCount"]].fillna(0)
features=["RecencyDays","PurchaseOrderCount","AvgOrderValue","TenureDays","MedianPurchaseGap","SinglePurchaseHistory","CancellationOrderCount","CancellationRate","CancellationValueRatio"]; logcols=["PurchaseOrderCount","AvgOrderValue"]
pipe=Pipeline([("feature_transform",ColumnTransformer([("log1p",FunctionTransformer(np.log1p,validate=False),logcols)],remainder="passthrough",verbose_feature_names_out=False)),("classifier",LogisticRegression(max_iter=2000,random_state=42))])
model=CalibratedClassifierCV(pipe,method="sigmoid",cv=5); model.fit(pd.concat([tr,va])[features],pd.concat([tr,va])["NoPurchaseNext180Days"])
a=ROOT/"artifacts"; a.mkdir(exist_ok=True); joblib.dump(model,a/"churn_pipeline.joblib")
(a/"model_metadata.json").write_text(json.dumps({"target":"NoPurchaseNext180Days","features":features,"log1p_features":logcols,"production_refit":"train+validation only","final_test_excluded":True},indent=2))
print("Saved churn_pipeline.joblib")
