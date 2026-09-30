from pathlib import Path
import numpy as np, pandas as pd
from lifetimes.utils import summary_data_from_transaction_data
from lifetimes import BetaGeoFitter, GammaGammaFitter
from common import load_transactions

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "online_retail_II.xlsx"
cutoff = pd.Timestamp("2011-06-09")

r = load_transactions(DATA)
ct = r[r["Customer ID"].notna()].copy()

p = ct[
    (ct.InvoiceDate <= cutoff) &
    (ct.Transaction_Type == "Valid Purchase")
].copy()

p["TransactionValue"] = p.Quantity * p.Price

tx = (
    p.assign(PurchaseDate=p.InvoiceDate.dt.floor("D"))
    .groupby(["Customer ID", "PurchaseDate"], as_index=False)
    .TransactionValue.sum()
)

s = summary_data_from_transaction_data(
    tx,
    customer_id_col="Customer ID",
    datetime_col="PurchaseDate",
    monetary_value_col="TransactionValue",
    observation_period_end=cutoff
).reset_index()

bg = BetaGeoFitter(penalizer_coef=0.1)

bg.fit(
    s["frequency"],
    s["recency"],
    s["T"]
)

rep = s[s["frequency"] > 0].copy()

gg = GammaGammaFitter(penalizer_coef=0.001)

gg.fit(
    rep["frequency"],
    rep["monetary_value"]
)

rep["PredictedGrossPurchaseCLV"] = gg.customer_lifetime_value(
    transaction_prediction_model=bg,
    frequency=rep["frequency"],
    recency=rep["recency"],
    T=rep["T"],
    monetary_value=rep["monetary_value"],
    time=12,
    discount_rate=0.01,
    freq="D"
)

hist = (
    tx.groupby("Customer ID")["TransactionValue"]
    .sum()
    .rename("HistoricalPurchaseValue")
    .reset_index()
)

out = (
    s
    .merge(hist, on="Customer ID", how="left")
    .merge(
        rep[["Customer ID", "PredictedGrossPurchaseCLV"]],
        on="Customer ID",
        how="left"
    )
)

out["CLVStatus"] = np.where(
    out["frequency"] > 0,
    "Predicted CLV Eligible",
    "Not Eligible - One-Time Customer"
)

out["ValueForRetention"] = np.where(
    out["CLVStatus"] == "Predicted CLV Eligible",
    out["PredictedGrossPurchaseCLV"],
    out["HistoricalPurchaseValue"]
)

out.to_csv(
    ROOT / "artifacts" / "clv_scores.csv",
    index=False
)

print("Saved clv_scores.csv")