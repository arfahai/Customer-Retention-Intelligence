
## 1. Project Overview

**Customer Retention Intelligence** is a customer-level analytics and machine-learning prototype for identifying customers who are likely to make **no valid purchase in the following 180 days**, estimating future gross purchase value, and prioritizing a limited retention queue.

The project combines:

- behavioral feature engineering
- temporal no-return prediction
- probability calibration
- customer lifetime value (CLV) estimation
- Expected Value at Risk (EVaR) prioritization
- Streamlit deployment

The final analytical notebook follows the workflow **evidence → experiment → comparison → decision → final implementation**.

---

## 2. Business Problem

A retail business cannot give high-touch retention treatment to every customer. The objective is therefore to create a reproducible decision-support workflow that answers three questions:

1. Which currently active customers are at risk of not returning within the next 180 days?
2. What is the customer's estimated future gross purchase value?
3. Given limited retention capacity, which customers should enter the priority queue?

The output is a **prioritization system**, not an automated decision about customers and not a direct revenue forecast.

---

## 3. Dataset

The project uses the **UCI Online Retail II** transaction dataset.

The source workbook is expected at:

```text
data/online_retail_II.xlsx
```

The workbook contains transaction-level retail records with fields including invoice, product, quantity, price, invoice date, customer ID, and country information.

Customer-level modeling is restricted to records with an identified `Customer ID`.

---

## 4. Data Preparation

The common preprocessing pipeline in `src/common.py`:

1. Loads both workbook sheets.
2. Combines them into one transaction table.
3. Parses `InvoiceDate` as a datetime.
4. Removes exact duplicate rows.
5. Identifies cancellation invoices using the `C` invoice prefix.
6. Classifies positive-quantity, positive-price transactions as `Valid Purchase`.
7. Places remaining transactions in `Other`.

The resulting transaction types are used consistently by the downstream modeling scripts.

### Modeling views

The project separates:

- **identified-customer view** — records with non-null `Customer ID`
- **valid-purchase view** — transactions classified as `Valid Purchase`
- **prediction cohort** — customers with at least one valid purchase in the relevant 180-day observation window
- **CLV view** — valid purchases aggregated to one customer-date transaction

---

## 5. Prediction Problem

### Target

```text
NoPurchaseNext180Days
```

A customer is labeled `1` when they have at least one valid purchase in the 180-day observation window but make **no valid purchase during the following 180-day outcome window**.

This is an **operational 180-day non-return definition**. It should not be interpreted as a claim that the customer has permanently churned.

### Observation and outcome windows

For each snapshot:

```text
180-day observation window
          ↓
      snapshot cutoff
          ↓
180-day outcome window
```

The production scoring cutoff is **9 June 2011**.

---

## 6. Temporal Evaluation Design

The analytical notebook uses historical snapshots instead of a random train/test split so that model evaluation respects time ordering.

The documented development sequence is:

| Period | Role |
|---|---|
| June 2010 | Training |
| September 2010 | Training |
| December 2010 | Validation |
| March 2011 | Diagnostic only |
| June 2011 | Final test |

The final test period is kept separate from model selection and tuning.

The production model is refit using the approved training + validation development data, while the final test period remains excluded from production fitting.

---

## 7. Customer Features

The core behavioral features include:

- `RecencyDays`
- `PurchaseOrderCount`
- `AvgOrderValue`
- `TenureDays`
- `MedianPurchaseGap`
- `SinglePurchaseHistory`

The production scoring pipeline also includes the cancellation-derived features evaluated in the analytical workflow:

- `CancellationOrderCount`
- `CancellationRate`
- `CancellationValueRatio`

Gap-based features that are structurally unavailable for one-time customers are handled explicitly rather than dropping those customers.

---

## 8. Model Development

The final churn/no-return model is a **Logistic Regression** pipeline with:

- `log1p` transformation for `PurchaseOrderCount` and `AvgOrderValue`
- logistic regression classifier
- sigmoid probability calibration
- temporal development/validation design

The analytical notebook records the experiments used before freezing the final methodology, including:

- redundant purchase-gap feature analysis
- feature ablations
- skew/transformation testing
- recency-only baseline
- cancellation-feature contribution
- calibration
- threshold/capacity behavior

### Why probability calibration?

The downstream retention layer uses a probability as an input to EVaR. Calibration is therefore evaluated alongside discrimination metrics such as ROC-AUC and PR-AUC.

---

## 9. Customer Lifetime Value (CLV)

CLV is estimated only for the identified-customer valid-purchase population.

### Transaction grain

Valid purchases are aggregated to:

```text
Customer ID + Purchase Date
```

This customer-date grain is then passed to `lifetimes`.

### Models

The CLV workflow uses:

1. **BG/NBD** — models expected future purchase frequency.
2. **Gamma-Gamma** — models expected monetary value per purchase for repeat-purchase customers.

The primary CLV horizon is:

```text
12 months
```

with a monthly discount-rate assumption of `1%` in the final prototype calculation.

The resulting field is named:

```text
PredictedGrossPurchaseCLV
```

It represents predicted **gross purchase value**, not profit or net economic CLV.

### One-time customers

Customers with no repeat-purchase frequency are retained in the decision layer rather than discarded.

For these customers:

```text
ValueForRetention = HistoricalPurchaseValue
```

For CLV-eligible repeat customers:

```text
ValueForRetention = PredictedGrossPurchaseCLV
```

---

## 10. Retention Prioritization

The final decision layer combines risk and value using:

```text
ExpectedValueAtRisk
    = CalibratedNoReturnProbability
      × ValueForRetention
```

EVaR is explicitly treated as a **prioritization score**, because the probability covers a 180-day non-return outcome while the CLV component represents a 12-month gross-purchase value.

The production queue is capacity-based rather than using test-set quartiles as production thresholds.

The current deployment assigns:

```text
Top 100 customers → Priority
Remaining eligible customers → Standard
```

The queue can be downloaded directly from the Streamlit application.

---

## 11. Production Artifacts

The `artifacts/` directory stores outputs used by the deployed application.

| Artifact | Purpose |
|---|---|
| `churn_pipeline.joblib` | Saved calibrated churn/no-return prediction pipeline |
| `churn_scores.csv` | Saved churn scoring output |
| `clv_scores.csv` | Customer-level CLV and value-for-retention output |
| `customer_scores.csv` | Final customer intelligence and retention-priority output |
| `model_metadata.json` | Production model configuration and metadata |

Artifacts allow the Streamlit application to read precomputed results rather than retraining models during normal dashboard use.

---

## 12. Project Structure

```text
Customer_Retention_Intelligence_Final_Expert_Compliant_Deployment/
│
├── app.py
├── README.md
├── requirements.txt
│
├── data/
│   ├── online_retail_II.xlsx
│   └── README.md
│
├── notebooks/
│   └── analysis_final.ipynb
│
├── src/
│   ├── common.py
│   ├── train_churn.py
│   ├── validate_model.py
│   ├── clv.py
│   └── score_customers.py
│
└── artifacts/
    ├── churn_pipeline.joblib
    ├── churn_scores.csv
    ├── clv_scores.csv
    ├── customer_scores.csv
    └── model_metadata.json
```

---

## 13. Installation

### Python

The project is intended for a Python 3.10–3.12 environment. The exact package versions are defined in `requirements.txt`.

### Create virtual environment

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
```

### Install dependencies

```powershell
pip install -r requirements.txt
```

> The dependency pins in `requirements.txt` should be kept synchronized with the Python version used for deployment.

---

## 14. Running the Offline Pipeline

Place the source workbook at:

```text
data/online_retail_II.xlsx
```

Then run the pipeline in this order:

```powershell
python src\train_churn.py
python src\clv.py
python src\score_customers.py
```

Expected sequence:

```text
Raw transaction data
        ↓
train_churn.py
        ↓
churn_pipeline.joblib
        ↓
clv.py
        ↓
clv_scores.csv
        ↓
score_customers.py
        ↓
customer_scores.csv
```

`validate_model.py` can be used to calculate validation metrics for the saved production pipeline on the configured scoring cohort.

---

## 15. Running the Streamlit Application

After the offline pipeline has generated the required artifacts:

```powershell
streamlit run app.py
```

The application provides:

- **Executive Overview** — eligible customers, priority customers, total EVaR, and highest-EVaR customers
- **Customer Lookup** — inspect a customer by ID
- **Retention Queue** — view and download the retention queue
- **Model Validation** — reference to final validation/test documentation
- **Methodology** — target, production-refit information, CLV approach, and EVaR interpretation

The Streamlit application **does not train models or fit Lifetimes**. It reads saved artifacts generated by the offline pipeline.

---

## 16. Reproducibility

For reproducibility:

1. Use the specified source workbook.
2. Use the package versions in `requirements.txt`.
3. Run the preprocessing and offline pipeline scripts in the documented order.
4. Keep the analytical notebook as the record of experiments and methodological decisions.
5. Keep the saved artifacts generated from the same source data and code version.

The notebook separates research/evaluation from production scoring:

```text
Research / Evaluation
    → temporal experiments and final test

Production
    → approved development-data refit
    → full eligible population scoring

Decision Layer
    → calibrated no-return probability × value for retention

Deployment
    → saved artifacts + Streamlit dashboard
```

---

## 17. Key Limitations

- `NoPurchaseNext180Days` is an operational non-return proxy, not permanent churn.
- CLV is a gross-purchase-value estimate, not profit.
- EVaR is a prioritization score, not a direct 180-day revenue forecast.
- The retention queue identifies customers for intervention; it does not prove that an intervention will cause an incremental purchase.
- Retention effectiveness should ultimately be measured through controlled experimentation using incremental repeat-purchase and revenue outcomes.
- Historical transaction data cannot by itself establish causal treatment effects.

---

## 18. Recommended Retention Evaluation

The deployment is intended to support a measurable retention experiment. Recommended operational evaluation includes:

| Segment | Example action | Success metric | Evaluation period |
|---|---|---|---|
| Priority / highest EVaR | Personalized or high-touch outreach | Incremental 30/60/90-day repeat purchase and incremental revenue | 30/60/90 days |
| High risk, lower value | Automated re-engagement | Repeat-purchase rate and cost per retained customer | 30/60 days |
| Standard | Existing lifecycle communication | Baseline repeat-purchase rate | 60/90 days |

The maximum contact cost should be defined from campaign economics before launch.

---

## 19. Analytical Reference

The full analytical record, experiments, comparisons, methodological decisions, and final implementation are contained in:

```text
notebooks/analysis_final.ipynb
```

The notebook should be read alongside this README when reviewing or studying the project methodology.

