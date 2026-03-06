## 1. Dataset Structure
- Column types and non-null counts
- Temporal split sanity check (train 2018–2021, test 2022–2023)
- Unique companies and rows per company
- Scalers and imputers must be **fit on train only**, then applied to test
- Lag features must use **only past data** (year `t-1` → year `t`)
- Never use features from year `t+1` to predict class at year `t`

Findings:
---

### 🔍 Finding: Incomplete Companies are Not Missing at Random

**104 companies have fewer than 4 years** in the training set:

| Pattern | Count | Interpretation |
|---|---|---|
| Missing at **end** (early exits) | 130 observations | Company stopped filing — likely distress |
| Missing at **start** (late entrants) | 38 observations | Young company, entered after 2018 |

**Early exits are strongly associated with class D:**

| Class | Early Exits | Full Dataset |
|-------|-------------|--------------|
| A     | 2.9%        | 8.5%         |
| B     | 13.1%       | 59.3%        |
| C     | 19.0%       | 23.2%        |
| D     | **65.0%**   | **8.9%**     |

65% of companies that disappeared before 2021 were class D — vs only 8.9% in the full dataset.  
This is **informative missingness**, not random. These companies stopped filing due to severe financial distress.

> **Survivorship bias**: companies present in the test set (2022–2023) are survivors by definition.  
> The model may underestimate D risk for companies that will exit mid-period. This is a known limitation.

---

### Time Dimension as an Asset: Lag Features

Since most companies have 4 consecutive years of contiguous data, we can engineer **trend features**  
that capture whether a company's financial health is improving or deteriorating.

A company with ROE dropping from `0.4 → 0.1` is riskier than one stable at `0.15`,  
even though the current value is higher. Raw snapshots miss this signal.

**Features to engineer (computed at year `t` using year `t-1`)**:

| Feature | Formula | Signal |
|---|---|---|
| `roe_prev` | `roe` at `t-1` | Profitability baseline |
| `roi_prev` | `roi` at `t-1` | Efficiency baseline |
| `current_ratio_prev` | `current_ratio` at `t-1` | Liquidity baseline |
| `roe_yoy` | `roe_t - roe_{t-1}` | Profitability trend |
| `roi_yoy` | `roi_t - roi_{t-1}` | Efficiency trend |
| `leverage_trend` | `leverage_t - leverage_{t-1}` | Increasing debt signal |
| `equity_growth` | `(equity_t - equity_{t-1}) / abs(equity_{t-1})` | Capital erosion signal |
| `is_last_observation` | 1 if final row for company | Exit/distress signal |
| `n_years_in_panel` | count of years in dataset | Short history = higher risk |

> Late entrants (38 companies missing 2018) will have `NaN` lag features on their first row → imputed with median during preprocessing.  
> `is_last_observation` and `n_years_in_panel` must be computed **before** the train/test split to avoid leakage.

---

### Cross-Validation Strategy

Standard k-fold CV is **not valid** here because it would leak future data into training.  
Use **time-based expanding window CV** on the training set:

| Fold | Train | Validation |
|---|---|---|
| 1 | 2018 | 2019 |
| 2 | 2018–2019 | 2020 |
| 3 | 2018–2020 | 2021 ← most important |

Use `sklearn.model_selection.TimeSeriesSplit` or build manually on `fiscal_year`.  
**Never shuffle** when splitting.

---

### Full Modeling Flow
Sort by company_id + fiscal_year

Compute panel features: is_last_observation, n_years_in_panel

Engineer lag features (shift within each company group)

Temporal train/val split for CV (no shuffle)

Fit scaler + imputer on train only → transform val and test

Train classifier on all train years (2018–2021)

Predict on test set (2022–2023)

Evaluate: Weighted F1 (primary), Confusion Matrix, Per-class Precision/Recall


## 2. Target Variable Analysis
- Class counts and percentages (A/B/C/D)
- Class distribution per `fiscal_year` — detect COVID drift in 2020–2021
- Class distribution by `ateco_sector`, `legal_form`, `region`

findings:

## Section 2 — Target Variable Analysis: Findings

### 2.1 Class Imbalance

The dataset is **moderately imbalanced**:

| Class | Count | % |
|-------|-------|---|
| A — Excellent | 1,003 | 8.5% |
| B — Good | 7,017 | 59.3% |
| C — Moderate risk | 2,750 | 23.3% |
| D — High risk | 1,058 | 8.9% |

B dominates the dataset. A and D are the minority classes and the hardest to classify correctly.
This means **accuracy is a misleading metric** — a model predicting B for everything would reach 59% accuracy.
→ Use **Weighted F1** as primary metric. Apply `class_weight="balanced"` in all models.

---

### 2.2 No COVID Drift Detected

Class distribution is **remarkably stable** across all 4 years:

| Year | A | B | C | D |
|------|---|---|---|---|
| 2018 | 254 | 1815 | 640 | 252 |
| 2019 | 260 | 1766 | 691 | 262 |
| 2020 | 239 | 1745 | 695 | 277 |
| 2021 | 250 | 1691 | 724 | 267 |

No significant spike in D during 2020–2021 despite COVID-19.
C shows a mild upward trend (+13% from 2018 to 2021) but no structural shift.
→ `fiscal_year` is **not a strong standalone predictor** — but keep it as a feature
to capture any subtle macro trend the model can learn.

---

### 2.3 Legal Form has Minimal Predictive Power

All legal forms (SRL, SPA, SAS, SNC, SAPA) show nearly identical class distributions,
all within ±2% of the global baseline.
→ `legal_form` is a **weak feature**. Encode it but don't expect high importance.

---

### 2.4 Region has Minimal Predictive Power

Regional differences are small. Notable exceptions:
- **Liguria** has the highest C rate (27.7%) and lowest B rate (53.9%) — slightly riskier
- **Piemonte** has the highest B rate (62.3%) — slightly healthier
- **Veneto** has the lowest A rate (7.0%)

Differences are marginal (~5% range across regions).
→ `region` is a **weak feature**. Consider grouping into macro-areas (Nord/Centro/Sud)
to reduce cardinality without losing the small signal.

---

### 2.5 ATECO Sector has Moderate Predictive Power

More meaningful variation than region or legal form:
- **Construction (41, 43)** — highest C rates (30%, 28.8%) and lowest A rates (~6%) →
  capital-intensive sectors with thin margins and high debt
- **Wholesale trade (46, 47)** — highest A rates (10.2%, 11.4%) → healthier cash flow
  businesses
- **IT services (62)** — lowest D rate (6.4%) → knowledge-based sectors are more resilient
- **Food manufacturing (10)** — highest D rate (10.5%)

→ `ateco_sector` is a **meaningful feature**. Keep at full granularity.
Also engineer **sector-relative ratios** (e.g., `roe - sector_median_roe`) to
capture how a company performs vs its peers.

---

### Preprocessing Decisions from Section 2

1. **Never use accuracy** — always Weighted F1
2. **Apply `class_weight="balanced"`** in all classifiers
3. **Keep `fiscal_year`** as a feature (mild macro signal)
4. **Encode `legal_form` and `region`** but expect low importance
5. **Keep `ateco_sector` at full granularity** — engineer sector-relative ratio features
6. **Consider macro-region grouping** (Nord/Centro/Sud) as an additional feature

## 3. Missing Values
- Missing count and % per column
- Structural vs random: `roe`/`leverage` missing when equity = 0
- `province` missingness — correlated with class?
- Missingness heatmap by `fiscal_year`

findings:


