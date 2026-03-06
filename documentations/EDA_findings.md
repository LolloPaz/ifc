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


## Section 3 — Missing Values: Findings

### 3.1 Overview

Only 3 columns have missing values:

| Column | Missing | % | Type |
|--------|---------|---|------|
| `province` | 919 | 7.77% | Random (MCAR) |
| `roe` | 45 | 0.38% | Structural |
| `leverage` | 45 | 0.38% | Structural |

The dataset is **exceptionally clean** — no balance sheet or income statement
items are missing.

---

### 3.2 ROE & Leverage — Structural Missingness (100% signal for class D)

All 45 null `roe` and `leverage` rows belong to the **exact same observations**
and all have **negative shareholders equity**.

When equity is negative:
- `roe = net_profit / equity` → mathematically valid but financially meaningless
- `leverage = total_debt / equity` → negative leverage is uninterpretable

The dataset correctly marks these as `NaN` rather than showing a misleading value.

**Critical finding: 100% of rows where ROE is null are class D.**

This means null `roe`/`leverage` is not a data quality issue —
it is the **strongest possible signal of distress** in the entire dataset.

→ **Do NOT impute these with median.** Instead:
1. Create `roe_is_null` binary feature (= 1 when equity is negative)
2. Impute the raw value with an **extreme negative sentinel** (e.g. `-99`)
   to preserve the distress signal for tree-based models
3. For linear models, impute with a value well below the 1st percentile

---

### 3.3 Negative Equity Deep Dive

Negative equity is **exclusive to class D** — zero cases in A, B, or C.
However it only covers **4.3% of all D rows** (45 / 1,058).

| Finding | Implication |
|---|---|
| Negative equity never appears in A/B/C | `roe_is_null` is a **perfect precision** signal — when it fires, it is always D |
| Only 4.3% of D rows have negative equity | `roe_is_null` has **very low recall** — catches only a small fraction of distressed companies |

This means negative equity is a **late-stage distress indicator** — the company
is already deep in trouble. The model must detect D much earlier using
deterioration signals from other features (leverage trend, ROI decline,
liquidity squeeze) before equity turns negative.

→ `roe_is_null` is a **high-precision, low-recall** feature. Keep it — it will
likely become a top split in tree-based models for class D. The bulk of D
detection must come from financial ratio trends.

---

### 3.4 Province — Missing Completely at Random (MCAR)

Province missingness rate is **uniform across all classes** (~7.2–7.9%):

| Class | Province missing % |
|-------|--------------------|
| A | 7.2% |
| B | 7.9% |
| C | 7.5% |
| D | 7.9% |

No correlation with class, sector, or fiscal year (stable ~229 missing per year).
This is **Missing Completely at Random (MCAR)** — likely companies that registered
without a province code in the source database.

→ **Impute with a new category `"UNKNOWN"`** — do not drop rows or use
mode imputation, as province is a categorical with 107 levels and the
missing pattern carries no financial signal.

---

### 3.5 Missingness is Stable Across Years

No year concentration — ~11–12 null `roe`/`leverage` and ~229 null `province`
per year consistently. No COVID-related reporting gaps detected.

---

### Preprocessing Decisions from Section 3

| Column | Treatment | Rationale |
|--------|-----------|-----------|
| `roe`, `leverage` | Add `roe_is_null` binary flag → impute with sentinel `-99` | Perfect D signal, must preserve |
| `province` | Impute with `"UNKNOWN"` category | MCAR, no financial signal |
| All other columns | No imputation needed | Dataset is complete |

### 4. Descriptive Statistics
- Summary stats for all numerical columns
- Log-scale histograms for balance sheet items (heavy right skew expected)
- Check impossible values: negative revenue, `debt_to_assets > 1`


→ These are legitimate extreme distress observations — do not drop them.
→ Winsorize `leverage` at 99th percentile but preserve the `roe_is_null` flag.

---

### 4.5 Critical Finding: Dataset is Synthetically Generated

findings:

#### Evidence

Ratio recomputation tests confirm perfect internal consistency:

| Test | Max error | Verdict |
|---|---|---|
| ROE recomputation | 5e-05 (float rounding) | ✅ Perfect |
| DTA recomputation | 5e-05 (float rounding) | ✅ Perfect |
| Profit margin recomputation | 5e-05 (float rounding) | ✅ Perfect |
| Accounting identity gap | Exactly 0 | ✅ Perfect |

**A perfect accounting identity gap of 0 across 11,828 rows is impossible
in real financial filing data.** Real statements always contain rounding
differences between filed totals. This is the clearest proof of synthetic
generation.

#### Scale Investigation

Attempting to recover a uniform scale factor:

| Reference | Implied scale vs typical Italian SRL |
|---|---|
| Revenue-based | ~161x |
| Asset-based | ~261x |

The **inconsistency rules out uniform scaling**. The generator did not
simply multiply real values by a constant. Instead it sampled absolute
values from a large-cap size distribution while calibrating financial
ratios to realistic SME dynamics.

The operating cost ratio (median 92.4%, range 58.5%–113.4%) confirms
the ratio dynamics are realistic and match real Italian sector economics.

#### Implications for Modeling

| Feature type | Reliability | Modeling decision |
|---|---|---|
| Financial ratios (`roe`, `roi`, etc.) | ✅ Fully reliable | Primary feature set |
| Ratio trends (YoY changes) | ✅ Fully reliable | Engineer as lag features |
| Operating cost ratio | ✅ Realistic | Use as engineered feature |
| Absolute monetary values | ⚠️ Unrealistic size | Log-transform, use as relative proxy only |
| Size-based benchmarks | ❌ Avoid | Cannot compare to real companies |

→ **Ratios and their trends are the primary feature set.**
→ Absolute values kept only as **relative size proxies** via log transformation.
→ Synthetic nature acknowledged as a **limitation** in the final presentation.

---

### Preprocessing Decisions from Section 4

| Variable | Treatment | Rationale |
|---|---|---|
| Balance sheet & income statement | `np.log1p` transform | Correct right skew for linear/distance models |
| `shareholders_equity` | `np.log1p` on positive values only + flag negatives | Bimodal distribution |
| `roe` | Winsorize at 1st/99th percentile | Extreme left outliers (-39.2) |
| `leverage` | Winsorize at 99th percentile | 5 extreme outliers (max 101) |
| `roi`, `current_ratio`, `quick_ratio` | No transformation | Clean bounded distributions |
| `debt_to_assets`, `profit_margin` | No transformation | Already bounded ranges |
| Extreme distress rows (45 rows) | Keep | Legitimate class D signal |

