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

### 6 
All 6 features are perfectly monotone. This is the best possible outcome.

Profitability features — strongest discriminators:

roe drops from 0.270 → -0.105: the jump from C to D is a cliff, not a step. Class D is not just "less profitable" — it's loss-making at the median.
roi decays smoothly 0.158 → 0.007: even class D companies generate marginally positive operating income at the median, meaning the distress signal in D comes more from the balance sheet than the P&L.
profit_margin mirrors roe — goes negative at D (−0.015). Consistent with a company where costs exceed revenue.

Balance sheet features — clean structural gradient:

debt_to_assets rises 0.474 → 0.851: class D is almost entirely debt-financed. At 0.851, shareholders' equity covers less than 15% of assets — one bad year away from negative equity.
leverage explodes 0.901 → 5.517: the jump from C (2.692) to D (5.517) is the sharpest step in the entire table. This is where the roe_is_null (negative equity) cases start pulling the median up.
current_ratio decreases 2.482 → 1.364: class D still has a ratio above 1, meaning liquidity is not the primary failure mode — solvency is. This matters for modeling: don't over-weight short-term liquidity features.


Modeling decisions this produces:

Ordinal encoding is fully justified — the gradient is real and consistent across all 6 features. We can use ordinal-aware losses or treat the problem as ordered classification.
roe, leverage, and debt_to_assets are the three strongest raw discriminators — they will dominate feature importance. Engineering their year-over-year trends is the highest-priority feature engineering task.
The C→D boundary is the hardest to learn from liquidity alone — current_ratio barely moves between C and D (1.461 vs 1.364). The model will need balance sheet deterioration trends, not just snapshots, to catch companies transitioning into D.
profit_margin and roi are correlated — both measure profitability. Worth checking in §7 whether they're redundant enough to drop one.


# §7 Findings — Correlation Analysis
The results are more severe than anticipated. There are two distinct collinearity problems that need separate treatment.

Problem 1: The absolute monetary variables are one giant collinearity cluster
Every balance sheet and income statement variable — total_fixed_assets, current_assets, shareholders_equity, total_debt, short_term_debt, long_term_debt, production_value, production_costs — correlates with every other at r ≥ 0.85, many at r ≥ 0.96. This is not a coincidence — they all measure the same thing: firm size. A bigger company has more assets, more debt, more revenue, more costs. There is essentially one latent factor here.
Decision: drop all 8 raw absolute variables from the feature set entirely. They carry no class signal beyond what the ratios already capture (confirmed by §6 — ratios showed perfect monotonicity, raw absolutes weren't even in that analysis). Keep only log(total_assets) as a single size proxy, log-transformed as decided in §4.

Problem 2: Three ratio pairs are perfectly redundant by construction

production_costs ↔ production_value r=+1.000 — same variable scaled
quick_ratio ↔ current_ratio r=+1.000 — in this dataset inventory is negligible, making them identical
debt_to_assets ↔ leverage r=+1.000 — two encodings of the same solvency concept

Decision: drop production_costs, quick_ratio, and leverage. Keep production_value (or derive operating_cost_ratio from it), current_ratio, and debt_to_assets.

Problem 3: roi ↔ roe r=+0.962 — near-redundant profitability pair
From §6 both showed monotone decreasing gradients. From §7 they are almost the same feature. However they are not identical — roi uses total assets as denominator (operational efficiency), roe uses equity (shareholder return). For class D companies with negative equity, roe breaks down while roi remains interpretable.
Decision: keep roi, drop roe as a raw feature. The roe_is_null binary flag (engineered in §3) already captures the unique information roe carries for class D.

Clean feature set entering feature engineering
KeepDropReasonlog(total_assets)all other absolutessize proxy onlyroiroeredundant with roi, roe_is_null flag keptcurrent_ratioquick_ratior=1.00debt_to_assetsleverager=1.00profit_marginproduction_costsr=1.00 with production_valueproduction_value (log)—size-relative revenueyears_in_business—independent, no collinearity
net_profit_loss and operating_income also correlate heavily with the size cluster — drop both, they are already encoded in roi and profit_margin.


### §8 Findings — Panel Completeness

96.5% of companies have a full 4-year history. Lag features are safe to engineer at scale. Only 104 companies (3.5%) have incomplete panels — this is a negligible edge case, not a structural problem.

The 25.4% NaN rate for lag features needs careful reading.
2,999 rows will have NaN lag features — one per company (their first observed year, where no t-1 exists). This is not a problem with the data, it's a mechanical consequence of the panel structure. Every company loses exactly one row to NaN lags regardless of how many years it has. The imputation strategy for these rows is: sector-year median, fit on training data only.

Early exits are a distress signal, late entrants are not.

67 early exits (stopped filing before 2021) — from §1 findings we know these skew heavily toward class D
38 late entrants (entered after 2018) — neutral, just young companies
The right chart confirms this: 39.1% of incomplete companies are class D, versus only 8.9% in the full dataset. Class D is 4.4x overrepresented among companies that disappeared from the panel.

This validates the is_last_observation binary flag engineered in §1 — it is capturing real distress, not noise.

Modeling decisions this produces:

Lag features are greenlit — 96.5% coverage means NaN imputation affects a small, well-understood minority of rows. No need for fallback architectures.
Impute NaN lags with sector-year median, fit on train only. Do not use global median — sector context matters (a construction company's median ROI differs from an IT company's).
n_years_in_panel and is_last_observation are confirmed features — the class D overrepresentation among incomplete companies proves these carry real predictive signal.
Do not drop incomplete companies — their 39.1% D rate makes them some of the most informative rows in the dataset for detecting distress. Dropping them would hurt recall on the hardest class to predict.



## §9 Findings — Sector & Geography

No rare sectors exist — cardinality strategy changes
Every ATECO sector has ≥50 observations. The planned "OTHER" grouping is not needed. Keep all sectors at full granularity — no cardinality reduction required.

Sector D-rate spread is narrow — weaker signal than expected
The D-rate ranges only from 6.4% (sector 62, IT) to 10.5% (sector 10, Food Manufacturing) — a spread of just 4 percentage points across the top 10 sectors. Compare this to §2's finding that class B dominates at 59% everywhere: sector shifts the D-rate by ±2% around the baseline, not by 10–15% as we hoped.
This is a significant downgrade from the §2 hypothesis. Sector is a weaker standalone discriminator than anticipated. It will not be a top feature by importance.
However the ROI/ROE medians tell a different story — sector 56 (Food & Beverage service) has median ROI of 0.189 vs sector 71 (Professional services) at 0.086. That's a 2x difference in profitability baseline across sectors. This is exactly why sector-relative ratios matter: a company in sector 71 with ROI=0.10 is performing well above its peers, while the same ROI=0.10 in sector 56 signals underperformance.
The sector signal lives in the residuals, not in the raw sector label. sector_roi_delta = roi - sector_median_roi and sector_roe_delta = roe - sector_median_roe will capture this — the raw ateco_sector label alone will not.

years_in_business is flat — confirmed dead feature
Median years in business: A=36, B=34, C=36, D=37. The boxes in the plot are virtually identical across all four classes. There is no age effect whatsoever — company age does not predict financial health in this dataset.
This contradicts the §8 hypothesis that older companies fail more due to survivorship. The reality is that the distribution is simply uniform — companies of all ages appear in all classes.
Decision: drop years_in_business from the feature set entirely. It carries no signal and will only add noise.

Modeling decisions this produces

Drop years_in_business — confirmed zero predictive power, flat across all classes
Keep ateco_sector at full granularity — no rare sectors to collapse
Engineer sector_roi_delta and sector_roe_delta — the real sector signal is in peer-relative performance, not the sector label itself. These are now the highest-priority engineered features from this section
Downgrade sector importance expectations — the raw ateco_sector one-hot will likely rank low in feature importance. Don't over-engineer sector interactions