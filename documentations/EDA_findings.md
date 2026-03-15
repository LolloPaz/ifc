# 📊 EDA & Modeling: Strategic Decisions
**Project:** Challenge 2 — Financial Health Classification
**Objective:** Multiclass classification (A, B, C, D) of current financial
health ($y_t$) using balance sheet data from the same year ($X_t$),
optionally enriched with historical trend features.

---

## 1. Class Imbalance

The target variable is strongly imbalanced:

| Class | Count | % |
|-------|-------|---|
| A — Excellent    | 1,003 | 8.5% |
| B — Good         | 7,017 | 59.3% |
| C — Moderate Risk| 2,750 | 23.3% |
| D — High Risk    | 1,058 | 8.9% |

B dominates at 59.3%. A and D are minority classes.
**Accuracy is a misleading metric** — a trivial model predicting always B
reaches 59% without learning anything.

→ Primary metric: **Weighted F1-Score**
→ Apply `class_weight="balanced"` in all classifiers

---

## 2. Informative Missingness — Early Exit Pattern

The most critical domain insight concerns companies with incomplete
panel histories.

By cross-referencing train (2018–2021) and test (2022–2023) datasets,
we identified **83 companies that never reappear in the test set** —
true "early exits" from the business registry.

**Their financial trajectory before disappearance:**
- **Across all historical years:** 56% D, 19% B, 19% C, 5% A
- **In their final observed year:** **100% Class D** (no exceptions)

**Statistical validation:** χ²=681.11, p<0.0001 — the missingness
pattern is definitively **not random (MNAR)**.

**Business interpretation:** In the Italian corporate registry, failure
to file financial statements is most likely a proxy signal for
liquidation or severe distress, rather than a data quality artefact.

**Modeling implications:**
- **Never drop these rows** — early exits contain the most valuable
  patterns for detecting fatal deterioration
- **No direct leakage** — "will exit next year" cannot be used as
  a feature. Engineer historical proxies instead:
  `n_years_in_panel`, year-over-year trend features
- `class_prev` — lagged target class at t-1. Valid feature (past
  label is observable at prediction time), but must be joined by
  `company_id + fiscal_year` **before** the CV split, never computed
  inside a fold
- **FE focus:** the B/C → D transition trajectory is a critical
  learning signal

---

## 3. Survivorship Bias — Empirical Assessment

The 83 early exits (Section 2) are absent from the test set by
definition — this constitutes a form of survivorship bias.
However, the practical impact is empirically small: Class D prevalence
varies only between 8.5% and 9.4% across 2018–2021, with no
accelerating trend (χ²=11.64, p=0.234).

**Conclusion:** The phenomenon exists and is acknowledged, but its
magnitude does not represent a critical distributional risk for the
test set. Standard class-balancing techniques remain the primary
mitigation.

---

## 4. Training Instance Definition

The task is **not** forecasting (predicting $y_{t+1}$ from $X_t$).
The objective is diagnosing **current** health state.

Each row is treated as an independent observation:
- **Target ($y_t$):** `financial_health_class` in year $t$
- **Features ($X_t$):** balance sheet snapshot in year $t$,
  enriched with lag features from $t-1$

The model learns the direct mapping $X_t \rightarrow y_t$.

---

## 5. Feature Engineering Architecture — Lag & Trend Features

Financial distress is a cumulative process. The current snapshot
$X_t$ will be enriched with:
- **Lag features:** key ratios at $t-1$ (e.g. `roe_lag1`)
- **Trend features:** YoY delta (e.g. `roe_t - roe_{t-1}`)

**Temporal edge cases:**
- **2018 (first year):** lag features are entirely NaN → 2018 is
  used only as a source for computing 2019 trends. Effective
  training runs on 2019–2021 rows only.
- **Late entrants:** companies appearing after 2018 have NaN lag
  features in their first year → handled natively by tree-based
  models (XGBoost/LightGBM) without global median imputation,
  which would falsify the trend signal.

---

## 6. Validation Strategy — Time-Based Expanding Window

Standard K-Fold with random shuffling is **strictly forbidden**
(temporal data leakage).

**Time-Based Expanding Window** simulates the natural flow of time:

| Fold | Train | Validation |
|------|-------|------------|
| 1 | 2019 | 2020 |
| 2 | 2019 + 2020 | 2021 |

Final validation score = arithmetic mean across both folds.
This guarantees that estimated performance faithfully reflects
what will happen on the test set (2022–2023).

> **Note:** the test set spans 2022–2023, one year beyond the last
> validation fold (2021). Macro conditions may have shifted in the
> gap — monitor test performance relative to fold 2 for unexpected
> distributional drift.

---

## 7. Categorical Features — Predictive Power Assessment

### Legal Form
All five legal forms (SRL, SPA, SAS, SNC, SAPA) are statistically
indistinguishable (<±1pp variation on every class). No logical
mechanism exists by which legal structure would drive financial
health independently of the financials themselves.

→ **Consider dropping** rather than encoding — 4 dummy columns
  wasted on zero-signal features.

### Region
Marginally more informative but signal remains narrow (~5pp range
on Class D). Notable exception: Liguria (C=27.7%, B=53.9%).
Do **not** aggregate into macro-areas (Nord/Centro/Sud) — with only
10 regions cardinality is not an issue, and aggregation destroys
the small existing signal (Liguria and Piemonte are both Nord but
behave oppositely).

→ Encode, but expect low importance.

### ATECO Sector
Mixed statistical picture:
- χ²=190.93, p=2.18e-29 — statistically significant
- Mutual Info=0.0096 — near-zero standalone power

The chi² significance is partly an artifact of large n. The real
signal lies in the **A/C axis**, not Class D (range: 5.1%–9.6%,
only 4.5pp spread):

| Sector | Key Signal |
|--------|------------|
| Construction (41, 43) | C=29–30%, A=6% — chronic deterioration |
| Real Estate (68) | **A=0.0%** — structurally never excellent |
| IT Services (62) | D=6.4% — highest resilience |
| Wholesale Trade (46, 47) | A=10–11% — healthiest cash flow |

→ Keep at full granularity.
→ Primary value: **sector-relative feature engineering**
  (e.g. `roe - sector_median_roe`).
→ Sector medians must be computed on **train set only** —
  computing on full data is data leakage.

---

## 8. Temporal Stability of Target Distribution

Chi² test on the yearly contingency table: **χ²=11.64, p=0.234** —
year-over-year variation is not statistically significant.

| Year | A    | B    | C    | D    |
|------|------|------|------|------|
| 2018 | 8.6% | 61.3%| 21.6%| 8.5% |
| 2019 | 8.7% | 59.3%| 23.2%| 8.8% |
| 2020 | 8.1% | 59.0%| 23.5%| 9.4% |
| 2021 | 8.5% | 57.7%| 24.7%| 9.1% |

No spike in Class D during 2020–2021 despite COVID-19.

→ `fiscal_year` has no demonstrated standalone predictive power.
  Include as a feature but expect near-zero importance.

---

## 9. Preprocessing Decisions — Master Table

| Decision | Rationale |
|----------|-----------|
| Weighted F1 as primary metric | Accuracy misleading under 59% imbalance |
| `class_weight="balanced"` in all models | Prevents majority-class dominance |
| Never drop early exit rows | Most informative distress patterns |
| No "exit next year" feature | Future information — data leakage |
| Sacrifice 2018 as training target | Lag features fully NaN for 2018 |
| Let trees handle lag NaNs natively | Global imputation falsifies trend signal |
| Keep `fiscal_year` as feature | Zero cost, minimal signal |
| Consider dropping `legal_form` | Zero signal, wastes 4 dummy columns |
| Encode `region`, expect low importance | Narrow signal, don't aggregate |
| Keep `ateco_sector` at full granularity | Needed for peer-relative FE |
| Compute sector medians on train only | Prevents data leakage |
| Time-Based Expanding Window validation | Prevents temporal data leakage |
| Flag `is_terminal_distress` before winsorize | Winsorizing first clips the extreme values used for flagging |
| Engineer `op_cost_ratio` explicitly | Not in raw data — strong C/D discriminative signal |
| Log-transform all balance sheet variables | Raw skewness 17–23 confirmed log-normal |
| Avoid linear models without VIF selection | Profitability and debt features are internally correlated |

---

## 10. Missing Values — Pattern Analysis

Three distinct missingness mechanisms identified:

| Feature | Missing | Missing % | Mechanism | Treatment |
|---------|---------|-----------|-----------|-----------|
| `province` | 919 | 7.8% | MCAR — uniform across classes and years | Mode imputation by region, or "Unknown" category |
| `roe` | 45 | 0.4% | **MNAR** — 100% Class D, caused by negative equity | Create `equity_negative` flag, then sentinel value or native NaN |
| `leverage` | 45 | 0.4% | **MNAR** — same 45 rows as ROE | Same treatment as ROE |

### ROE & Leverage: Informative Missingness

The 45 rows with null ROE are not a data quality issue.
ROE = Net Profit / Equity is mathematically undefined when equity < 0.
All 45 cases have equity < 0, and all 45 are Class D.

This is the most extreme form of financial distress in the dataset:
accumulated losses have fully eroded the equity base — the stage
immediately preceding formal bankruptcy.

**Modeling implications:**
- **Never impute with global median** — semantically incorrect,
  destroys the signal
- **Engineer `equity_negative` binary flag (0/1)** before any
  imputation — this feature alone perfectly identifies these 45 cases
- **For residual NaN handling:** use a sentinel value at the extreme
  left tail (1st percentile of ROE) or leave as NaN for native
  tree-based handling

### Province: MCAR — No Action Required

Missing rate is uniform across all classes (7.2%–7.9%) and all years
(228–232 per year). No correlation with the target.
→ Impute with regional mode or encode as "Unknown" category.

---

## 11. Descriptive Statistics — Structural Findings

### 11.1 Operating Cost Ratio — Engineer Explicitly

`op_cost_ratio = production_costs / production_value` is not present
in the raw dataset but shows a perfectly monotonic gradient:

| Class | Median op_cost_ratio |
|-------|----------------------|
| A | 0.902 |
| B | 0.917 |
| C | 0.933 |
| D | 0.996 |

Class D operates at near-breakeven — no margin to absorb debt service,
depreciation, or any external shock. Kruskal-Wallis p≈0.

→ Add as an explicit engineered feature. Highest discriminative power
  on the ambiguous C→D boundary.

### 11.2 Log Transform — Required for Balance Sheet Variables

Raw skewness 17–23 on all balance sheet variables. Log-transform
reduces skewness to 0.36–0.47 (confirmed log-normal distribution).

→ Apply `log(x)` to: `total_assets`, `production_value`, `total_debt`,
  `current_assets`, `total_fixed_assets`.
→ For `shareholders_equity`: log only when > 0, NaN otherwise.
  **Do not use `log(|x|)`** — it masks the negative-equity signal which
  is the strongest distress indicator in the dataset.

### 11.3 Multicollinearity Note

Profitability features (ROE, ROI, profit_margin) and debt features
(leverage, DTA) are internally correlated. Tree-based models handle
this natively via random feature subsampling. If testing a linear
baseline (logistic regression), apply VIF-based feature selection
before fitting — correlated inputs will produce unstable coefficients.

---

## 12. Key Findings — Business Insights & Modeling Implications

### 12.1 Terminal Distress is Deterministic, Not Probabilistic

50 rows satisfy at least one hard condition (equity < 0, DTA > 1,
leverage > 50). All 50 are Class D — no exceptions.
The three conditions are algebraically identical: equity collapse
causes all three simultaneously.

In the Italian corporate registry, negative shareholders' equity signals
accumulated losses that have fully eroded the capital base — the stage
immediately preceding formal insolvency proceedings.

**→ Pre-classify these 50 rows with a deterministic rule before training.**
**→ The model's real challenge is the ~1,000 ambiguous Class D rows.**
**→ These 50 "free" D predictions inflate recall metrics — report**
**separately when evaluating model performance on Class D.**

```python
# Apply on raw values, before any winsorization
df["is_terminal_distress"] = (
    (df["shareholders_equity"] < 0) |
    (df["debt_to_assets"] > 1)      |
    (df["leverage"] > 50)
).astype(int)
```

---

### 12.2 Financial Distress is a Process, Not an Event

83 companies that disappear from the registry show a clear trajectory:
- **All prior years:** mixed B/C/D distribution
- **Final observed year:** 100% Class D — no exceptions
- **Statistical confirmation:** χ²=681.11, p<0.0001 (MNAR)

Failure to file financial statements in Italy is not an administrative
oversight — it is the operational signature of liquidation or bankruptcy
proceedings in progress.

**→ Engineer trajectory features to capture progressive deterioration:**
- `n_years_in_panel` — consecutive filing count up to year t
- `Δroe`, `Δlev`, `Δroi` — year-over-year delta features
- `class_prev` — lagged target class at t-1, joined by
  `company_id + fiscal_year` before the CV split (not inside folds)

**→ Never drop early-exit rows — they contain the most informative**
**distress patterns in the entire dataset.**

---

### 12.3 Perfect Ordinal Gradient Across All Features

All 8 ratio features show perfectly monotonic medians A→B→C→D
(Kruskal-Wallis p≈0 for all). There are no inversions.

| Feature | Class A | Class D | Range | Key Threshold |
|---|---|---|---|---|
| ROE | 0.270 | −0.105 | 0.375 | < 0 → D signal |
| ROI | 0.158 | 0.007 | 0.151 | < 0.05 → C/D signal |
| Profit Margin | 0.087 | −0.015 | 0.102 | < 0 → D signal |
| Leverage | 0.901 | 5.517 | 4.616 | > 3.0 → C/D signal |
| Debt-to-Assets | 0.474 | 0.851 | 0.377 | > 0.75 → D signal |
| Current Ratio | 2.482 | 1.364 | 1.118 | < 1.0 → stress |
| Quick Ratio | 1.489 | 0.818 | 0.671 | < 1.0 → D territory |
| Op. Cost Ratio | 0.902 | 0.996 | 0.094 | > 0.95 → D signal |

The target has a **latent ordinal structure** (A < B < C < D in risk).
→ **Test OrdinalClassifier as a baseline** alongside LightGBM multiclass.

---

### 12.4 The Hard Classification Boundary is C↔D, Not A↔D

A vs D separation is near-trivial — ratio distributions are far apart.
The real challenge is **C vs D**: IQR overlap is substantial on every
feature (visible in boxplots for leverage and debt-to-assets).
Most misclassification errors will occur on this boundary.

→ Use **asymmetric sample weights** to penalize C→D errors more heavily
  than D→C (misclassifying a distressed company as moderate risk is a
  worse business outcome than the reverse).

```python
# Starting values — tune D weight in [2.0, 3.0, 5.0] via CV
sample_weights = train_df[TARGET].map(
    {"A": 1.0, "B": 1.0, "C": 1.5, "D": 3.0}
)
```

---

### 12.5 Sector Signal Lives in Relative Performance, Not Absolute Values

ATECO sector alone has near-zero predictive power (MI=0.0096).
The real signal emerges from **peer-relative features**:

```python
# Compute on TRAIN SET ONLY — apply to test to avoid leakage
sector_medians = train_df.groupby("sector_name")[RATIO_COLS].median()
for col in RATIO_COLS:
    train_df[f"{col}_vs_sector"] = (
        train_df[col] - train_df["sector_name"].map(sector_medians[col])
    )
```

A company with ROE=5% in Real Estate (sector median=0%) is structurally
different from one with ROE=5% in IT Services (sector median=12%).
The absolute value is identical; the peer-relative signal is opposite.

Notable structural exceptions where sector itself carries signal:
- **Real Estate:** A=0.0% — structurally incapable of Excellent rating
- **Construction:** C=29–30% — chronic margin compression, high debt
- **IT Services:** D=6.4% — knowledge-based resilience, lowest distress rate

---

### 12.6 Feature Engineering Priority Stack

Ordered by expected impact on Weighted F1:

| Priority | Feature | Type | Note |
|---|---|---|---|
| 1 | `is_terminal_distress` | Binary flag | Deterministic — 50 D rows guaranteed |
| 2 | `op_cost_ratio` | Engineered ratio | Not in raw dataset — strong C/D signal |
| 3 | `Δroe`, `Δlev`, `Δroi` | YoY delta | Captures deterioration trajectory |
| 4 | `n_years_in_panel` | Count | Proxy for stability and survival |
| 5 | `roe_vs_sector_median` | Relative ratio | Peer-relative performance signal |
| 6 | `log(assets)`, `log(debt)` | Log transform | Reduces skewness from 17–23 to <0.5 |

**Critical pipeline order — sequence matters:**
1. Flag `is_terminal_distress` on **raw** values
2. Engineer `op_cost_ratio`, lag/delta features, `n_years_in_panel`
3. Compute sector medians on **train fold only** (inside CV loop)
4. Apply log transforms to balance sheet variables
5. Winsorize ROE `[-0.325, 1.098]` and leverage `[0.828, 8.166]`
6. Encode categoricals (`region`, `ateco_sector`)


### 13.5 Feature Selection Strategy — Permutation Importance over Manual Dropping

Correlation-based manual dropping is suboptimal for tree-based models.
Correlated features can still contribute complementary splits in
different regions of the feature space.

**Exceptions — safe to drop manually (r=+1.000, algebraic duplicates):**
- `quick_ratio` → identical to `current_ratio` in this dataset
- `debt_to_assets` → algebraic function of `leverage`
- `production_costs` → fixed ratio of `production_value` (derive `op_cost_ratio` instead)

**For all other correlated features:** use permutation importance
after initial model training. Let the model decide which of `roe` vs
`roi`, or `total_debt` vs `short_term_debt`, carries more signal.
Drop features with permutation importance ≈ 0 across 10 repeats.

For any linear baseline (logistic regression): apply iterative
VIF pruning (threshold VIF > 10) before fitting.

§9 — Sector & Geography
- D-rate range across sectors: 4.1pp (6.4–10.5%) → near-zero standalone signal
- years_in_business: range 3 yrs across classes → drop or keep with near-zero importance
- ROI/ROE vary 2x across sectors → peer-relative features are mandatory
- Sector signal lives in ROI/ROE relative to peers, not in the sector dummy itself
