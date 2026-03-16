Riscrivo tutto in modo **compliant con l’EDA_findings** (niente robe non supportate, e usando la stessa terminologia).

***

## 📊 SLIDE 3 – Project Setup & Challenge Design

### Visual Content

**Title:** Project Setup & Challenge Design  

**Intro paragraph (opzionale):**  
Modern ML infrastructure with strict temporal constraints

**Bullets:**
- **Tech stack:** uv package manager + pyproject.toml (code on GitHub)
- **Challenge:** Classify financial health (A/B/C/D) at year t using features from year t [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/154764613/d47ff9e5-3f77-417a-9f7c-8e8386781cc9/EDA_findings.md)
- Each \((X_t, y_t)\) pair indexed by (company_id, fiscal_year) treated as independent observation [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/154764613/d47ff9e5-3f77-417a-9f7c-8e8386781cc9/EDA_findings.md)
- **Critical constraint:** Prevent temporal leakage (train on past, validate on future only) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/154764613/d47ff9e5-3f77-417a-9f7c-8e8386781cc9/EDA_findings.md)

***

### Speaker Script (≈ 60–65 sec)

**Opening (10 sec):**  
"Let me start from how we set up the project and how we formalised the challenge."

**Task definition (10–15 sec):**  
"This is a multiclass classification problem on four financial health categories A, B, C, D. We follow the EDA definition: the target is the current health state \(y_t\), and features are the balance sheet snapshot at year \(t\), optionally enriched with information from \(t-1\)." [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/154764613/d47ff9e5-3f77-417a-9f7c-8e8386781cc9/EDA_findings.md)

**Data structure (10–15 sec):**  
"We treat each company–year as an independent observation. A row for company 42 in 2019 is a separate training instance from the same company in 2020. Formally, each pair \((X_t, y_t)\) indexed by (company_id, fiscal_year) is one supervised example." [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/154764613/d47ff9e5-3f77-417a-9f7c-8e8386781cc9/EDA_findings.md)

**Temporal constraint (15 sec):**  
"A key point from the EDA is that this is not a forecasting task \(y_{t+1}\) from \(X_t\), but a diagnosis of \(y_t\) from \(X_t\). To avoid temporal leakage, we always train on earlier years and validate on later years—no random K-fold, only time-based splits." [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/154764613/d47ff9e5-3f77-417a-9f7c-8e8386781cc9/EDA_findings.md)

**Tech stack (10 sec):**  
"On the engineering side we use uv with a pyproject.toml to manage dependencies reproducibly, and we keep all code in a GitHub repository, so the full pipeline—from EDA to model training—can be reproduced end-to-end."

**Transition (5 sec):**  
"With the challenge framed, we can look at what the class distribution and imbalance mean for our evaluation strategy."

***

## 📊 SLIDE 4 – Critical Finding: Severe Class Imbalance

### Visual Content

**Title:** Critical Finding: Severe Class Imbalance  

**Intro paragraph:**  
Class B dominates at 59.3%, while A and D are minority classes [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/154764613/d47ff9e5-3f77-417a-9f7c-8e8386781cc9/EDA_findings.md)

**Bullets:**
- Accuracy is misleading — a trivial model that always predicts “B” reaches ~59% [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/154764613/d47ff9e5-3f77-417a-9f7c-8e8386781cc9/EDA_findings.md)
- **Primary metric:** Weighted F1-Score [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/154764613/d47ff9e5-3f77-417a-9f7c-8e8386781cc9/EDA_findings.md)
- **Strategy:** Apply `class_weight="balanced"` in all classifiers [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/154764613/d47ff9e5-3f77-417a-9f7c-8e8386781cc9/EDA_findings.md)

***

### Speaker Script (≈ 30–35 sec)

**The problem (10 sec):**  
"The target is strongly imbalanced: Class B alone is 59.3% of the dataset, while A and D are around 8–9% each. A trivial model that always predicts B already reaches about 59% accuracy without learning anything." [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/154764613/d47ff9e5-3f77-417a-9f7c-8e8386781cc9/EDA_findings.md)

**The solution – metric (10 sec):**  
"For this reason accuracy is explicitly discarded in the EDA. We use Weighted F1 as primary metric, so that errors on minority classes A and D are not washed out by the majority class." [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/154764613/d47ff9e5-3f77-417a-9f7c-8e8386781cc9/EDA_findings.md)

**The solution – strategy (10 sec):**  
"Concretely, all our classifiers will use `class_weight='balanced'`. This follows the master table of preprocessing decisions and is meant to prevent the model from collapsing to 'always B'." [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/154764613/d47ff9e5-3f77-417a-9f7c-8e8386781cc9/EDA_findings.md)

**Transition (5 sec):**  
"Once the imbalance is under control, the EDA highlights a much more subtle pattern: companies that disappear from the registry."

***

## 📊 SLIDE 5 – Most Critical Insight: Early Exit Pattern

### Visual Content

**Title:** Most Critical Insight: Early Exit Pattern  

**Intro paragraph:**  
83 companies disappear from the registry between train (2018–2021) and test (2022–2023) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/154764613/d47ff9e5-3f77-417a-9f7c-8e8386781cc9/EDA_findings.md)

**Bullets:**
- **Final observed year:** 100% Class D (no exceptions) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/154764613/d47ff9e5-3f77-417a-9f7c-8e8386781cc9/EDA_findings.md)
- χ² = 681.11, p < 0.0001 — missingness is definitively not random (MNAR) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/154764613/d47ff9e5-3f77-417a-9f7c-8e8386781cc9/EDA_findings.md)
- Business interpretation: Failure to file statements is a proxy for liquidation or severe distress [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/154764613/d47ff9e5-3f77-417a-9f7c-8e8386781cc9/EDA_findings.md)

***

### Speaker Script (≈ 30 sec)

**Pattern (10 sec):**  
"The most critical domain insight concerns companies with incomplete panel histories. By cross-referencing train 2018–2021 and test 2022–2023, we identified 83 companies that never reappear in the test set—true early exits from the business registry." [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/154764613/d47ff9e5-3f77-417a-9f7c-8e8386781cc9/EDA_findings.md)

**Statistical proof (10 sec):**  
"In their final observed year, 100% of these early exits are Class D, with no exceptions. A chi-squared test gives χ² ≈ 681 and p < 0.0001, so the missingness pattern is definitively not random—it is MNAR." [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/154764613/d47ff9e5-3f77-417a-9f7c-8e8386781cc9/EDA_findings.md)

**Business meaning (10 sec):**  
"In the Italian corporate registry, failing to file financial statements is therefore interpreted as an operational signature of liquidation or severe distress, not a benign data quality issue." [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/154764613/d47ff9e5-3f77-417a-9f7c-8e8386781cc9/EDA_findings.md)

***

## 📊 SLIDE 6 – Early Exit — Modeling Implications

### Visual Content

**Title:** Early Exit — Modeling Implications  

**Bullets:**
- **Never drop these rows** — early exits contain the most informative distress patterns [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/154764613/d47ff9e5-3f77-417a-9f7c-8e8386781cc9/EDA_findings.md)
- Engineer trajectory features: n_years_in_panel, YoY deltas (ROE, leverage, ROI) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/154764613/d47ff9e5-3f77-417a-9f7c-8e8386781cc9/EDA_findings.md)
- **Open question:** 2018 rows — follow EDA suggestion (use only as lag source) vs. keeping them with `is_first_year_obs` + NaN handling
- Use lagged target (class_prev) where available — joined before CV split, never inside folds [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/154764613/d47ff9e5-3f77-417a-9f7c-8e8386781cc9/EDA_findings.md)

***

### Speaker Script (≈ 35 sec)

**Strategic decision (10 sec):**  
"The EDA is very clear: we must never drop these early-exit rows. They contain the most informative patterns for detecting fatal financial deterioration, exactly the behaviour we want the model to learn." [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/154764613/d47ff9e5-3f77-417a-9f7c-8e8386781cc9/EDA_findings.md)

**Feature engineering (15 sec):**  
"Instead, we engineer trajectory features: the number of consecutive years a company has filed (`n_years_in_panel`), year-over-year deltas in profitability and leverage, and a lagged target `class_prev` at \(t-1\). These are all explicitly recommended to capture that distress is a process, not an event." [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/154764613/d47ff9e5-3f77-417a-9f7c-8e8386781cc9/EDA_findings.md)

**Open question (5 sec):**  
"One open point we will resolve empirically is what to do with 2018 rows: the EDA suggests using 2018 only as a source year for lags, but we are also considering keeping them with a `is_first_year_obs` flag and relying on tree-based models to handle the resulting NaNs."

**Transition (5 sec):**  
"Beyond early exits, we also found a segment of companies where distress is completely deterministic."

***

## 📊 SLIDE 7 – Terminal Distress is Deterministic

### Visual Content

**Title:** Terminal Distress is Deterministic  

**Intro paragraph:**  
50 companies meet hard distress conditions — all are Class D (100%) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/154764613/d47ff9e5-3f77-417a-9f7c-8e8386781cc9/EDA_findings.md)

**Bullets:**
- Equity ≤ 0, or Debt-to-Assets ≥ 1, or Leverage ≥ 50 (algebraically equivalent) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/154764613/d47ff9e5-3f77-417a-9f7c-8e8386781cc9/EDA_findings.md)
- Pre-classify with deterministic rule on raw values before any winsorization [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/154764613/d47ff9e5-3f77-417a-9f7c-8e8386781cc9/EDA_findings.md)
- Real challenge: ~1,000 “ambiguous” Class D rows remain for the model to learn [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/154764613/d47ff9e5-3f77-417a-9f7c-8e8386781cc9/EDA_findings.md)

***

### Speaker Script (≈ 30 sec)

**The pattern (10 sec):**  
"The EDA isolates 50 rows that satisfy at least one of three hard conditions: shareholders’ equity ≤ 0, debt-to-assets ≥ 1, or leverage ≥ 50. All three conditions are algebraically equivalent, and all 50 cases are Class D with no exceptions." [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/154764613/d47ff9e5-3f77-417a-9f7c-8e8386781cc9/EDA_findings.md)

**The strategy (10 sec):**  
"These represent terminal distress—capital fully eroded, just before formal insolvency. The EDA recommends flagging them with an `is_terminal_distress` binary feature and effectively pre-classifying them using a deterministic rule, applied on raw values before any winsorisation." [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/154764613/d47ff9e5-3f77-417a-9f7c-8e8386781cc9/EDA_findings.md)

**The real challenge (10 sec):**  
"The model’s real challenge then becomes the remaining ~1,000 Class D cases, which are much more ambiguous. These 50 'free' D predictions can inflate recall, so the EDA suggests reporting them separately when evaluating Class D performance." [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/154764613/d47ff9e5-3f77-417a-9f7c-8e8386781cc9/EDA_findings.md)

***

Ecco le **ultime 2 slide** riscritte in modo pienamente compliant con l’EDA_findings.

***

## 📊 SLIDE 8 – Perfect Ordinal Structure

### Visual Content

**Title:** Perfect Ordinal Structure  

**Intro paragraph:**  
All 8 key financial ratios show a perfectly monotonic gradient A → B → C → D [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/154764613/d47ff9e5-3f77-417a-9f7c-8e8386781cc9/EDA_findings.md)

**Bullets:**
- **ROE (median):** 0.270 (A) → −0.105 (D) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/154764613/d47ff9e5-3f77-417a-9f7c-8e8386781cc9/EDA_findings.md)
- **Leverage (median):** 0.901 (A) → 5.517 (D) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/154764613/d47ff9e5-3f77-417a-9f7c-8e8386781cc9/EDA_findings.md)
- **Op. Cost Ratio (median):** 0.902 (A) → 0.996 (D) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/154764613/d47ff9e5-3f77-417a-9f7c-8e8386781cc9/EDA_findings.md)
- Kruskal–Wallis p ≈ 0 for all 8 ratios, no inversions across classes [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/154764613/d47ff9e5-3f77-417a-9f7c-8e8386781cc9/EDA_findings.md)
- Suggests testing an OrdinalClassifier baseline alongside multiclass LightGBM [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/154764613/d47ff9e5-3f77-417a-9f7c-8e8386781cc9/EDA_findings.md)

***

### Speaker Script (≈ 25–30 sec)

**The finding (10–15 sec):**  
"When we look at medians by class, all eight main financial ratios exhibit a perfectly monotonic gradient from A to D. For example, median ROE goes from about 27% in Class A down to roughly –10.5% in Class D, leverage rises from around 0.9 to over 5.5, and the operating cost ratio increases from 0.902 to 0.996 as we move from A to D." [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/154764613/d47ff9e5-3f77-417a-9f7c-8e8386781cc9/EDA_findings.md)

**Statistical confirmation (5–10 sec):**  
"Kruskal–Wallis tests return p≈0 for all eight ratios, and the EDA explicitly notes that there are no inversions along the A–B–C–D axis. So the ordinal structure is not just conceptual, it is statistically clean." [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/154764613/d47ff9e5-3f77-417a-9f7c-8e8386781cc9/EDA_findings.md)

**Modeling implication (5 sec):**  
"On this basis the EDA recommends testing an OrdinalClassifier as a baseline, in addition to standard multiclass models like LightGBM, to exploit the natural risk ordering of the target." [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/154764613/d47ff9e5-3f77-417a-9f7c-8e8386781cc9/EDA_findings.md)

**Transition (5 sec):**  
"However, even with such a clean ordinal structure, one boundary remains notably difficult: the split between C and D."

***

## 📊 SLIDE 9 – The Hard Boundary: C vs D

### Visual Content

**Title:** The Hard Boundary: C vs D  

**Intro paragraph:**  
A vs D separation is near-trivial — most errors concentrate on the C ↔ D boundary [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/154764613/d47ff9e5-3f77-417a-9f7c-8e8386781cc9/EDA_findings.md)

**Bullets:**
- Substantial IQR overlap between Classes C and D on all key ratios (e.g. leverage, debt-to-assets) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/154764613/d47ff9e5-3f77-417a-9f7c-8e8386781cc9/EDA_findings.md)
- Most misclassification errors are expected on the C ↔ D boundary [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/154764613/d47ff9e5-3f77-417a-9f7c-8e8386781cc9/EDA_findings.md)
- Business view: misclassifying D as C is worse than C as D [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/154764613/d47ff9e5-3f77-417a-9f7c-8e8386781cc9/EDA_findings.md)
- **Strategy:** Asymmetric sample weights — penalize C→D errors more heavily than D→C [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/154764613/d47ff9e5-3f77-417a-9f7c-8e8386781cc9/EDA_findings.md)
- Starting weights (EDA suggestion): A=1.0, B=1.0, C=1.5, D=3.0, to be tuned via CV [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/154764613/d47ff9e5-3f77-417a-9f7c-8e8386781cc9/EDA_findings.md)

***

### Speaker Script (≈ 30 sec)

**Easy vs hard (10 sec):**  
"The EDA highlights that separating A from D is almost trivial: their ratio distributions are far apart. The real difficulty lies on the C versus D boundary, where interquartile ranges for variables like leverage and debt-to-assets heavily overlap." [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/154764613/d47ff9e5-3f77-417a-9f7c-8e8386781cc9/EDA_findings.md)

**Error concentration and business impact (10 sec):**  
"Most misclassification errors are expected exactly on C↔D. From a business perspective, confusing a truly distressed D company for a moderate-risk C is much more problematic than the opposite, because it means systematically underestimating risk." [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/154764613/d47ff9e5-3f77-417a-9f7c-8e8386781cc9/EDA_findings.md)

**Strategy (10 sec):**  
"To reflect this asymmetry, the EDA recommends using asymmetric sample weights in training. A suggested starting point is to weight A and B as 1.0, C as 1.5, and D as 3.0, and then tune these values via time-based cross-validation." [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/154764613/d47ff9e5-3f77-417a-9f7c-8e8386781cc9/EDA_findings.md)

***

