Here's the EDA roadmap for Challenge 2, structured as notebook sections:

## EDA Roadmap — `01_eda.ipynb`

### 1. Dataset Structure
- Shape of train and test sets
- Column types and non-null counts
- Temporal split sanity check (train 2018–2021, test 2022–2023)
- Unique companies and rows per company

***

### 2. Target Variable Analysis
- Class counts and percentages (A/B/C/D)
- Class distribution per `fiscal_year` — detect COVID drift in 2020–2021
- Class distribution by `ateco_sector`, `legal_form`, `region`

***

### 3. Missing Values
- Missing count and % per column
- Structural vs random: `roe`/`leverage` missing when equity = 0
- `province` missingness — correlated with class?
- Missingness heatmap by `fiscal_year`

***

### 4. Descriptive Statistics
- Summary stats for all numerical columns
- Log-scale histograms for balance sheet items (heavy right skew expected)
- Check impossible values: negative revenue, `debt_to_assets > 1`

***

### 5. Outlier Detection
- 1st/99th percentile bounds for all numerical columns
- Accounting identity check: `total_assets = equity + total_debt`
- Flag extreme `leverage` values (> 50)

***

### 6. Feature Separability by Class
- Violin plots of key ratios by class: `roe`, `roi`, `current_ratio`, `debt_to_assets`, `profit_margin`, `leverage`
- **Goal**: verify ordinal signal A > B > C > D exists in the raw features

***

### 7. Correlation Analysis
- Heatmap of all numerical features
- Identify highly correlated pairs (> 0.85) — informs feature selection later

***

### 8. Panel Completeness
- Years per company distribution
- % of companies with full 4-year history
- **Goal**: confirm lag/YoY features are safe to engineer

***

### 9. Sector & Geography
- Class distribution by top 10 ATECO sectors
- `years_in_business` by class
- Average `roi`/`roe` by sector

***

### End of each section → 1 markdown cell with:
1. Key finding
2. Preprocessing decision it implies
