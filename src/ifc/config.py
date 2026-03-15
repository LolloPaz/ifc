from pathlib import Path

#Paths 
ROOT = Path(__file__).parents[2]
DATA_DIR = ROOT / "data" / "processed"
OUTPUT_PATH = ROOT / "figure"
TRAIN_FILE = DATA_DIR / "train_data.csv"
TEST_FILE = DATA_DIR / "test_features.csv"
NOTEBOOKS_DIR = ROOT / "notebooks"
FIGURES_DIR = NOTEBOOKS_DIR / "figures"
OUTPUT_PATH = FIGURES_DIR  # alias per compatibilità

#Challenge 2: Financial Health Classification
TARGET = "financial_health_class"
CLASSES = ["A", "B", "C", "D"]  # ordinal: A (best) → D (worst)
CLASS_ORDER = {c: i for i, c in enumerate(CLASSES)}  # {"A":0, "B":1, "C":2, "D":3}

# Temporal split
TRAIN_YEARS = [2018, 2019, 2020, 2021]
TEST_YEARS = [2022, 2023]

# Columns
ID_COLS = ["company_id", "fiscal_year"]

CATEGORICAL_COLS = ["ateco_sector", "province", "region", "legal_form"]

NUMERICAL_COLS = [
    # Balance sheet
    "total_fixed_assets", "current_assets", "total_assets",
    "shareholders_equity", "total_debt", "short_term_debt", "long_term_debt",
    # Income statement
    "production_value", "production_costs", "operating_income",
    "financial_income", "financial_expenses", "net_profit_loss",
    # Ratios
    "roe", "roi", "profit_margin", "leverage",
    "debt_to_assets", "current_ratio", "quick_ratio",
    # Other
    "years_in_business",
]

# Targets to drop — avoid leakage from other challenge targets
DROP_COLS = ["bankruptcy_next_year", "revenue_change"]

# Reproducibility 
SEED = 54

