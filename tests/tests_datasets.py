import pandas as pd
import pytest
from pathlib import Path

# Configura i percorsi dinamicamente
ROOT = Path.cwd()
DATA_DIR = ROOT / "data" / "processed"

@pytest.fixture(scope="module")
def train_data():
    """Loads a small sample of the training data for testing."""
    file_path = DATA_DIR / 'train_data.csv'
    assert file_path.exists(), f"File not found: {file_path}"
    return pd.read_csv(file_path, nrows=100)

@pytest.fixture(scope="module")
def test_data():
    """Loads a small sample of the test data for testing."""
    file_path = DATA_DIR / 'test_features.csv'
    assert file_path.exists(), f"File not found: {file_path}"
    return pd.read_csv(file_path, nrows=100)

def test_data_is_not_empty(train_data, test_data):
    """Test that the loaded datasets actually contain data."""
    assert len(train_data) > 0, "Training dataset is completely empty."
    assert len(test_data) > 0, "Test dataset is completely empty."

def test_fiscal_year_ranges(train_data, test_data):
    """
    Test that the fiscal years are correctly split:
    - Train data should only contain years from 2018 to 2021.
    - Test data should only contain years 2022 and 2023.
    """
    
    # 1. Define the expected years for both sets
    expected_train_years = {2018, 2019, 2020, 2021}
    expected_test_years = {2022, 2023}
    
    # 2. Get the actual unique years present in the datasets
    actual_train_years = set(train_data['fiscal_year'].dropna().unique())
    actual_test_years = set(test_data['fiscal_year'].dropna().unique())
    
    # 3. Find any years that shouldn't be there
    invalid_train_years = actual_train_years - expected_train_years
    invalid_test_years = actual_test_years - expected_test_years
    
    # 4. Assert that there are no invalid years
    assert not invalid_train_years, \
        f"Data leakage or error: Found unexpected years in train data: {invalid_train_years}"
        
    assert not invalid_test_years, \
        f"Data leakage or error: Found unexpected years in test data: {invalid_test_years}"

def test_target_column_exists_in_train(train_data):
    """Test that the required target for Task 2 exists in the training set."""
    target_col = 'financial_health_class'
    assert target_col in train_data.columns, \
        f"Missing target column '{target_col}' in training data."

def test_target_column_is_missing_in_test(test_data):
    """Test that the target column is NOT present in the test features."""
    target_col = 'financial_health_class'
    assert target_col not in test_data.columns, \
        "Test dataset should not contain the target column."

def test_feature_columns_match(train_data, test_data):
    """Test that train and test datasets have the exact same feature columns."""
    # Remove task targets from train columns to get pure features
    targets = {'financial_health_class', 'bankruptcy_next_year', 'revenue_change'}
    train_features = set(train_data.columns) - targets
    test_features = set(test_data.columns)
    
    missing_in_test = train_features - test_features
    extra_in_test = test_features - train_features
    
    assert not missing_in_test, f"Test data is missing features: {missing_in_test}"
    assert not extra_in_test, f"Test data has unexpected extra features: {extra_in_test}"

def test_company_id_is_unique(train_data, test_data):
    """Test that 'company_id' combined with 'fiscal_year' creates a unique row."""
    # Assuming the dataset is a panel (company over different years)
    train_duplicates = train_data.duplicated(subset=['company_id', 'fiscal_year']).sum()
    test_duplicates = test_data.duplicated(subset=['company_id', 'fiscal_year']).sum()
    
    assert train_duplicates == 0, f"Found {train_duplicates} duplicate company/year pairs in train data."
    assert test_duplicates == 0, f"Found {test_duplicates} duplicate company/year pairs in test data."

def test_target_has_valid_classes(train_data):
    """Test that the multiclass target only contains expected values."""
    expected_classes = {'A', 'B', 'C', 'D'} # Assuming these are the standard 4 classes
    actual_classes = set(train_data['financial_health_class'].dropna().unique())
    
    # Check if the actual classes are a subset of the expected classes
    invalid_classes = actual_classes - expected_classes
    assert not invalid_classes, f"Found invalid target classes in data: {invalid_classes}"