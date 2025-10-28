import os
import pandas as pd
import pytest
from adaptive_self_assessment.spec import SPEC_WS1, SPEC_WS2, get_spec_cols

@pytest.mark.parametrize("ws1_path", [
    "data/processed/ws1_synthetic_240531_processed/1_syntheticdata_informationliteracy.csv",
])
def test_get_spec_cols_ws1(ws1_path):
    if not os.path.exists(ws1_path):
        pytest.skip(f"File not found: {ws1_path}")
    df = pd.read_csv(ws1_path)
    pra_col, pca_cols, ca_cols, ra_col, ignore_cols = get_spec_cols(df, SPEC_WS1)
    
    # 内容確認
    print("=== SPEC_WS1 Columns ===")
    print(f"Past Overall Assessment Column: {pra_col}")
    print(f"Past Checklist Columns: {pca_cols}")
    print(f"Current Checklist Columns: {ca_cols}")  
    print(f"Current Overall Assessment Column: {ra_col}")
    print(f"Ignore Columns: {ignore_cols}")


    # アサーション例
    assert ca_cols is not None
    assert isinstance(ca_cols, list)
    assert len(ca_cols) > 0

@pytest.mark.parametrize("ws2_path", [
    "data/processed/w2-synthetic_20250326_1300_processed/ws2_1_information_1300_processed.csv",
])

def test_get_spec_cols_ws2(ws2_path):
    if not os.path.exists(ws2_path):
        pytest.skip(f"File not found: {ws2_path}")
    df = pd.read_csv(ws2_path)
    pra_col, pca_cols, ca_cols, ra_col, ignore_cols = get_spec_cols(df, SPEC_WS2)
    
    # 内容確認
    print("=== SPEC_WS2 Columns ===")
    print(f"Past Overall Assessment Column: {pra_col}")
    print(f"Past Checklist Columns: {pca_cols}")
    print(f"Current Checklist Columns: {ca_cols}")  
    print(f"Current Overall Assessment Column: {ra_col}")
    print(f"Ignore Columns: {ignore_cols}")


    # アサーション例
    assert pra_col is not None
    assert isinstance(pra_col, str)
    assert pca_cols is not None
    assert isinstance(pca_cols, list)
    assert ca_cols is not None
    assert isinstance(ca_cols, list)
