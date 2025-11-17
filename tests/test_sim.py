import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import pytest

from scripts.simulation.run_ws1_sim import run_ws1_simulation
from scripts.simulation.run_ws2_sim import run_ws2_simulation

@pytest.mark.parametrize("ws1_path", [
    "data/processed/ws1_synthetic_240531_processed/1_syntheticdata_informationliteracy.csv",
])
def test_run_ws1_simulation(ws1_path):
    print("=== WS1 Simulation Test ===")
    # データの読み込み
    if not os.path.exists(ws1_path):
        pytest.skip(f"File not found: {ws1_path}")
    df_ws1 = pd.read_csv(ws1_path)

    # 訓練データとテストデータに分割（80%訓練、20%テスト）
    df_ws1_train, df_ws1_test = train_test_split(df_ws1, test_size=0.2, random_state=42)
    print(f"Training data size: {len(df_ws1_train)}, Test data size: {len(df_ws1_test)}")
    
    # シミュレーションの実行
    sim_results = run_ws1_simulation(
        RC_THRESHOLD=0.85,
        RI_THRESHOLD=0.75,
        skill_name="informationliteracy",
        model_type="logistic_regression",
        train_df=df_ws1_train,
        test_df=df_ws1_test,
        logs_path="outputs/logs/tests/ws1_simulation_test_logs.csv"
    )

    # 結果の検証
    assert isinstance(sim_results, dict)
    print("Simulation Result:", sim_results)

@pytest.mark.parametrize("ws2_path", [
    "data/processed/w2-synthetic_20250326_1300_processed/ws2_1_information_1300_processed.csv",
])
def test_run_ws2_simulation(ws2_path):
    print("=== WS2 Simulation Test ===")
    # データの読み込み
    if not os.path.exists(ws2_path):
        pytest.skip(f"File not found: {ws2_path}")
    df_ws2 = pd.read_csv(ws2_path)

    # 訓練データとテストデータに分割（80%訓練、20%テスト）
    df_ws2_train, df_ws2_test = train_test_split(df_ws2, test_size=0.2, random_state=42)
    print(f"Training data size: {len(df_ws2_train)}, Test data size: {len(df_ws2_test)}")
    
    # シミュレーションの実行
    sim_results = run_ws2_simulation(
        RC_THRESHOLD=0.85,
        RI_THRESHOLD=0.75,
        skill_name="information",
        model_type="logistic_regression",
        train_df=df_ws2_train,
        test_df=df_ws2_test,
        logs_path="outputs/logs/tests/ws2_simulation_test_logs.csv"
    )

    # 結果の検証
    assert isinstance(sim_results, dict)
    print("Simulation Result:", sim_results)