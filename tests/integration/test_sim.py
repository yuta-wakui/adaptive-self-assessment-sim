import os
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from adaptive_self_assessment.simulation.ws1 import run_ws1_simulation
from adaptive_self_assessment.simulation.ws2 import run_ws2_simulation

@pytest.mark.parametrize("ws1_path", [
    "data/sample/ws1/ws1_data_sample.csv",
])
def test_run_ws1_simulation(ws1_path):
    print("=== WS1 Simulation Test ===")

    # データの読み込み
    if not os.path.exists(ws1_path):
        pytest.skip(f"File not found: {ws1_path}")
    df_ws1 = pd.read_csv(ws1_path)

    # 訓練データとテストデータに分割（80%訓練、20%テスト）
    df_ws1_train, df_ws1_test = train_test_split(df_ws1, test_size=0.2, random_state=42,shuffle=True)
    print(f"Training data size: {len(df_ws1_train)}, Test data size: {len(df_ws1_test)}")
    
    # シミュレーションの実行
    sim_results = run_ws1_simulation(
        RC_THRESHOLD=0.80,
        RI_THRESHOLD=0.70,
        skill_name="sample",
        model_type="logistic_regression",
        train_df=df_ws1_train,
        test_df=df_ws1_test,
        logs_path="outputs/logs/test_logs/ws1_simulation_test_logs.csv"
    )

    # 結果の検証
    assert isinstance(sim_results, dict)
    assert 0<= sim_results["ruction_rate"] <=100

    expected_keys = {
        "skill",
        "model_type",
        "RC_THRESHOLD",
        "RI_THRESHOLD",
        "num_train",
        "num_test",
        "accuracy_over_threshold",
        "accuracy_all",
        "coverage_over_threshold",
        "total_questions",
        "avg_answered_questions",
        "avg_complemented_questions",
        "reduction_rate",
        "avg_response_time",
        "max_response_time",
        "min_response_time",
    }
    missing = expected_keys - set(sim_results.keys())
    assert not missing, f"Missing keys in simulation results: {missing}"
    print("WS1 simulation Results: ", sim_results)

@pytest.mark.parametrize("ws2_path", [
    "data/sample/ws2/ws2_data_sample.csv",
])
def test_run_ws2_simulation(ws2_path):
    print("=== WS2 Simulation Test ===")

    # データの読み込み
    if not os.path.exists(ws2_path):
        pytest.skip(f"File not found: {ws2_path}")
    df_ws2 = pd.read_csv(ws2_path)

    # 訓練データとテストデータに分割（80%訓練、20%テスト）
    df_ws2_train, df_ws2_test = train_test_split(df_ws2, test_size=0.2, random_state=42,shuffle=True)
    print(f"Training data size: {len(df_ws2_train)}, Test data size: {len(df_ws2_test)}")
    
    # シミュレーションの実行
    sim_results = run_ws2_simulation(
        RC_THRESHOLD=0.8,
        RI_THRESHOLD=0.7,
        skill_name="sample",
        model_type="logistic_regression",
        train_df=df_ws2_train,
        test_df=df_ws2_test,
        logs_path="outputs/logs/test_logs/ws2_simulation_test_logs.csv"
    )

    # 結果の検証
    assert isinstance(sim_results, dict)
    assert 0<= sim_results["ruction_rate"] <=100

    expected_keys = {
        "skill",
        "model_type",
        "RC_THRESHOLD",
        "RI_THRESHOLD",
        "num_train",
        "num_test",
        "accuracy_over_threshold",
        "accuracy_all",
        "coverage_over_threshold",
        "total_questions",
        "avg_answered_questions",
        "avg_complemented_questions",
        "reduction_rate",
        "avg_response_time",
        "max_response_time",
        "min_response_time",
    }
    missing = expected_keys - set(sim_results.keys())
    assert not missing, f"Missing keys in simulation results: {missing}" 
    print("WS2 simulation Results: ", sim_results)
