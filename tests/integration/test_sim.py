import os
import yaml
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from adaptive_self_assessment.simulation.ws1 import run_ws1_simulation
from adaptive_self_assessment.simulation.ws2 import run_ws2_simulation

def _load_config(config_path = "configs/config.yaml"):
    """
    load configuration from a YAML file
    
    Parameters:
    ----------
    config_path: str
        path to the configuration file (default: "configs/config.yaml")
    
    Returns:
    -------
    cfg: Dict[str, Any]
        configuration dictionary loaded from the file
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    return cfg

@pytest.mark.parametrize(
    ("ws1_path", "cfg_path"),
    [
        ("data/sample/ws1/ws1_data_sample.csv", "configs/config.yaml"),
    ]
)
def test_run_ws1_simulation(ws1_path, cfg_path):
    print("=== WS1 Simulation Test ===")

    # load data
    if not os.path.exists(ws1_path):
        pytest.skip(f"File not found: {ws1_path}")
    df_ws1 = pd.read_csv(ws1_path)

    # load config
    if not os.path.exists(cfg_path):
        pytest.skip(f"Config file not found: {cfg_path}")
    cfg = _load_config(cfg_path)

    # split into train and test sets (80% train, 20% test)
    df_ws1_train, df_ws1_test = train_test_split(df_ws1, test_size=0.2, random_state=42,shuffle=True)
    print(f"Training data size: {len(df_ws1_train)}, Test data size: {len(df_ws1_test)}")


    # run simulation
    sim_results, logs_df = run_ws1_simulation(
        train_df=df_ws1_train,
        test_df=df_ws1_test,
        cfg=cfg,
    )

    # validate results
    assert isinstance(sim_results, dict)
    assert isinstance(logs_df, pd.DataFrame)
    assert 0.0 <= float(sim_results["reduction_rate"]) <= 100.0


    expected_keys = {
        "skill","model_type","RC_THRESHOLD","RI_THRESHOLD","num_train","num_test",
        "accuracy_over_threshold","accuracy_all",
        "f1_macro_over_threshold","f1_macro_all",
        "coverage_over_threshold","total_questions",
        "avg_answered_questions","avg_complemented_questions",
        "avg_complement_accuracy", 
        "reduction_rate",
        "avg_response_time","max_response_time","min_response_time",
    }

    missing = expected_keys - set(sim_results.keys())
    assert not missing, f"Missing keys in simulation results: {missing}"
    print("WS1 simulation Results: ", sim_results)

@pytest.mark.parametrize(
    ("ws2_path", "cfg_path"),
    [
        ("data/sample/ws2/ws2_data_sample.csv", "configs/config.yaml"),
    ]
)
def test_run_ws2_simulation(ws2_path, cfg_path):
    print("=== WS2 Simulation Test ===")

    # load data
    if not os.path.exists(ws2_path):
        pytest.skip(f"File not found: {ws2_path}")
    df_ws2 = pd.read_csv(ws2_path)

    # load config
    if not os.path.exists(cfg_path):
        pytest.skip(f"Config not found: {cfg_path}")
    cfg = _load_config(cfg_path)

    # split into train and test sets (80% train, 20% test)
    df_ws2_train, df_ws2_test = train_test_split(df_ws2, test_size=0.2, random_state=42,shuffle=True)
    print(f"Training data size: {len(df_ws2_train)}, Test data size: {len(df_ws2_test)}")
    
    # run simulation
    sim_results, logs_df = run_ws2_simulation(
        train_df=df_ws2_train,
        test_df=df_ws2_test,
        cfg=cfg,
    )

    # validate results
    assert isinstance(sim_results, dict)
    assert isinstance(logs_df, pd.DataFrame)
    assert 0.0 <= float(sim_results["reduction_rate"]) <= 100.0


    expected_keys = {
        "skill","model_type","RC_THRESHOLD","RI_THRESHOLD","num_train","num_test",
        "accuracy_over_threshold","accuracy_all",
        "f1_macro_over_threshold","f1_macro_all",
        "coverage_over_threshold","total_questions",
        "avg_answered_questions","avg_complemented_questions",
        "avg_complement_accuracy", 
        "reduction_rate",
        "avg_response_time","max_response_time","min_response_time",
    }

    missing = expected_keys - set(sim_results.keys())
    assert not missing, f"Missing keys in simulation results: {missing}" 
    print("WS2 simulation Results: ", sim_results)