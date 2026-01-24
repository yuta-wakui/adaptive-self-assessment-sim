import os
import pandas as pd
import numpy as np
import pytest
import yaml

from adaptive_self_assessment.components.selector import select_question, set_selector_seed
from adaptive_self_assessment.components.predictor import (
    predict_item_ws1, predict_item_ws2,
    predict_overall_ws1, predict_overall_ws2
)
from adaptive_self_assessment.components.model_store import ModelStore

def _load_config(config_path: str = "configs/config.yaml"):
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
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

@pytest.mark.parametrize("ws1_path", [
    "data/sample/ws1/ws1_data_sample.csv",
])
def test_predict_item_ws1(ws1_path):
    # load data
    if not os.path.exists(ws1_path):
        pytest.skip(f"File not found: {ws1_path}")
    df_w1 = pd.read_csv(ws1_path)

    # get column names
    cfg = _load_config()
    data_cfg = cfg.get("data", {})
    ca_cols_ws1 = data_cfg.get("ws1", {}).get("item_cols", [])
    if not ca_cols_ws1:
        raise ValueError("item_cols must be specified in config for WS1.")

    np.random.seed(42)
    store = ModelStore()
    fold = 0

    used_id = int(np.random.choice(df_w1.index))
    df_w1_train = df_w1[df_w1.index != used_id]

    print(f"Using user ID: {used_id} for testing.")

    Ca = {}
    C = ca_cols_ws1.copy()

    set_selector_seed(123)

    while C:
        selected_question = select_question(C)
        Ca[selected_question] = int(df_w1.loc[used_id, selected_question])
        C.remove(selected_question)

        preds, confidences = predict_item_ws1(
            Ca=Ca,
            C=C,
            df_train=df_w1_train,
            cfg=cfg,
            fold=fold,
            store=store,
            random_state=42,
        )

        assert set(preds.keys()) == set(C)
        assert set(confidences.keys()) == set(C)

    assert len(Ca) == len(ca_cols_ws1)
    assert C == []

@pytest.mark.parametrize("ws2_path", [
    "data/sample/ws2/ws2_data_sample.csv",
])
def test_predict_item_ws2(ws2_path):
    # load data
    if not os.path.exists(ws2_path):
        pytest.skip(f"File not found: {ws2_path}")
    df_w2 = pd.read_csv(ws2_path)

    # get column names
    cfg = _load_config()
    data_cfg = cfg.get("data", {})
    ws2_cfg = data_cfg.get("ws2", {})
    pra_col_ws2 = ws2_cfg.get("past_overall_col", "")
    pca_cols_ws2 = ws2_cfg.get("past_item_cols", [])
    ca_cols_ws2 = ws2_cfg.get("current_item_cols", [])
    if not pra_col_ws2 or not pca_cols_ws2 or not ca_cols_ws2:
        raise ValueError("past_overall_col, past_item_cols and current_item_cols must be specified in config for WS2.")
    np.random.seed(0)
    store = ModelStore()
    fold = 0

    used_id = int(np.random.choice(df_w2.index))
    df_w2_train = df_w2[df_w2.index != used_id]

    print(f"Using user ID: {used_id} for testing.")

    Pra = int(df_w2.loc[used_id, pra_col_ws2])
    Pca = {c: int(df_w2.loc[used_id, c]) for c in pca_cols_ws2}
    Ca = {}
    C = ca_cols_ws2.copy()

    set_selector_seed(123)

    while C:
        selected_question = select_question(C)
        Ca[selected_question] = int(df_w2.loc[used_id, selected_question])
        C.remove(selected_question)

        preds, confidences = predict_item_ws2(
            Pra=Pra,
            Pca=Pca,
            Ca=Ca,
            C=C,
            df_train=df_w2_train,
            cfg=cfg,
            fold=fold,
            store=store,
            random_state=42,
        )

        assert set(preds.keys()) == set(C)
        assert set(confidences.keys()) == set(C)

    assert len(Ca) == len(ca_cols_ws2)
    assert C == []

@pytest.mark.parametrize("ws1_path", ["data/sample/ws1/ws1_data_sample.csv"])
def test_predict_overall_ws1(ws1_path):
    if not os.path.exists(ws1_path):
        pytest.skip(f"File not found: {ws1_path}")
    df_w1 = pd.read_csv(ws1_path)

    cfg = _load_config()
    ws1_cfg = cfg.get("data", {}).get("ws1", {})
    ca_cols_ws1 = ws1_cfg.get("item_cols", [])
    ra_col_ws1 = ws1_cfg.get("overall_col")
    if not ca_cols_ws1 or ra_col_ws1 is None:
        raise ValueError("item_cols and overall_col must be specified in config for WS1.")

    np.random.seed(0)
    store = ModelStore()
    fold = 0

    used_id = int(np.random.choice(df_w1.index))
    df_w1_train = df_w1[df_w1.index != used_id]

    print(f"Using user ID: {used_id} for testing.")

    Ca = {c: int(df_w1.loc[used_id, c]) for c in ca_cols_ws1}

    pred, confidence = predict_overall_ws1(
        Ca=Ca,
        df_train=df_w1_train,
        cfg=cfg,
        fold=fold,
        store=store,
        random_state=42,
    )

    assert pred in [1, 2, 3, 4]
    assert 0.0 <= confidence <= 1.0


@pytest.mark.parametrize("ws2_path", ["data/sample/ws2/ws2_data_sample.csv"])
def test_predict_overall_ws2(ws2_path):
    if not os.path.exists(ws2_path):
        pytest.skip(f"File not found: {ws2_path}")
    df_w2 = pd.read_csv(ws2_path)

    cfg = _load_config()
    ws2_cfg = cfg.get("data", {}).get("ws2", {})
    pra_col_ws2 = ws2_cfg.get("past_overall_col", "")
    pca_cols_ws2 = ws2_cfg.get("past_item_cols", [])
    ca_cols_ws2 = ws2_cfg.get("current_item_cols", [])
    ra_col_ws2 = ws2_cfg.get("current_overall_col")
    if not pra_col_ws2 or not pca_cols_ws2 or not ca_cols_ws2 or ra_col_ws2 is None:
        raise ValueError("past_overall_col, past_item_cols, current_item_cols and current_overall_col must be specified in config for WS2.")

    np.random.seed(0)
    store = ModelStore()
    fold = 0

    used_id = int(np.random.choice(df_w2.index))
    df_w2_train = df_w2[df_w2.index != used_id]

    print(f"Using user ID: {used_id} for testing.")

    Pra = int(df_w2.loc[used_id, pra_col_ws2])
    Pca = {c: int(df_w2.loc[used_id, c]) for c in pca_cols_ws2}
    Ca = {c: int(df_w2.loc[used_id, c]) for c in ca_cols_ws2}

    pred, confidence = predict_overall_ws2(
        Pra=Pra,
        Pca=Pca,
        Ca=Ca,
        df_train=df_w2_train,
        cfg=cfg,
        fold=fold,
        store=store,
        random_state=42,
    )

    assert pred in [1, 2, 3, 4]
    assert 0.0 <= confidence <= 1.0