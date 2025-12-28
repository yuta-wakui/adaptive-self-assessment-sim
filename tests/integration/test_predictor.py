import os
import pandas as pd
import numpy as np
import pytest
import yaml

from adaptive_self_assessment.selector import select_question, set_selector_seed
from adaptive_self_assessment.predictor import predict_item_ws1, predict_item_ws2, predict_overall_ws1, predict_overall_ws2
from adaptive_self_assessment.model_store import ModelStore

def _load_config(config_path: str = "configs/config.yaml"):
    """
    YAML形式の設定ファイルを読み込む関数
    
    Parameters:
    ----------
    config_path: str
        設定ファイルのパス（デフォルト； "configs/config.yaml"）
    
    Returns:
    -------
    cfg: Dict[str, Any] 
        読み込んだ設定内容を辞書形式で返す
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    return cfg

@pytest.mark.parametrize("ws1_path", [
    "data/sample/ws1/ws1_data_sample.csv",
])
def test_predict_item_ws1(ws1_path):
    print("=== Item Prediction Test in WS1 ===")
    # データの読み込み
    if not os.path.exists(ws1_path):
        pytest.skip(f"File not found: {ws1_path}")
    df_w1 = pd.read_csv(ws1_path)

    # 列名の取得
    cfg = _load_config()
    data_cfg = cfg.get("data", {})
    ca_cols_ws1 = data_cfg.get("ws1", {}).get("ca_cols", [])
    if not ca_cols_ws1:
        raise ValueError("ca_cols must be specified in config for WS1.")

    # 乱数固定（テスト再現性）
    np.random.seed(42)

    # モデルキャッシュ
    store = ModelStore()
    fold = 0

    use_id = int(np.random.choice(df_w1.index))
    print( "User ID for prediction:", use_id)
    print("--------------------------------")

    # 自分のデータを訓練データから除外
    df_w1_train = df_w1[df_w1.index != use_id]

    Ca = {}
    C = ca_cols_ws1.copy()

    set_selector_seed(int(np.random.randint(0, 100)))

    while C:
        selected_question = select_question(C)

        Ca[selected_question] = int(df_w1.loc[use_id, selected_question])
        C.remove(selected_question)

        print("Selected Question: ", selected_question)
        print("Current Known Answers: ", Ca)
        remaining_truth = {c: int(df_w1.loc[use_id, c]) for c in C}
        print("Remaining Questions: ", remaining_truth)

        preds, confidences = predict_item_ws1(
            Ca=Ca,
            C=C,
            df_train=df_w1_train,
            cfg=cfg,
            fold=fold,
            store=store,
            random_state=42,
        )

        print(f"Predicted: {preds}")
        print(f"Confidences: {confidences}")
        print("--------------------------------")

    assert len(Ca) == len(ca_cols_ws1)
    assert C == []

@pytest.mark.parametrize("ws2_path", [
    "data/sample/ws2/ws2_data_sample1.csv",
])
def test_predict_item_ws2(ws2_path):
    print("=== Item Prediction Test in WS2 ===")
    # データの読み込み
    if not os.path.exists(ws2_path):
        pytest.skip(f"File not found: {ws2_path}")
    df_w2 = pd.read_csv(ws2_path)

    # 列名の取得
    cfg = _load_config()
    data_cfg = cfg.get("data", {})
    ws2_cfg = data_cfg.get("ws2", {})
    pra_col_ws2 = ws2_cfg.get("pra_col", "")
    pca_cols_ws2 = ws2_cfg.get("pca_cols", [])
    ca_cols_ws2 = ws2_cfg.get("ca_cols", [])
    if not pra_col_ws2 or not pca_cols_ws2 or not ca_cols_ws2:
        raise ValueError("pra_col, pca_cols and ca_cols must be specified in config for WS2.")

    np.random.seed(0)

    store = ModelStore()
    fold = 0


    use_id = int(np.random.choice(df_w2.index))
    print("User ID for prediction:", use_id)
    print("--------------------------------")

    df_w2_train = df_w2[df_w2.index != use_id]

    Pra = int(df_w2.loc[use_id, pra_col_ws2])
    Pca = {c: int(df_w2.loc[use_id, c]) for c in pca_cols_ws2}
    Ca = {}
    C = ca_cols_ws2.copy()

    set_selector_seed(int(np.random.randint(0, 10000)))

    print("Past Overall Assessment (Pra):", Pra)
    print("Past Checklist Answers (Pca):", Pca)

    while C:
        selected_question = select_question(C)
        Ca[selected_question] = int(df_w2.loc[use_id, selected_question])
        C.remove(selected_question)

        print("Selected Question:", selected_question)
        print("Current Known Answers:", Ca)
        remaining_truth = {c: int(df_w2.loc[use_id, c]) for c in C}
        print("Remaining Questions:", remaining_truth)

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

        print(f"Predicted: {preds}")
        print(f"Confidences: {confidences}")
        print("--------------------------------")

    assert len(Ca) == len(ca_cols_ws2)
    assert C == []

@pytest.mark.parametrize("ws1_path", ["data/sample/ws1/ws1_data_sample.csv"])
def test_predict_overall_ws1(ws1_path):
    print("=== Overall Assessment Prediction Test in WS1 ===")

    if not os.path.exists(ws1_path):
        pytest.skip(f"File not found: {ws1_path}")
    df_w1 = pd.read_csv(ws1_path)

    cfg = _load_config()
    data_cfg = cfg.get("data", {})
    ws1_cfg = data_cfg.get("ws1", {})
    ca_cols_ws1 = ws1_cfg.get("ca_cols", [])
    ra_col_ws1 = ws1_cfg.get("ra_col")
    if not ca_cols_ws1 or ra_col_ws1 is None:
        raise ValueError("ca_cols and ra_col must be specified in config for WS1.")

    np.random.seed(0)

    store = ModelStore()
    fold = 0

    used_id = int(np.random.choice(df_w1.index))
    print("User ID for prediction:", used_id)
    print("--------------------------------")

    df_w1_train = df_w1[df_w1.index != used_id]

    Ca = {c: int(df_w1.loc[used_id, c]) for c in ca_cols_ws1}
    true_ra = int(df_w1.loc[used_id, ra_col_ws1])

    print(f"Checklist Answers (Ca): {Ca}")
    print(f"True Overall Assessment (Ra): {true_ra}")

    pred, confidence = predict_overall_ws1(
        Ca=Ca,
        df_train=df_w1_train,
        cfg=cfg,
        fold=fold,
        store=store,
        random_state=42,
    )

    print(f"Predicted Overall Assessment: {pred}")
    print(f"Confidence: {confidence}")

    assert pred in [1, 2, 3, 4]
    assert 0.0 <= confidence <= 1.0


@pytest.mark.parametrize("ws2_path", ["data/sample/ws2/ws2_data_sample1.csv"])
def test_predict_overall_ws2(ws2_path):
    print("=== Overall Assessment Prediction Test in WS2 ===")

    if not os.path.exists(ws2_path):
        pytest.skip(f"File not found: {ws2_path}")
    df_w2 = pd.read_csv(ws2_path)

    cfg = _load_config()
    data_cfg = cfg.get("data", {})
    ws2_cfg = data_cfg.get("ws2", {})
    pra_col_ws2 = ws2_cfg.get("pra_col", "")
    pca_cols_ws2 = ws2_cfg.get("pca_cols", [])
    ca_cols_ws2 = ws2_cfg.get("ca_cols", [])
    ra_col_ws2 = ws2_cfg.get("ra_col")
    if not pra_col_ws2 or not pca_cols_ws2 or not ca_cols_ws2 or ra_col_ws2 is None:
        raise ValueError("pra_col, pca_cols, ca_cols and ra_col must be specified in config for WS2.")

    np.random.seed(0)

    store = ModelStore()
    fold = 0

    used_id = int(np.random.choice(df_w2.index))
    print("User ID for prediction:", used_id)
    print("--------------------------------")

    df_w2_train = df_w2[df_w2.index != used_id]

    Pra = int(df_w2.loc[used_id, pra_col_ws2])
    Pca = {c: int(df_w2.loc[used_id, c]) for c in pca_cols_ws2}
    Ca = {c: int(df_w2.loc[used_id, c]) for c in ca_cols_ws2}
    true_ra = int(df_w2.loc[used_id, ra_col_ws2])

    print(f"Past Overall Assessment (Pra): {Pra}")
    print(f"Past Checklist Answers (Pca): {Pca}")
    print(f"Current Checklist Answers (Ca): {Ca}")
    print(f"True Overall Assessment (Ra): {true_ra}")

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

    print(f"Predicted Overall Assessment: {pred}")
    print(f"Confidence: {confidence}")

    assert pred in [1, 2, 3, 4]
    assert 0.0 <= confidence <= 1.0