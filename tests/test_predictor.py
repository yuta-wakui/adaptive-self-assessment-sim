import os
import pandas as pd
import numpy as np
import pytest

from adaptive_self_assessment.selector import select_question, set_selector_seed
from adaptive_self_assessment.spec import SPEC_WS1, SPEC_WS2, get_spec_cols
from adaptive_self_assessment.predictor import predict_item_ws1, predict_item_ws2, predict_overall_ws1

@pytest.mark.parametrize("ws1_path", [
    "data/processed/ws1_synthetic_240531_processed/1_syntheticdata_informationliteracy.csv",
])
def test_predict_item_ws1(ws1_path):
    print("=== Item Prediction Test in WS1 ===")
    # データの読み込み
    if not os.path.exists(ws1_path):
        pytest.skip(f"File not found: {ws1_path}")
    df_w1 = pd.read_csv(ws1_path)

    # 列名の取得
    _, _, ca_cols_ws1, _, _ = get_spec_cols(df_w1, SPEC_WS1)

    # テストするユーザーIDをランダムに選択（1からデータ数までの範囲）
    use_id = np.random.randint(1, len(df_w1)+1)
    print( "User ID for prediction:", use_id)
    print("--------------------------------")

    # 自分のデータを訓練データから除外
    df_w1_train = df_w1[df_w1.index != use_id]

    # そのユーザーの真値を取得
    Ca = {}
    C = ca_cols_ws1.copy()

    # 質問セレクタのシード設定
    set_selector_seed(np.random.randint(0, 10000))

    # 逐次推定
    while C:
        # 1項目選択して回答を取得
        selected_question = select_question(C)

        # 実際の回答をCaに追加し、Cから削除
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
            random_state=42
        )

        print(f"Predicted: {preds}")
        print(f"Confidences: {confidences}")
        print("--------------------------------")

    # 最終的に全項目が確定していること
    assert len(Ca) == len(ca_cols_ws1)
    assert C == []

@pytest.mark.parametrize("ws2_path", [
    "data/processed/w2-synthetic_20250326_1300_processed/ws2_1_information_1300_processed.csv",
])
def test_predict_item_ws2(ws2_path):
    print("=== Item Prediction Test in WS2 ===")
    # データの読み込み
    if not os.path.exists(ws2_path):
        pytest.skip(f"File not found: {ws2_path}")
    df_w2 = pd.read_csv(ws2_path)

    # 列名の取得
    pra_col_ws2, pca_cols_ws2, ca_cols_ws2, _, _ = get_spec_cols(df_w2, SPEC_WS2)

    # テストするユーザーIDをランダムに選択（1からデータ数までの範囲）
    use_id = np.random.randint(1, len(df_w2)+1)
    print( "User ID for prediction:", use_id)
    print("--------------------------------")

    # 自分のデータを訓練データから除外
    df_w2_train = df_w2[df_w2.index != use_id]

    # そのユーザーの真値を取得
    Pra = int(df_w2.loc[use_id, pra_col_ws2])
    Pca = {c: int(df_w2.loc[use_id, c]) for c in pca_cols_ws2}
    Ca = {}
    C = ca_cols_ws2.copy()

    # 質問セレクタのシード設定
    set_selector_seed(np.random.randint(0, 10000))

    print("Past Overall Assessment (Pra): ", Pra)
    print("Past Checklist Answers (Pca): ", Pca)

    # 逐次推定
    while C:
        # 1項目選択して回答を取得
        selected_question = select_question(C)

        # 実際の回答をCaに追加し、Cから削除
        Ca[selected_question] = int(df_w2.loc[use_id, selected_question])
        C.remove(selected_question)

        print("Selected Question: ", selected_question)
        print("Current Known Answers: ", Ca)
        remaining_truth = {c: int(df_w2.loc[use_id, c]) for c in C}
        print("Remaining Questions: ", remaining_truth)
        preds, confidences = predict_item_ws2(
            Pra=Pra,
            Pca=Pca,
            Ca=Ca,
            C=C,
            df_train=df_w2_train,
            random_state=42
        )

        print(f"Predicted: {preds}")
        print(f"Confidences: {confidences}")
        print("--------------------------------")

    # 最終的に全項目が確定していること
    assert len(Ca) == len(ca_cols_ws2)
    assert C == []

@pytest.mark.parametrize("ws1_path", [
    "data/processed/ws1_synthetic_240531_processed/1_syntheticdata_informationliteracy.csv",
])
def test_predict_overall_ws1(ws1_path):
    print("=== Overall Assessment Prediction Test in WS1 ===")
    # データの読み込み
    if not os.path.exists(ws1_path):
        pytest.skip(f"File not found: {ws1_path}")
    df_w1 = pd.read_csv(ws1_path)

    # 列名の取得
    _, _, ca_cols_ws1, ra_col_ws1, _ = get_spec_cols(df_w1, SPEC_WS1)

    # テストするユーザーIDをランダムに選択（1からデータ数までの範囲）
    used_id = np.random.randint(1, len(df_w1)+1)
    print( "User ID for prediction:", used_id)
    print("--------------------------------")


    # 自分のデータを訓練データから除外
    df_w1_train = df_w1[df_w1.index != used_id]

    # そのユーザーの真値を取得
    Ca = {c: int(df_w1.loc[used_id, c]) for c in ca_cols_ws1}
    true_ra = int(df_w1.loc[used_id, ra_col_ws1])

    print(f"Checklist Answers (Ca): {Ca}")
    print(f"True Overall Assessment (Ra): {true_ra}")

    # 総合評価の予測
    preds, confidences = predict_overall_ws1(
        Ca=Ca,
        df_train=df_w1_train,
        random_state=42
    )

    print(f"Predicted Overall Assessment: {preds}")
    print(f"Confidence: {confidences}")

    assert preds in [1, 2, 3, 4]
    assert 0.0 <= confidences <= 1.0


