import numpy as np
import pandas as pd
import os
import time
from datetime import datetime
from typing import List, Dict, Any, Optional

from adaptive_self_assessment.config import load_config
from adaptive_self_assessment.selector import select_question, set_selector_seed
from adaptive_self_assessment.predictor import predict_item_ws2, predict_overall_ws2

def run_ws2_simulation(
        train_df: pd.DataFrame = None,
        test_df: pd.DataFrame = None, 
    ) -> Dict[str, Any]:
    """
    2回分の自己評価データで適応型自己評価のシミュレーションを実行する関数
    Parameters:
    -----------
        train_df: pd.DataFrame
            訓練用のデータ
        test_df: pd.DataFrame
            テスト用のデータ
    Returns:
        results: Dict[str, any]
            シミュレーション結果
    """
    if train_df is None or test_df is None:
        raise ValueError("train_df and test_df must be provided.")

    # configの読み込み
    cfg = load_config()

    # 閾値
    thresholds_cfg = cfg.get("thresholds", {})
    RC_THRESHOLD: float = float(thresholds_cfg.get("RC", 0.80))  # 項目補完の信頼度閾値
    RI_THRESHOLD: float = float(thresholds_cfg.get("RI", 0.70))  # 総合評価推定の信頼度閾値

    # データ設定
    data_cfg = cfg.get("data", {})
    common_cfg = data_cfg.get("common", {})
    ws2_cfg = data_cfg.get("ws2", {})

    skill_name: str = common_cfg.get("skill-name", "")
    id_col: str = common_cfg.get("id_col", "ID")

    if not skill_name:
        skill_name = "unknown_skill"

    if id_col not in train_df.columns or id_col not in test_df.columns:
        raise ValueError(f"id_col '{id_col}' not found in train_df or test_df.")

    pra_col: str = ws2_cfg.get("pra_col", "")
    pca_cols: List[str] = ws2_cfg.get("pca_cols", [])
    ra_col: str = ws2_cfg.get("ra_col", "")
    ca_cols_ws2: List[str] = ws2_cfg.get("ca_cols", [])

    if (not pra_col) or (not pca_cols) or (not ra_col) or (not ca_cols_ws2):
        raise ValueError("pra_col, pca_cols, ra_col, and ca_cols must be specified in config for WS2.")

    # モデルタイプ
    model_cfg = cfg.get("model", {})
    overall_model_type: str = model_cfg.get("overall", {}).get("type", "logistic_regression")

    # ログ設定
    logging_cfg = cfg.get("logging", {})
    save_logs: bool = bool(logging_cfg.get("save_logs", True))
    log_dir: str = logging_cfg.get("log_dir", "outputs/logs/ws2")
    timestamped: bool = bool(logging_cfg.get("timestamped", True))

    # 訓練データに必要な列が存在するか確認
    for col in pca_cols + ca_cols_ws2 + [pra_col, ra_col]:
        if col not in train_df.columns:
            raise ValueError(f"Column '{col}' not found in train_df.")
    
    # テストデータに必要な列が存在するか確認
    for col in pca_cols + ca_cols_ws2 + [pra_col, ra_col]:
        if col not in test_df.columns:
            raise ValueError(f"Column '{col}' not found in test_df.")

    # 各ユーザーの結果格納用リスト
    logs: List[Dict[str, Any]] = []

    # 各ユーザーに対してシミュレーションを実行
    for _, user in test_df.iterrows():
        user_id = user[id_col] # ユーザーIDを取得

        Pra: int = int(user[pra_col]) # 過去の総合評価
        Pca: Dict[str, int] = {c: int(user[c]) for c in pca_cols} # 過去のチェックリスト結果
        C: List[str] = ca_cols_ws2.copy() # 未回答項目リスト
        Ca: Dict[str, int] = {} # 回答or補完済み項目

        answered_items: List[str] = [] # 実際に回答項目
        complemented_items: List[tuple] = [] # 補完された項目
        correct_complement_items: List[str] = [] # 正解した補完項目

        # 質問セレクタのシード設定（ユーザーごとに変える）
        set_selector_seed(np.random.randint(0, 10000))

        # 処理開始時間
        start_time = time.time()   

        # 未回答項目がある限り繰り返す
        while C:

            # 質問項目の選択
            ci = select_question(C)

            # 回答処理
            answer = int(user[ci]) # 実際の回答を使用
            Ca[ci] = answer # 回答をCaに追加
            C.remove(ci) # Cから削除
            answered_items.append(ci) # 実際に回答したことを記録

            # 残り項目の補完
            Rc_preds, Rc_confidences = predict_item_ws2(
                Pra=Pra,
                Pca=Pca,
                Ca=Ca,
                C=C,
                df_train=train_df,
                random_state=42
            )

            # 信頼度が閾値以上のものをCaに追加
            for item in C[:]:  # Cのコピーを使用してループ
                conf = float(Rc_confidences.get(item, 0.0)) # 信頼度を取得
                # 信頼度が閾値を超えた場合
                if conf >= RC_THRESHOLD:
                    pred_value = int(Rc_preds[item])
                    Ca[item] = pred_value # Caに追加
                    C.remove(item) # Cから削除
                    actual_value = int(user[item]) # 実際の値を取得
                    complemented_items.append((item, pred_value, conf, actual_value)) # 補完された情報を記録 
                
        # すべての項目が回答or補完された後に総合評価を推定
        Ra_pred, Ra_conf = predict_overall_ws2(
            Pra=Pra,
            Pca=Pca,
            Ca=Ca,
            df_train=train_df,
            random_state=42
        )

        # 処理時間の記録
        end_time = time.time()
        time_log = end_time - start_time

        # 総合評価予測が閾値をこえたかどうか
        is_confident_prediction = (Ra_conf >= RI_THRESHOLD)

        # 実際の総合評価を取得
        actual_Ra = int(user[ra_col])

        # 補完された項目の正解率を計算
        complement_accuracy: Optional[float] = None # 補完の正解率
        correct_count = 0
        for item, pred_val, _, actual_val in complemented_items:
            if pred_val == actual_val:
                correct_complement_items.append(item)
                correct_count += 1
        if complemented_items:
            complement_accuracy = correct_count / len(complemented_items)

        # 各userのシミュレーション結果を格納
        user_log = {
            "user_id": user_id,
            "skill" : skill_name,
            "total_questions": len(ca_cols_ws2),
            "num_answered_questions": len(answered_items),
            "num_complemented_questions": len(complemented_items),
            "predicted_ra": Ra_pred,
            "actual_ra": actual_Ra,
            "confidence": Ra_conf,
            "is_confident": is_confident_prediction,
            "correct": int(Ra_pred == actual_Ra),
            "complement_accuracy": complement_accuracy,
            "answered_items": answered_items,
            "complemented_items": complemented_items,
            "correct_complement_items": correct_complement_items,
            "response_time": time_log,
            "RC_THRESHOLD": RC_THRESHOLD,
            "RI_THRESHOLD": RI_THRESHOLD,
            "num_train": len(train_df),
            "model_type": overall_model_type,
        }
        logs.append(user_log)
    
    # 結果集計
    logs_df = pd.DataFrame(logs) # 各ユーザーの結果をデータフレームに変換

    accuracy_over_threshold = None # 閾値を超えた自動推定の精度
    accuracy_all = None # 全体の自動推定の精度
    coverage_over_threshold = None # 閾値を超えた自動推定の割合

    # 閾値を超えた自動推定の精度
    confident_df = logs_df[logs_df["is_confident"] == True]
    if not confident_df.empty:
        accuracy_over_threshold = float(confident_df["correct"].mean() * 100)
        coverage_over_threshold = len(confident_df) / len(logs_df) * 100
    
    # 自動推定の全体の精度
    accuracy_all = float(logs_df["correct"].mean() * 100)

    # 平均回答数・平均補完数・削減率を計算
    avg_answered_questions = float(logs_df["num_answered_questions"].mean())
    avg_complemented_questions = float(logs_df["num_complemented_questions"].mean())
    total_questions = len(ca_cols_ws2)
    reduction_rate = float((1.0 - avg_answered_questions / total_questions) * 100.0)

    # レスポンス時間の統計量
    if logs_df["response_time"].empty:
        avg_rt = max_rt = min_rt = None
    else:
        avg_rt = float(logs_df["response_time"].mean())
        max_rt = float(logs_df["response_time"].max())
        min_rt = float(logs_df["response_time"].min())

    # シミュレーション結果を辞書にまとめる
    sim_results = {
        "skill": skill_name,
        "model_type": overall_model_type,
        "RC_THRESHOLD": RC_THRESHOLD,
        "RI_THRESHOLD": RI_THRESHOLD,
        "num_train": len(train_df),
        "num_test": len(test_df),
        "accuracy_over_threshold": accuracy_over_threshold,
        "accuracy_all": accuracy_all,
        "coverage_over_threshold": coverage_over_threshold,
        "total_questions": total_questions,
        "avg_answered_questions": avg_answered_questions,
        "avg_complemented_questions": avg_complemented_questions,
        "reduction_rate": reduction_rate,
        "avg_response_time": avg_rt,
        "max_response_time": max_rt,
        "min_response_time": min_rt,
    }

    # ログを保存
    if save_logs and log_dir:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join(log_dir, run_id) if timestamped else log_dir
        os.makedirs(out_dir, exist_ok=True)

        rc_str = str(RC_THRESHOLD).replace(".", "p")
        ri_str = str(RI_THRESHOLD).replace(".", "p")
        log_filename = f"ws2_{skill_name}_rc{rc_str}_ri{ri_str}_logs.csv"
        logs_path = os.path.join(out_dir, log_filename)

        logs_df.to_csv(logs_path, index=False)
    
    return sim_results