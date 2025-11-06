import numpy as np
import pandas as pd
import os
import time
from typing import List, Tuple, Dict, Any

from adaptive_self_assessment.spec import SPEC_WS1, get_spec_cols
from adaptive_self_assessment.selector import select_question, set_selector_seed
from adaptive_self_assessment.predictor import predict_item_ws1, predict_overall_ws1

def run_ws1_simulation(
        RC_THRESHOLD: float = 0.80, 
        RI_THRESHOLD: float = 0.70, 
        skill_name: str = None,
        model_type: str = "logistic_regression",
        train_df: pd.DataFrame = None,
        test_df: pd.DataFrame = None, 
        logs_path: str = None
    ) -> Dict[str, Any]:
    """
    1回分の自己評価データで適応型自己評価のシミュレーションを実行する関数
    Parameters:
    -----------
        RC_THRESHOLD: float
            補完の信頼度閾値
        RI_THRESHOLD: float
            総合評価の信頼度閾値
        model_type: str
            総合評価推定に使用するモデルのタイプ
        train_df: pd.DataFrame
            訓練用のデータ
        test_df: pd.DataFrame
            テスト用のデータ
        logs_path: str
            ログ保存先パス
    Returns:
        results: Dict[str, any]
            シミュレーション結果
    """
    # 各ユーザーの結果格納用リスト
    logs = []
    # シミュレーション結果格納用辞書
    sim_results = ()

    # 列名を取得
    _, _, ca_cols_ws1, ra_col, _ = get_spec_cols(train_df, SPEC_WS1)

    # 各ユーザーに対してシミュレーションを実行
    for idx, user in test_df.iterrows():
        user_id = user["ID"] # ユーザーIDを取得
        C = ca_cols_ws1.copy() # 未回答項目リスト
        Ca = {} # 回答or補完済み項目

        answered_items = [] # 実際に回答項目
        complemented_items = [] # 補完された項目
        time_log = None # 処理時間を記録
        complement_accuracy = None # 補完の正解率
        correct_complement_items = [] # 正解した補完項目
        is_confident_prediction = False # 総合評価予測が閾値をこえたかどうか

        # 質問セレクタのシード設定
        set_selector_seed(np.random.randint(0, 10000))

        # 処理開始時間
        start_time = time.time()  

        # 未回答項目がある限り繰り返す
        while C:
            
            # 質問項目の選択
            ci = select_question(C)

            # 回答処理
            answer = user[ci] # 実際の回答を使用
            Ca[ci] = answer # 回答をCaに追加
            C.remove(ci) # Cから削除
            answered_items.append(ci) # 実際に回答したことを記録

            # 未回答項目Cを推定
            Rc_preds, Rc_confidences = predict_item_ws1(Ca, C, train_df, random_state=42)

            # 信頼度が閾値以上のものをCaに追加
            for item in C[:]:  # Cのコピーを使用してループ
                conf = Rc_confidences.get(item, 0) # 信頼度を取得
                # 信頼度が閾値を超えた場合
                if conf > RC_THRESHOLD:
                    pred_value = Rc_preds[item]
                    Ca[item] = pred_value # Caに追加
                    C.remove(item) # Cから削除
                    actual_value = user[item] # 実際の値を取得
                    complemented_items.append((item, pred_value, conf, actual_value)) # 補完された情報を記録
        
        # すべての項目が回答or補完された後、総合評価を推定
        Ra_pred, Ra_conf = predict_overall_ws1(Ca, train_df, model_name=model_type, random_state=42)

        # 処理時間の記録
        end_time = time.time()
        time_log = end_time - start_time

        # 信頼度が閾値を超えた場合
        if Ra_conf >= RI_THRESHOLD:
            is_confident_prediction = True

        # 実際の総合評価を取得
        actual_Ra = user[ra_col]

        # 補完された項目の正解率を計算
        sum = 0
        for item, pred_val, conf, actual_val in complemented_items:
            if pred_val == actual_val:
                correct_complement_items.append(item)
                sum += 1
        if complemented_items:
            complement_accuracy = sum / len(complemented_items)

        # 各ユーザーのlogを作成
        user_log = {
            "user_id": user_id,
            "skill": skill_name,
            "total_questions": len(ca_cols_ws1),
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
            "model_type": model_type
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
        accuracy_over_threshold = confident_df["correct"].mean() * 100
        coverage_over_threshold = len(confident_df) / len(logs_df) * 100
    
    # 自動推定の全体の精度
    accuracy_all = logs_df["correct"].mean() * 100

    # シミュレーション結果を辞書にまとめる
    sim_results = {
        "skill": skill_name,
        "model_type": model_type,
        "RC_THRESHOLD": RC_THRESHOLD,
        "RI_THRESHOLD": RI_THRESHOLD,
        "num_samples": len(test_df),
        "accuracy_over_threshold": accuracy_over_threshold,
        "accuracy_all": accuracy_all,
        "coverage_over_threshold": coverage_over_threshold,
        "total_questions": len(ca_cols_ws1),
        "avg_answered_questions": logs_df["num_answered_questions"].mean(),
        "avg_complemented_questions": logs_df["num_complemented_questions"].mean(),
        "logs": logs_df,
        "avg_response_time": logs_df["response_time"].mean(),
        "max_response_time": logs_df["response_time"].max(),
        "min_response_time": logs_df["response_time"].min(),
    }

    # ログを保存
    if logs_path:
        os.makedirs(os.path.dirname(logs_path), exist_ok=True)
        logs_df.to_csv(logs_path, index=False)
    
    return sim_results