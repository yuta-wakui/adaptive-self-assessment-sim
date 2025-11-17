import pandas as pd
import numpy as np
import random
import time
import os
import glob
from models.predictor import predict_items, predict_assessment
from utils.selector import select_question

def run_simulation(RC_THRESHOLD=0.90, RI_THRESHOLD=0.75, model_type="logistic", train_df=None, test_df=None, logs_path=None):
    """
    適応型自己評価のシミュレーションを実行する関数
    Args:
        RC_THRESHOLD (float): 補完の信頼度閾値
        RI_THRESHOLD (float): 総合評価の信頼度閾値
        model_type (str): 総合評価推定に使用するモデルのタイプ
        train_df (pd.DataFrame): 訓練用のデータ
        test_df (pd.DataFrame): テスト用のデータ
        logs_path (str): ログ保存先パス
    Returns:
        summary (dict): シミュレーション結果の要約
        results_df (pd.DataFrame): 各ユーザーの詳細な結果
    """

    results = [] # 結果を格納するリスト

    for idx, user in test_df.iterrows():
        user_id = user["ID"]
        C = [col for col in train_df.columns if col.startswith("w4-") and col not in ["w4-assessment-result", "w4-reflection-length"]]
        Ca = {}
        Pca = {k: user[k] for k in train_df.columns if k.startswith("w3-") and k not in ["w3-assessment-result"]}
        Pra = user["w3-assessment-result"]

        answered_items = [] # 回答済み項目
        complemented_items = [] # 補完された項目
        time_logs = [] # 各ステップの処理時間
        is_confident_prediction = False # 総合評価予測が閾値をこえたかどうか

        while C:
            start_time = time.time()  # 処理開始時間
            
            # 質問項目の選択
            ci = select_question(C)

            # 回答
            answer = user[ci] # 実際の回答を使用
            Ca[ci] = answer
            C.remove(ci)
            answered_items.append(ci)

            # 残り項目の補完
            Rc_preds, Rc_confidences = predict_items(Pra, Pca, Ca, C, train_df, user_id)

            # 信頼度が閾値以上のものをCaに追加
            for item in C[:]:  # Cのコピーを使用してループ
                conf = Rc_confidences.get(item, 0)
                if conf > RC_THRESHOLD:
                    pred_value = Rc_preds[item]
                    actual_value = user[item]
                    Ca[item] = pred_value
                    C.remove(item)
                    complemented_items.append((item, pred_value, conf, actual_value))

            end_time = time.time()  # 処理終了時間
            time_logs.append(end_time - start_time)  # 処理時間を記録

        # 総合評価の推定
        Ra_pred, Ra_conf = predict_assessment(Pra, Pca, Ca, train_df, user_id, model_type=model_type)

        # 信頼度が閾値を超えた場合
        if Ra_conf >= RI_THRESHOLD:
            is_confident_prediction = True

        actual_Ra = user["w4-assessment-result"] # 実際の総合評価

        # 補完された項目の正解率を計算
        correct_complements = sum(1 for item, pred_val, conf, actual_val in complemented_items if pred_val == actual_val)
        if complemented_items:
            complement_accuracy = correct_complements / len(complemented_items)
        else:
            complement_accuracy = np.nan

        # 各userのシミュレーション結果を格納
        result = {
            "user_id": user_id,
            "num_questions": len(answered_items),
            "num_completions": len(complemented_items),
            "completion_accuracy": complement_accuracy,
            "predicted_ra": Ra_pred,
            "actual_ra": actual_Ra,
            "confidence": Ra_conf,
            "is_confident": is_confident_prediction,
            "correct": int(Ra_pred == actual_Ra),
            "total_response_time": sum(time_logs),
            "avg_response_time": np.mean(time_logs) if time_logs else np.nan,
            "max_response_time": max(time_logs) if time_logs else np.nan,
            "min_response_time": min(time_logs) if time_logs else np.nan,
        }

        results.append(result)

    # 結果集計
    # 各userの詳細な結果
    results_df = pd.DataFrame(results)
    # 閾値を超えた自動推定の精度 
    confident_df = results_df[results_df["is_confident"] == True]
    accuracy_confident_auto = confident_df["correct"].mean() * 100
    avg_confidence_over_threshold = confident_df["confidence"].mean()

    # 閾値を超えた自動推定の割合
    coverage = len(confident_df) / len(results_df) * 100

    # 自動推定の全体の精度
    accuracy_all_auto = results_df["correct"].mean() * 100
    avg_confidence_all_auto = results_df["confidence"].mean()

    # 閾値を超えなかったら直接質問をすると仮定した場合の全体の精度
    df_with_direct_question = results_df.copy()
    df_with_direct_question.loc[df_with_direct_question["is_confident"] == False, "correct"] = 1
    accuracy_with_direct_question = df_with_direct_question["correct"].mean() * 100

    # シミュレーション結果の要約
    summary = {
        "model": "adaptive3",
        "num_samples": len(test_df),
        "RC_THRESHOLD": RC_THRESHOLD,
        "RI_THRESHOLD": RI_THRESHOLD,
        "avg_questions": results_df["num_questions"].mean(),
        "accuracy_confident_auto": accuracy_confident_auto,
        "accuracy_all_auto": accuracy_all_auto,
        "accuracy_with_direct_question": accuracy_with_direct_question,
        "coverage": coverage,
        "avg_confidence_over_threshold": avg_confidence_over_threshold,
        "avg_confidence_all_auto": avg_confidence_all_auto,
        "avg_completions": results_df["num_completions"].mean(),
        "avg_completion_accuracy": results_df["completion_accuracy"].dropna().mean() * 100,
        "avg_total_response_time": results_df["total_response_time"].mean(),
        "avg_avg_response_time": results_df["avg_response_time"].mean(),
        "max_response_time": results_df["max_response_time"].max(),
        "min_response_time": results_df["min_response_time"].min(),
    }

    # CSVに保存
    if logs_path is not None:
        os.makedirs(os.path.dirname(logs_path), exist_ok=True)  # ディレクトリがなければ作成
        results_df.to_csv(logs_path, index=False)  

    return summary, results_df