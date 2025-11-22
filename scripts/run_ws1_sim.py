import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from adaptive_self_assessment.simulation.ws1 import run_ws1_simulation


def run_simulations():
    # シミュレーションのパラメータ設定
    model_name = "logistic_regression" # 総合評価推定に使用するモデル
    dir_ws1 = "data/processed/ws1_synthetic_240531_processed"
    dir_ws2 = "data/processed/w2-synthetic_20250326_1300_processed"
    RC_THRESHOLD = 0.80
    RI_THRESHOLD = 0.70
    K = 5 # 交差検証の分割数

    # １回分の自己評価データでシミュレーション
    print("=== WS1 Simulation Tests Started ===")

    # 全ての能力の結果格納用リスト
    results_all_ws1 = []

    # 対象ファイル名をソートして取得
    filenames_ws1 = sorted([f for f in os.listdir(dir_ws1) if f.endswith(".csv")])

    # 各能力ごとにシミュレーションを実行
    for filename in filenames_ws1:

        # データの読み込み
        csv_path = os.path.join(dir_ws1, filename)
        df = pd.read_csv(csv_path)
        skill_name = filename.split("_")[2].replace(".csv", "")  # ファイル名から能力名を抽出

        print(f"==== WS1: {skill_name} ====")
        print(f"Using model: {model_name}")

        # クラス分布を表示
        label_counts = df["assessment-result"].value_counts()
        print("Class distribution in the dataset:")
        print(label_counts)

        # クラス数が全体の10%以上のものだけを使用
        valid_labels = label_counts[label_counts >= len(df) * 0.1].index
        removed_labels = label_counts[~label_counts.index.isin(valid_labels)]
        df = df[df["assessment-result"].isin(valid_labels)].copy()
        print(f"Filtered dataset size: {len(df)} after removing rare classes.")
        print(f"Removed classes: {removed_labels.to_dict()}")

        # StratifiedFoldの設定
        skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=42)
        y = df["assessment-result"].values

        # 各Foldの結果格納用リスト
        fold_results = []

        # 各分割でシミュレーションを実行
        for fold, (train_idx, test_idx) in enumerate(skf.split(df, y)):
            print(f"---- WS1: Fold {fold + 1} ----")
            train_df = df.iloc[train_idx]
            test_df = df.iloc[test_idx]

            # 適応型自己評価のシミュレーションの実行
            log_sim = run_ws1_simulation(
                RC_THRESHOLD=RC_THRESHOLD,
                RI_THRESHOLD=RI_THRESHOLD,
                skill_name=skill_name,
                model_type=model_name,
                train_df=train_df,
                test_df=test_df,
                logs_path=f"outputs/logs/ws1/ws1_{skill_name}_fold{fold + 1}_logs.csv"
            )
            fold_results.append(log_sim)

        # 結果をデータフレームに変換
        df_sim = pd.DataFrame(fold_results)

        # 結果を集計
        result = {
            "skill_name": skill_name,
            "model": model_name,
            "total_questions": int(df_sim["total_questions"].mean()),
            "avg_answered_questions": df_sim["avg_answered_questions"].mean(),
            "avg_complemented_questions": df_sim["avg_complemented_questions"].mean(),
            "avg_reduction_rate": df_sim["reduction_rate"].mean(),
            "accuracy_over_threshold": df_sim["accuracy_over_threshold"].mean(),
            "coverage_over_threshold": df_sim["coverage_over_threshold"].mean(),
            "accuracy_all": df_sim["accuracy_all"].mean(),
            "avg_response_time": df_sim["avg_response_time"].mean(),
        }
        
        results_all_ws1.append(result)

    # 結果を保存
    results_ws1 = pd.DataFrame(results_all_ws1)
    os.makedirs("outputs/results_csv", exist_ok=True)
    results_ws1.to_csv(f"outputs/results_csv/ws1/ws1_simulation_result_rc{RC_THRESHOLD}_ri{RI_THRESHOLD}.csv", index=False)

    print("=== WS1 Simulation Tests Completed ===")

if __name__ == "__main__":
    run_simulations()