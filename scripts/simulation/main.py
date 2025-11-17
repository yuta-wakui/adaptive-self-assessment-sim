import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import pytest

from scripts.simulation.run_ws1_sim import run_ws1_simulation
from scripts.simulation.run_ws2_sim import run_ws2_simulation

# シミュレーションの環境設定
model_name = "logistic regression" # 総合評価推定に使用するモデル
dir_ws1 = "data/processed/ws1_synthetic_240531_processed"
dir_ws2 = "data/processed/w2-synthetic_20250326_1300_processed"
RC_THRESHOLD = 0.85
RI_THRESHOLD = 0.75
K = 5 # 交差検証の分割数

# if __name__ == "__main__":
def test_simulations():
    print("=== Starting Simulation Tests ===")

    # 全ての能力の結果格納用リスト
    results_all_ws1 = []
    results_all_ws2 = []

    # １回分の自己評価データでシミュレーション
    print("=== WS1 Simulation Tests Started ===")
    # 対象ファイル名をソートして取得
    filenames_ws1 = sorted([f for f in os.listdir(dir_ws1) if f.endswith(".csv")])

    # 各能力ごとにシミュレーションを実行
    for filename in filenames_ws1:
        continue
        # csvファイルのみ処理
        if not filename.endswith(".csv"):
            print(f"Skipping non-csv file: {filename}")
            continue

        # データの読み込み
        csv_path = os.path.join(dir_ws1, filename)
        df = pd.read_csv(csv_path)
        skill_name = filename.split("_")[2].replace(".csv", "")  # ファイル名から能力名を抽出

        print(f"==== WS1: {skill_name}のシミュレーションを開始 ====")
        print(f"使用するモデル: {model_name}")

        # 各Foldの結果格納用リスト
        results_sim = []

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
                logs_path=f"outputs/logs/tests/ws1_{skill_name}_fold{fold + 1}_logs.csv"
            )
            results_sim.append(log_sim)

        # 結果をデータフレームに変換
        df_sim = pd.DataFrame(results_sim)

        # 結果を集計
        result = {
            "skill_name": skill_name,
            "model": model_name,
            "total_questions": int(df_sim["total_questions"].mean()),
            "coverage_over_threshold": df_sim["coverage_over_threshold"].mean(),
            "avg_answered_questions": df_sim["avg_answered_questions"].mean(),
            "avg_complemented_questions": df_sim["avg_complemented_questions"].mean(),
            "accuracy_over_threshold": df_sim["accuracy_over_threshold"].mean(),
            "accuracy_all": df_sim["accuracy_all"].mean(),
            "avg_response_time": df_sim["avg_response_time"].mean(),
        }
        
        results_all_ws1.append(result)

    # 結果を保存
    results_ws1 = pd.DataFrame(results_all_ws1)
    results_ws1.to_csv(f"outputs/logs/tests/ws1_simulation_test_summary_rc{RC_THRESHOLD}_ri{RI_THRESHOLD}.csv", index=False)

    print("=== WS1 Simulation Tests Completed ===")

    # 2回分の自己評価データでシミュレーション
    print("=== WS2 Simulation Tests Started ===")
    # 対象ファイル名をソートして取得
    filenames_ws2 = sorted([f for f in os.listdir(dir_ws2) if f.endswith(".csv")])

    # 各能力ごとにシミュレーションを実行
    for filename in filenames_ws2:
        # csvファイルのみ処理
        if not filename.endswith(".csv"):
            print(f"Skipping non-csv file: {filename}")
            continue

        # データの読み込み
        csv_path = os.path.join(dir_ws2, filename)
        df = pd.read_csv(csv_path)
        skill_name = filename.split("_")[2]  # ファイル名から能力名を抽出

        print(f"==== WS2: {skill_name}のシミュレーションを開始 ====")
        print(f"使用するモデル: {model_name}")

        # 各Foldの結果格納用リスト
        results_sim = []

        # クラス分布を表示
        label_counts = df["w4-assessment-result"].value_counts()
        print("Class distribution in the dataset:")
        print(label_counts)

        # クラス数が全体の10%以上のものだけを使用
        valid_labels = label_counts[label_counts >= len(df) * 0.1].index
        removed_labels = label_counts[~label_counts.index.isin(valid_labels)]
        df = df[df["w4-assessment-result"].isin(valid_labels)].copy()
        print(f"Filtered dataset size: {len(df)} after removing rare classes.")
        print(f"Removed classes: {removed_labels.to_dict()}")

        # StratifiedFoldの設定
        skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=42)
        y = df["w4-assessment-result"].values

        # 各分割でシミュレーションを実行
        for fold, (train_idx, test_idx) in enumerate(skf.split(df, y)):
            print(f"---- WS2: Fold {fold + 1} ----")
            train_df = df.iloc[train_idx]
            test_df = df.iloc[test_idx]

            # 適応型自己評価のシミュレーションの実行
            log_sim = run_ws2_simulation(
                RC_THRESHOLD=RC_THRESHOLD,
                RI_THRESHOLD=RI_THRESHOLD,
                skill_name=skill_name,
                model_type=model_name,
                train_df=train_df,
                test_df=test_df,
                logs_path=f"outputs/logs/tests/ws2_{skill_name}_fold{fold + 1}_logs.csv"
            )
            results_sim.append(log_sim)

        # 結果をデータフレームに変換
        df_sim = pd.DataFrame(results_sim)

        # 結果を集計
        result = {
            "skill_name": skill_name,
            "model": model_name,
            "total_questions": int(df_sim["total_questions"].mean()),
            "coverage_over_threshold": df_sim["coverage_over_threshold"].mean(),
            "avg_answered_questions": df_sim["avg_answered_questions"].mean(),
            "avg_complemented_questions": df_sim["avg_complemented_questions"].mean(),
            "accuracy_over_threshold": df_sim["accuracy_over_threshold"].mean(),
            "accuracy_all": df_sim["accuracy_all"].mean(),
            "avg_response_time": df_sim["avg_response_time"].mean(),
        }
        
        results_all_ws2.append(result)

    # 結果を保存
    results_ws2 = pd.DataFrame(results_all_ws2)
    results_ws2.to_csv(f"outputs/logs/tests/ws2_simulation_test_summary_rc{RC_THRESHOLD}_ri{RI_THRESHOLD}.csv", index=False)  

    print("=== WS2 Simulation Tests Completed ===")



