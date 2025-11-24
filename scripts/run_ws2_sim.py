import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from adaptive_self_assessment.simulation.ws2 import run_ws2_simulation

def run_simulations(
        model_name: str = "logistic_regression",
        dir_ws2: str = None,
        RC_THRESHOLD: float = 0.80,
        RI_THRESHOLD: float = 0.70,
        K: int = 5,
        output_csv_path: str = None
):
    """
    2回分の自己評価データで適応型自己評価のシミュレーションを実行する関数
    Parameters:
    -----------
        model_name: str
            総合評価推定に使用するモデル
        dir_ws2: str
            WS2データのディレクトリパス
        RC_THRESHOLD: float
            補完の信頼度閾値
        RI_THRESHOLD: float
            総合評価の信頼度閾値
        K: int
            交差検証の分割数
        output_csv_path: str
            結果保存先パス
    Returns:
        results_ws2: pd.DataFrame
            シミュレーション結果のデータフレーム
    """

    # 2回分の自己評価データでシミュレーション
    print("=== WS2 Simulation Tests Started ===")

    # 全ての能力の結果格納用リスト
    results_all_ws2 = []

    # ディレクトリが指定されいない、存在しない場合は
    if dir_ws2 is None or not os.path.exists(dir_ws2):
        raise ValueError(f"Directory for WS2 data is not specified or does not exist: {dir_ws2}")
    
    # 対象ファイル名をソートして取得
    filenames_ws2 = sorted([f for f in os.listdir(dir_ws2) if f.endswith(".csv")])

    # 各能力ごとにシミュレーションを実行
    for filename in filenames_ws2:

        # データの読み込み
        csv_path = os.path.join(dir_ws2, filename)
        df = pd.read_csv(csv_path)
        skill_name = filename.split("_")[2]  # ファイル名から能力名を抽出

        print(f"==== WS2: {skill_name} ====")
        print(f"Using model: {model_name}")

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


        # 各Foldの結果格納用リスト
        fold_results = []
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
                logs_path=f"outputs/logs/ws2/ws2_{skill_name}_fold{fold + 1}_rc{RC_THRESHOLD}_ri{RI_THRESHOLD}_logs.csv"
            )
            fold_results.append(log_sim)

        # 結果をデータフレームに変換
        df_sim = pd.DataFrame(fold_results)

        # 結果を集計
        result = {
            "skill_name": skill_name,
            "model": model_name,
            "RC_THRESHOLD": RC_THRESHOLD,
            "RI_THRESHOLD": RI_THRESHOLD,
            "total_questions": int(df_sim["total_questions"].mean()),
            "avg_answered_questions": df_sim["avg_answered_questions"].mean(),
            "avg_complemented_questions": df_sim["avg_complemented_questions"].mean(),
            "avg_reduction_rate": df_sim["reduction_rate"].mean(),
            "accuracy_over_threshold": df_sim["accuracy_over_threshold"].mean(),
            "coverage_over_threshold": df_sim["coverage_over_threshold"].mean(),
            "accuracy_all": df_sim["accuracy_all"].mean(),
            "avg_response_time": df_sim["avg_response_time"].mean(),
        }
        
        results_all_ws2.append(result)

    # 結果を保存
    results_ws2 = pd.DataFrame(results_all_ws2)

    if output_csv_path is None:
        # デフォルトの保存先パスを設定
        output_csv_path = f"outputs/results_csv/ws2/ws2_simulation_result_rc{RC_THRESHOLD}_ri{RI_THRESHOLD}.csv"

    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    results_ws2.to_csv(output_csv_path, index=False)  

    print(f"Saved WS2 results to: {output_csv_path}")
    print("=== WS2 Simulation Tests Completed ===")

    return results_ws2

if __name__ == "__main__":
    # シミュレーションのパラメータ設定
    MODEL_NAME = "logistic_regression" # 総合評価推定に使用するモデル
    DIR_WS2 = "data/processed/w2-synthetic_20250326_1300_processed"
    RC_THRESHOLD = 0.80
    RI_THRESHOLD = 0.70
    K = 5 # 交差検証の分割数
    output_csv_path = (f"outputs/results_csv/ws2/ws2_simulation_result_rc{RC_THRESHOLD}_ri{RI_THRESHOLD}.csv")

    results_ws2 = run_simulations(
        model_name=MODEL_NAME,
        dir_ws2=DIR_WS2,
        RC_THRESHOLD=RC_THRESHOLD,
        RI_THRESHOLD=RI_THRESHOLD,
        K=K,
        output_csv_path=output_csv_path
    )