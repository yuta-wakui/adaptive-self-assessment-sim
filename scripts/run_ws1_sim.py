import os
import pandas as pd
import argparse
from sklearn.model_selection import StratifiedKFold

from datetime import datetime

from adaptive_self_assessment.simulation.ws1 import run_ws1_simulation

def run_simulations(
        model_name: str = "logistic_regression",
        dir_ws1: str = None,
        RC_THRESHOLD: float = 0.80,
        RI_THRESHOLD: float = 0.70,
        K: int = 5,
        output_csv_path: str = None
):
    """
    1回分の自己評価データで適応型自己評価のシミュレーションを実行する関数
    Parameters:
    -----------
        model_name: str
            総合評価推定に使用するモデル
        dir_ws1: str
            WS1データのディレクトリパス
        RC_THRESHOLD: float
            補完の信頼度閾値
        RI_THRESHOLD: float
            総合評価の信頼度閾値
        K: int
            交差検証の分割数
        output_csv_path: str
            結果保存先パス
    Returns:
        results_ws1: pd.DataFrame
            シミュレーション結果のデータフレーム
    """

    # １回分の自己評価データでシミュレーション
    print("=== WS1 Simulation Tests Started ===")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 全ての能力の結果格納用リスト
    results_all_ws1 = []

    # ディレクトリが指定されいない、存在しない場合は
    if dir_ws1 is None or not os.path.exists(dir_ws1):
        raise ValueError(f"Directory for WS1 data is not specified or does not exist: {dir_ws1}")
    
    # ログ用ディレクトリの作成
    log_dir = os.path.join("outputs", "logs", "ws1", run_id)
    os.makedirs(log_dir, exist_ok=True)
    print(f"Logs will be saved to: {log_dir}")

    # 対象ファイル名をソートして取得
    filenames_ws1 = sorted([f for f in os.listdir(dir_ws1) if f.endswith(".csv")])

    # ログファイル名用の閾値文字列
    rc_str = str(RC_THRESHOLD).replace(".", "p")
    ri_str = str(RI_THRESHOLD).replace(".", "p")

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

            # ログパスの作成
            log_filename = f"ws1_{skill_name}_fold{fold + 1}_rc{rc_str}_ri{ri_str}_logs.csv"
            log_path = os.path.join(log_dir, log_filename)

            
            # 適応型自己評価のシミュレーションの実行
            log_sim = run_ws1_simulation(
                RC_THRESHOLD=RC_THRESHOLD,
                RI_THRESHOLD=RI_THRESHOLD,
                skill_name=skill_name,
                model_type=model_name,
                train_df=train_df,
                test_df=test_df,
                logs_path=log_path
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
            "num_folds": K,
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
    
    if output_csv_path is None:
        rc_str = str(RC_THRESHOLD).replace(".", "p")
        ri_str = str(RI_THRESHOLD).replace(".", "p")
        output_csv_path = f"outputs/results_csv/ws1/ws1_results_rc{rc_str}_ri{ri_str}_{run_id}.csv"

    dirpath = os.path.dirname(output_csv_path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    results_ws1.to_csv(output_csv_path, index=False)
    
    print(f"Saved WS1 results to: {output_csv_path}")
    print("=== WS1 Simulation Tests Completed ===")

    return results_ws1

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Path to WS1 processed CSVs")
    parser.add_argument("--rc", type=float, default=0.80)
    parser.add_argument("--ri", type=float, default=0.70)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--output", type=str, default=None)
    
    args = parser.parse_args()

    results_ws1 = run_simulations(
        model_name="logistic_regression",
        dir_ws1=args.data_dir,
        RC_THRESHOLD=args.rc,
        RI_THRESHOLD=args.ri,
        K=args.k,
        output_csv_path=args.output
    )