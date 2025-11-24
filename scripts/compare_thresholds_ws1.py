import os
import pandas as pd
from itertools import product

from scripts.run_ws1_sim import run_simulations as run_ws1_sim

def compare_thresholds_ws1(
        model_name: str = "logistic_regression",
        dir_ws1: str = None,
        RC_VALUES: list[float] = None,
        RI_VALUES: list[float] = None,
        K: int = 5,
        output_csv_path: str = None
):
    """
    1回分の自己評価データで適応型自己評価のシミュレーションを複数の閾値で実行し、結果を比較する関数
    Parameters:
    -----------
        model_name: str
            総合評価推定に使用するモデル
        dir_ws1: str
            WS1データのディレクトリパス
        RC_VALUES: list[float]
            補完の信頼度閾値のリスト
        RI_VALUES: list[float]
            総合評価の信頼度閾値のリスト
        K: int
            交差検証の分割数
        output_csv_path: str
            結果保存先パス
    Returns:
        df_all: pd.DataFrame
            すべての閾値組み合わせのシミュレーション結果のデータフレーム
    """

    if dir_ws1 is None or not os.path.exists(dir_ws1):
        raise ValueError(f"Directory for WS1 data is not specified or does not exist: {dir_ws1}")
    
    if RC_VALUES is None or RI_VALUES is None:
        raise ValueError("RC_VALUES and RI_VALUES must be provided.")
    
    all_results: list[pd.DataFrame] = []

    # 各閾値組み合わせでシミュレーションを実行
    for rc, ri in product(RC_VALUES, RI_VALUES):
        print(f"=== Running WS1 Simulation for RC_THRESHOLD={rc}, RI_THRESHOLD={ri} ===")

        # シミュレーションの実行
        df_result = run_ws1_sim(
            model_name=model_name,
            dir_ws1=dir_ws1,
            RC_THRESHOLD=rc,
            RI_THRESHOLD=ri,
            K=K,
            output_csv_path=f"outputs/results_csv/ws1/ws1_simulation_result_rc{rc}_ri{ri}.csv"
        )

        all_results.append(df_result)
    
    # すべての結果を結合
    df_all = pd.concat(all_results, ignore_index=True)

    # 結果の保存
    if output_csv_path is None:
        output_csv_path = "outputs/results_csv/ws1/ws1_threshold_comparison_results.csv"

    os.makedirs("outputs/results_csv/ws1", exist_ok=True)
    df_all.to_csv(output_csv_path, index=False)

    print(f"Saved WS1 threshold comparison to : {output_csv_path}")

    return df_all

if __name__ == "__main__":
    # シミュレーションの環境設定
    MODEL_NAME = "logistic_regression" # 総合評価推定に使用するモデル
    DIR_WS1 = "data/processed/ws1_synthetic_240531_processed"
    RC_VALUES = [0.75, 0.80, 0.85]  # 補完の信頼度閾値のリスト
    RI_VALUES = [0.65, 0.70, 0.75]  # 総合評価の信頼度閾値のリスト
    K = 5 # 交差検証の分割数
    output_csv_path = "outputs/results_csv/ws1/ws1_threshold_comparison_results.csv"

    comparison_result = compare_thresholds_ws1(
        model_name=MODEL_NAME,
        dir_ws1=DIR_WS1,
        RC_VALUES=RC_VALUES,
        RI_VALUES=RI_VALUES,
        K=K,
        output_csv_path = output_csv_path
    )    