import os
import pandas as pd
import argparse
from itertools import product

from run_ws2_sim import run_simulations as run_ws2_sim

def compare_thresholds_ws2(
        model_name: str = "logistic_regression",
        dir_ws2: str = None,
        RC_VALUES: list[float] = None,
        RI_VALUES: list[float] = None,
        K: int = 5,
        output_csv_path: str = None
):
    """
    2回分の自己評価データで適応型自己評価のシミュレーションを複数の閾値で実行し、結果を比較する関数
    Parameters:
    -----------
        model_name: str
            総合評価推定に使用するモデル
        dir_ws2: str
            WS2データのディレクトリパス
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

    if dir_ws2 is None or not os.path.exists(dir_ws2):
        raise ValueError(f"Directory for WS2 data is not specified or does not exist: {dir_ws2}")
    
    if RC_VALUES is None or RI_VALUES is None:
        raise ValueError("RC_VALUES and RI_VALUES must be provided.")
    
    all_results: list[pd.DataFrame] = []

    # 各閾値組み合わせでシミュレーションを実行
    for rc, ri in product(RC_VALUES, RI_VALUES):
        print(f"=== Running WS2 Simulation for RC_THRESHOLD={rc}, RI_THRESHOLD={ri} ===")

        # ログファイル名用の閾値文字列
        rc_str = str(rc).replace(".", "p")
        ri_str = str(ri).replace(".", "p")

        sim_output_path = f"outputs/results_csv/ws2/sim_results/ws2_simulation_result_rc{rc_str}_ri{ri_str}.csv"

        # シミュレーションの実行
        df_result = run_ws2_sim(
            model_name=model_name,
            dir_ws2=dir_ws2,
            RC_THRESHOLD=rc,
            RI_THRESHOLD=ri,
            K=K,
            output_csv_path=sim_output_path
        )

        all_results.append(df_result)
    
    # すべての結果を結合
    df_all = pd.concat(all_results, ignore_index=True)

    # 結果の保存
    if output_csv_path is None:
        output_csv_path = "outputs/results_csv/ws2/cmp_thresholds/ws2_threshold_comparison_results.csv"

    output_dir = os.path.dirname(output_csv_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    df_all.to_csv(output_csv_path, index=False)

    print(f"Saved WS2 threshold comparison to : {output_csv_path}")

    return df_all

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Path to WS2 processed CSVs")
    parser.add_argument("--rc_values", nargs="+", type=float, required=True, help="List of RC threshold values")
    parser.add_argument("--ri_values", nargs="+", type=float, required=True, help="List of RI threshold values")
    parser.add_argument("--k", type=int, default=5, help="Number of folds for cross-validation")
    parser.add_argument("--output", type=str, default=None, help="Output CSV path for comparison results")

    args = parser.parse_args()

    # 総合評価推定に使用するモデル
    MODEL_NAME = "logistic_regression" 

    comparison_result = compare_thresholds_ws2(
        model_name=MODEL_NAME,
        dir_ws2=args.data_dir,
        RC_VALUES=args.rc_values,
        RI_VALUES=args.ri_values,
        K=args.k,
        output_csv_path = args.output
    )    