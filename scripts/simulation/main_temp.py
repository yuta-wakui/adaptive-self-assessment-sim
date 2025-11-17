import pandas as pd
import numpy as np
import os
from sklearn.model_selection import StratifiedKFold
from adaptive_self_assessment.adaptive_self_assessment.selector import set_selector_seed
from simulation.base_model import run_base_model
from simulation.simulation import run_simulation

set_selector_seed(42)  # セレクタの乱数シードを固定

# パスと閾値の設定
model_name = "logistic"  # 総合評価推定に使用するモデルのタイプ
data_dir = "../../data/processed/w2-synthetic_20250326_1300_processed" # 対象データのディレクトリ
output_csv = f"../../outputs/results_csv/adaptive_simulation_{model_name}.csv" # 結果保存先
RC_THRESHOLD = 0.85 # チェック項目の補完の信頼度閾値
RI_THRESHOLD = 0.75 # 総合評価の信頼度閾値

# 交差検証の分割数
K = 5

# 対象能力
skills = {
    1: "information",
    2: "thinking",
    3: "writing",
    4: "presentation",
    5: "quant",
    6: "learning",
    7: "act",
    8: "teamwork",
}

# 結果格納用リスト
results_all = []

# 各能力でシミュレーション
for skill_num, skill_name in skills.items():
    print(f"==== {skill_name}のシミュレーションを開始 ====")
    print(f"使用するモデル: {model_name}")

    # データの読み込み
    csv_path = os.path.join(data_dir, f"ws2_{skill_num}_{skill_name}_1300_adjusted.csv")
    df = pd.read_csv(csv_path)

    # # データ数が少ないクラスは削除
    # label_counts = df["w4-assessment-result"].value_counts()
    # valid_labels = label_counts[label_counts >= 100].index

    # # もともとのデータ件数とクラス数
    # original_count = len(df)
    # original_classes = label_counts.shape[0]

    # # フィルタリング
    # df = df[df["w4-assessment-result"].isin(valid_labels)].copy()

    # # 削除件数を計算
    # filtered_count = len(df)
    # filtered_classes = valid_labels.shape[0]
    # num_removed = original_count - filtered_count
    # num_classes_removed = original_classes - filtered_classes

    # # 削除があったら表示
    # if num_removed > 0:
    #     print(f"{num_removed}件のデータと{num_classes_removed}クラスが削除されました。")

    # クラスが1種のみの場合はスキップ
    if df["w4-assessment-result"].nunique() < 2:
        print(f"クラスが1種のみのため {skill_name} はスキップします。")
        continue

    # StratifiedKFoldの設定
    y = df["w4-assessment-result"].values
    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=42)

    # ベースモデル（全項目使用）の評価
    base_acc, base_std = run_base_model(
        df,
        model_type=model_name,
        logs_path=f"../logs/base_model_results_{skill_name}_{model_name}.csv",
        cv_splits=K
    )

    print(f"==== 能力{skill_name}のベースモデル評価 ====")
    print(f"平均正解率: {base_acc}")
    print(f"標準偏差: {base_std}")

    # %表記に変換
    base_acc_pct = round(base_acc * 100, 4)
    base_std_pct = round(base_std * 100, 4)

    # 結果保存用リスト
    summary_list = []

    # 各foldで適応型自己評価のシミュレーションを実行
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(df, y)):
        print(f"==== Fold {fold_idx + 1} ====")

        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]

        # 適応型自己評価のシミュレーショ
        summary, results = run_simulation(
            RC_THRESHOLD=RC_THRESHOLD,
            RI_THRESHOLD=RI_THRESHOLD,
            model_type=model_name,
            train_df=train_df,
            test_df=test_df,
            logs_path=f"../logs/simulation_results_{skill_name}_{model_name}.csv"
        )
        summary_list.append(summary)

    # DataFrameに変換
    df_simu = pd.DataFrame(summary_list)

    # 各モデルの平均・標準偏差を計算
    mean, std = df_simu.mean(numeric_only=True).round(4), df_simu.std(numeric_only=True).round(4)

    # 各モデルの要約結果を保存用リストに追加
    for sim_name, mean, std in zip(
        ["adaptive"],[mean],[std]
    ):
        row = {
            "skill": skill_name,
            "model": sim_name,
            "total_questions": len([col for col in df.columns if col.startswith("w4-") and col not in ["w4-assessment-result", "w4-reflection-length"]]),
            "base_accuracy": base_acc_pct,
            "base_accuracy_std": base_std_pct,
            **mean.to_dict(),
            **{f"{k}_std": v for k, v in std.items()}
        }
        results_all.append(row)

# 結果を保存
results_df = pd.DataFrame(results_all)
os.makedirs("logs", exist_ok=True)
results_df.to_csv(output_csv, index=False)
print(f"\n 全能力のシミュレーション結果を {output_csv} に保存しました。")