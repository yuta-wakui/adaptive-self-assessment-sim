# %% [markdown]
# # 複数の補間アルゴリズムの比較
# %%   [markdown]
# ## 概要
# 使用データ：1回分の自己評価データ（合成データ）

# 目的変数：各チェック項目の自己評価結果
#
# 特徴量：その他のチェック項目の自己評価結果
#
# 比較対象アルゴリズム：Logistic Regression, Random Forest, Gradient Boosting, LightGBM, k-NN, Deep Learning, SVM
#
# 評価指標：Accuracy, f1_macro
#
# 評価方法：k-fold cross validation (k=5)
# %% [markdown]
# ## 準備
# %%
# ライブラリーのインポート
import pandas as pd 
import numpy as np
# %%
# データファイルのパス取得
import os 
for dirname, _, filenames in os.walk('../data/processed/ws1_synthetic_240531_processed'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# %%
# データの読み込み
df_info = pd.read_csv('../data/processed/ws1_synthetic_240531_processed/1_syntheticdata_information.csv')
df_thinking = pd.read_csv('../data/processed/ws1_synthetic_240531_processed/2_syntheticdata_thinking.csv')
df_writing = pd.read_csv('../data/processed/ws1_synthetic_240531_processed/3_syntheticdata_writing.csv')
df_presen = pd.read_csv('../data/processed/ws1_synthetic_240531_processed/4_syntheticdata_presen.csv')
df_quant = pd.read_csv('../data/processed/ws1_synthetic_240531_processed/5_syntheticdata_quant.csv')
df_learning = pd.read_csv('../data/processed/ws1_synthetic_240531_processed/6_syntheticdata_learning.csv')
df_act = pd.read_csv('../data/processed/ws1_synthetic_240531_processed/7_syntheticdata_act.csv')
df_teamwork = pd.read_csv('../data/processed/ws1_synthetic_240531_processed/8_syntheticdata_teamwork.csv')

# %%
df_info.head(3)
# %%
# スキル名とデータフレームを対応付けて辞書で保存
skills = ['info', 'thinking', 'writing', 'presen', 'quant', 'learning', 'act', 'teamwork']
dfs = [df_info, df_thinking, df_writing, df_presen, df_quant, df_learning, df_act, df_teamwork]

# この際、使用しない列を削除しておく
drop_columns = ['ID', 'assessment-result', 'reflection-length']
for df in dfs:
    df.drop(columns=drop_columns, inplace=True)

df_dict = dict(zip(skills, dfs))
# %%
# モデルの定義
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from lightgbm import LGBMClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

models = {
    "Logistic Regression": make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=5000,
            random_state=42
        )
    ),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "LightGBM": LGBMClassifier(random_state=42, n_jobs=1, verbose=-1),
    "MLP": make_pipeline(
        StandardScaler(),
        MLPClassifier(random_state=42, max_iter=5000)
    ),
    "k-NN": make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(),
    ),
    "SVM": make_pipeline(
        StandardScaler(),
        SVC(random_state=42)
    )
}
# %%
# 実行する能力を指定指定
run_skills = ['info', 'thinking', 'writing', 'presen', 'quant', 'learning', 'act', 'teamwork']
# %%
# 分割数5のクロスバリデーションで学習と評価を実施
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results = []

# 各スキルごとに処理
for skill, df in df_dict.items():

    if skill not in run_skills:
        continue

    print(f'Processing skill: {skill}')

    # 各チェック項目ごとに処理
    for target in df.columns:
        print(f'Processing target: {target} of {skill}')

        # 目的変数と特徴量に分割
        X = df.drop(columns=[target]).copy()
        y = df[target]

        # 目的変数が1クラスのみの場合はスキップ
        if y.unique().size < 2:
            print(f'Skill: {skill}, Target: {target} is not multiple')
            continue

        # 各モデルごとにクロスバリデーションを実施
        for fold_id, (train_index, valid_index) in enumerate(cv.split(X, y)):
            print(f'Processing fold: {fold_id+1} in {target} of {skill}')
            X_tr = X.iloc[train_index]
            y_tr = y.iloc[train_index]
            X_val = X.iloc[valid_index]
            y_val = y.iloc[valid_index]

            # 各モデルごとに学習と評価
            for model_name, model in models.items():
                m = clone(model)

                try: 
                    m.fit(X_tr, y_tr)
                    y_pred = m.predict(X_val)

                    acc = accuracy_score(y_val, y_pred)
                    f1_macro = f1_score(y_val, y_pred, average='macro')

                    results.append({
                        'skill': skill,
                        'target': target,
                        'fold_id': fold_id+1,
                        'model': model_name,
                        'accuracy': acc,
                        'f1_macro': f1_macro
                    })
                
                except Exception as e:
                    print(f"Error training {model_name} on {target} of {skill}, fold {fold_id}: {e}")
                    
                    results.append({
                        'skill': skill,
                        'target': target,
                        'fold_id': fold_id+1,
                        'model': model_name,
                        'accuracy': np.nan,
                        'f1_macro': np.nan
                    })
                    continue

# 結果の保存
results_df = pd.DataFrame(results)
results_df
# %%
# 能力ごとに各モデルのの平均精度を比較

# 能力ごとにグループ化して各モデルの平均精度を計算
summary_by_skill = (
    results_df
    .groupby(['skill', 'model'])
    .agg({'accuracy': 'mean', 'f1_macro': 'mean'})
    .sort_values(by=['skill', 'accuracy'], ascending=[True, False])
)

print("能力ごとの各モデルの平均精度")
display(summary_by_skill)
# %%
# 能力とチェック項目ごとにfold平均を計算

# 能力とチェック項目でグループ化して各foldの平均精度を計算計算
summary_by_item = (
    results_df
    .groupby(['skill', 'target', 'model'])
    .agg({'accuracy': 'mean', 'f1_macro': 'mean'})
    .sort_values(['skill', 'target', 'accuracy'], ascending=[True, True, False])
)

print("能力、項目ごとの各モデルの平均精度")
display(summary_by_item)
# %%
# summary_by_skillとsummary_by_itemを表形式で変換

# accuracyとf1_macroをそれぞれpivotして表形式に変換
acc_table = (
    summary_by_skill
    .reset_index()
    .pivot(index="skill", columns="model", values="accuracy")
    .round(3)
)
f1_table = (
    summary_by_skill
    .reset_index()
    .pivot(index="skill", columns="model", values="f1_macro")
    .round(3)
)

# 列（モデル）の並び順を統一
model_order = ["Logistic Regression", "Random Forest", "Gradient Boosting", "LightGBM", "MLP", "k-NN", "SVM"]
acc_table = acc_table[model_order]
f1_table = f1_table[model_order]

# 行（スキル）の並び順を統一
skill_order = ["info", "thinking", "writing", "presen", "quant", "learning", "act", "teamwork"]
acc_table = acc_table.reindex(skill_order)
f1_table = f1_table.reindex(skill_order)
# %%
# %% summary_by_skill / summary_by_item / acc_table / f1_table を CSV で保存

# 出力先ディレクトリ
save_dir = "../outputs/results_csv/ws1/compare_completion_algorithms/"
os.makedirs(save_dir, exist_ok=True)

# 1. 能力、項目、fold、モデルごとの結果
results_path = os.path.join(save_dir, "ws1_results_cv_detail.csv")
results_df.to_csv(results_path, index=False)
print(f"Saved CV results → {results_path}")

# 2. 能力、モデルごとの平均精度
summary_skill_path = os.path.join(save_dir, "ws1_summary_by_skill.csv")
summary_by_skill.to_csv(summary_skill_path)
print(f"Saved summary_by_skill → {summary_skill_path}")

# 3. 能力、項目、モデルごとの平均精度
summary_item_path = os.path.join(save_dir, "ws1_summary_by_item.csv")
summary_by_item.to_csv(summary_item_path)
print(f"Saved summary_by_item → {summary_item_path}")

# 4. ピボット済みの表（論文用の表形式）
acc_path = os.path.join(save_dir, "ws1_accuracy_table_by_skill.csv")
f1_path = os.path.join(save_dir, "ws1_f1macro_table_by_skill.csv")

acc_table.to_csv(acc_path)
f1_table.to_csv(f1_path)

print("Saved pivot tables:")
print(f"- {acc_path}")
print(f"- {f1_path}")
# %%