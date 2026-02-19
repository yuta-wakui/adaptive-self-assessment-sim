# adaptive-self-assessment-sim
少ない質問数で評価精度を保つ ー 機械学習による適応型自己評価シミュレーションのフレームワーク

大量の質問に対する回答負担を軽減するため、
回答パターンに応じて質問を選択し、
未回答項目を機械学習で補完して評価結果を推定します。

本フレームワークにより、「すべての質問に答えなくても同じ評価ができるか？」を検証できます。

教育評価・アンケート設計・自己評価システムの研究および設計検討に利用できます。

## デモ
サンプルデータでシミュレーションを実行できます。
```bash
python scripts/run_adaptive_sim.py --config configs/config.yaml
```

実行例：
```text
=== WS2 Simulation Started ===
input: data/sample/ws2/ws2_data_sample.csv
skill: sample
model(overall): logistic_regression
question_selection.strategy: random
thresholds: RC=0.8, RI=0.7
cv: folds=2, stratified=True, seed=42
Dropped ignore_items columns: ['past_reflection_length', 'current_reflection_length']

==== WS2: sample ====
csv: data/sample/ws2/ws2_data_sample.csv
n_rows: 100, n_cols: 33
---- Fold 1/2 ----
---- Fold 2/2 ----

=== WS2 Simulation Results ===
average_answered_questions: 3.88 / 10
reduction_rate: 61.2%
accuracy_all: 0.6400
f1_macro_all: 0.4566

Saved results to: outputs/results/ws2/sim_results/20260219_194536/ws2_results_rc0p8_ri0p7_sample.csv
Saved fold results to: outputs/results/ws2/sim_results/20260219_194536/ws2_fold_results_rc0p8_ri0p7_sample.csv
Saved user logs to: outputs/logs/ws2/20260219_194536/ws2_user_logs_rc0p8_ri0p7_sample.csv

=== WS2 Simulation Completed ===
```

## 特徴
- 機械学習による適応型自己評価アルゴリズムのシミュレーション環境
- 単回評価（WS1）と継続評価（WS2；過去＋現在）に対応
- 回答パターンに応じた質問選択戦略に対応（現在はランダム選択のみ実装）
- 未回答項目を信頼度に基づいて逐次補完
- Cross-validationによる再現性のある性能評価
- YAML設定のみで実験条件・モデル・閾値を変更可能
- 全項目回答（非適応型）との比較により質問削減率と精度のトレードを検証可能

## インストール

### 前提条件
- Python 3.10 以上
- pip

### インストール手順
```bash
# 1. リポジトリをクローン
git clone https://github.com/yuta-wakui/adaptive-self-assessment-sim.git

# 2. ディレクトリ移動
cd adaptive-self-assessment-sim

# 3. 仮想環境の作成
python -m venv venv

# 4. 仮想環境の有効化
source venv/bin/activate # macOS / Linux
venv\Scripts\activate # Windows

# 5. パッケージをインストール
pip install -e .
```

### インストール確認
```bash
python -c "import adaptive_self_assessment; print('OK')"
```

### （任意）テスト用依存関係
テストを実行する場合：
```bash
pip install pytest
pytest
```

## 使用方法

### 1. 設定ファイルを確認
サンプル設定ファイルをそのまま使用できます。

`configs/config.yaml`

### 2. 適応型自己評価シミュレーションの実行
```bash
python scripts/run_adaptive_sim.py --config configs/config.yaml
```

実行するとクロスバリデーションによるシミュレーションが行われ、
結果がコンソールに表示されます。

出力例：
```text
=== WS2 Simulation Results ===
average_answered_questions: 3.88 / 10
reduction_rate: 61%
accuracy_all: 0.6400
f1_macro_all: 0.4566
```

また、結果ファイルは以下に保存されます：
```bash
outputs/results/
outputs/logs/
```

### 3. 非適応（ベースライン）との比較
```bash
python scripts/run_non_adaptive_sim.py --config configs/config.yaml
```

これにより「全項目回答」との性能比較が可能です。

出力例：
```text
=== WS2 Non-Adaptive Results ===
Use all questions:
accuracy_all: 0.6800
f1_macro_all: 0.5003
```

## シミュレーション設定
### データセット配置
本フレームワークではCSV形式の評価データを使用します。

データは任意の場所に配置できますが、管理しやすさの観点から`data/`ディレクトリに置くことを推奨します。

### 実験設定（YAMLファイル）
シミュレーションの挙動はYAMLファイルで制御します。

まずはテンプレートをコピーして使用してください。（[configs/template.yaml](configs/template.yaml)）

各パラメータの役割は以下の通りです。

### configパラメータ一覧

| セクション                 | パラメータ                 | 型            | 説明                                       |
| --------------------- | --------------------- | ------------ | ---------------------------------------- |
| `mode`                | `ws1 / ws2`           | string       | 使用するデータ形式を指定（`ws1`: 単回評価 / `ws2`: 継続評価）  |
| `data.common`         | `skill_name`          | string       | 評価対象スキル名（ログ・出力に使用）                       |
|                       | `id_col`              | string       | 評価対象者の識別子列名                              |
|                       | `ignore_items`        | list[string] | 評価に使用しない列名のリスト（任意）                       |
| `data.ws1`            | `input_path`          | string       | WS1用CSVデータのパス                            |
|                       | `overall_col`         | string       | 総合評価の正解ラベル列                              |
|                       | `item_cols`           | list[string] | 質問項目列（カテゴリ値または0/1を想定）                    |
| `data.ws2`            | `input_path`          | string       | WS2用CSVデータのパス                            |
|                       | `past_overall_col`    | string       | 過去の総合評価列                                 |
|                       | `past_item_cols`      | list[string] | 過去の質問項目列（カテゴリ値または0/1）                    |
|                       | `current_overall_col` | string       | 今回の総合評価列                                 |
|                       | `current_item_cols`   | list[string] | 今回の質問項目列（カテゴリ値または0/1）                    |
| `model.item_model`    | `type`                | string       | 未回答項目補完モデル（現在は `logistic_regression` のみ） |
|                       | `params`              | dict         | モデルパラメータ（任意）                             |
| `model.overall_model` | `type`                | string       | 総合評価推定モデル（現在は `logistic_regression` のみ）  |
|                       | `params`              | dict         | モデルパラメータ（任意）                             |
| `thresholds`          | `RC`                  | float        | 未回答項目補完の信頼度閾値                             |
|                       | `RI`                  | float        | 総合評価を確定する信頼度閾値                           |
| `cv`                  | `folds`               | int          | クロスバリデーション分割数                            |
|                       | `stratified`          | bool         | 層化分割を行うか                                 |
|                       | `random_seed`         | int          | 乱数シード                                    |
| `question_selection`  | `strategy`            | string       | 質問選択戦略（現在は `"random"` のみ）                |
| `results`             | `save_csv`            | bool         | 結果CSVを保存するか                              |
|                       | `output_dir`          | string       | 出力先ディレクトリ                                |
|                       | `timestamped`         | bool         | タイムスタンプ付きフォルダを作成するか                      |
|                       | `filename_suffix`     | string       | 出力ファイル名のサフィックス                           |
|                       | `save_fold_results`   | bool         | 各foldの結果を保存するか                           |
| `logging`             | `save_logs`           | bool         | 個別ログを保存するか                               |
|                       | `log_dir`             | string       | ログ保存先ディレクトリ                              |
|                       | `timestamped`         | bool         | タイムスタンプ付きフォルダを作成するか                      |

## テスト
本リポジトリでは、主要コンポーネントのユニットテストと、サンプルデータを用いた統合テストを提供します。
### 実行方法
```bash
pytest 
```

標準出力も含めて確認したい場合：
```bash
pytest -s 
```

特定のテストだけ実行したい場合：
```bash
pytest tests/unit/test_selector.py -s
pytest tests/integration/test_sim.py -s
```

## ライセンス
MIT License