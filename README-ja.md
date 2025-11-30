# adaptive-self-assessment-sim

## 概要
本リポジトリは、**機械学習ベースの適応型自己評価（Adaptive Self-Assessment）システム**のシミュレーションフレームワークです。  
学習者の回答パターンに応じてチェックリスト項目の提示・補完を動的に行うことで、**すべての項目に回答した場合と同等の評価精度を維持しながら、回答項目数の削減を実現すること**を目的としています。

本フレームワークは以下２種類の自己評価に対応しています：
- **WS1 — 単一セッション自己評価**
- **WS2 — 2回分の自己評価（前回 + 今回）を活用した自己評価**

## 背景と目的
ルーブリックやチェックリストを用いた**間主観的評価（inter-subjective assessment）**は、学習者の自己理解や振り返りを促進し、継続的な成長を支援します。
一方で現場では、**評価項目が多いことによる回答負担・評価疲れ**が生じやすく、モチベーションや回答の質の低下が課題となっています。

そこで本プロジェクトでは、回答パターンに応じた項目選択と機械学習による回答補完を組み合わせることで、**「必要最低限の回答で、すべての項目を使った場合と同等の精度の総合評価を実現する」**  
ことを目指しています。

本システムは将来的に、教育現場・企業研修・自己評価ツールなどへの応用を想定しています。

## 適応型自己評価アルゴリズム
未回答項目の補完・総合評価の推定は、以下2つの信頼度閾値によって制御されます：

- **Rc：未回答項目の補完（completion）をするかどうかの判断基準**
- **Ri：最終的な総合評価を確定してよいかの判断基準**

### 疑似コード（Pseudo-code）
```pseudo
Inputs:
    I      : チェックリスト項目の全集合
    Pra    : 前回の総合評価（WS2 の場合のみ、それ以外は null）
    Pca    : 前回のチェックリスト回答（WS2 の場合のみ、それ以外は null）
    Rc     : 項目補完の信頼度閾値
    Ri     : 最終評価確定の信頼度閾値

Outputs:
    pred_R : 予測された総合評価
    Ca     : 回答または補完が確定した項目集合

Initialization:
    C  = I        # 未回答項目集合
    Ca = {}       # 回答または補完が反映された項目集合

Procedure:
    while C is not empty:
        # 次に質問する項目を選択（現段階の実装はランダム）
        ci = select_next_item(C)
        ans = query_response(ci)

        # 回答を確定リストに追加
        Ca[ci] = ans
        remove ci from C

        # 現時点の回答から、未回答項目の推定
        (pred_C, conf_C) = completion_model.predict(C, Ca, Pra, Pca)

        # 信頼度がRcを超える項目は回答済みとして扱う
        for each item j in C:
            if conf_C[j] ≥ Rc:
                Ca[j] = pred_C[j]
                remove j from C

    # すべてのチェックリスト項目項目の値が確定したら、総合評価を推定
    (pred_R, conf_R) = overall_estimator.predict(Ca, Pra, Pca)

    if conf_R < Ri:
    # 信頼度がRi未満の場合は、人による評価を要求
        pred_R = request_manual_rating()    

Return pred_R, Ca
```

## 環境構築
### セットアップ
```bash
git clone https://github.com/yuta-wakui/adaptive-self-assessment-sim.git
cd adaptive-self-assessment-sim
```

### 仮想環境（推奨）
```bash
python -m venv venv
source venv/bin/activate # Windows: venv\Scripts\activate
```
### 依存関係のインストール
本プロジェクトで必要となる Python パッケージをインストールします:
```bash
pip install -r requirements.txt
```
または、開発モードでインストールする場合:
```bash
pip install -e .
```
## 使い方
### シミュレーション
#### Command-line Arguments
以下の引数はスクリプト実行時に指定できます:

| Argument     | Meaning                                           | Example                                                 |
| ------------ | ------------------------------------------------- | ------------------------------------------------------- |
| `--data_dir` | 使用するデータセットのディレクトリ (WS1 or WS2)     | `data/sample/ws1`                                       |
| `--rc`       | 項目補完（completion）を実施するための信頼度閾値    | 0.80                                                  |
| `--ri`       | 総合評価を確定してよいかを判断するための信頼度閾値 | 0.70                                                  |
| `--k`        | 交差検証の分割数                  | 5                                                    |
| `--output`   | シミュレーション結果の保存先パス                   | `outputs/results_csv/ws1/sim_results/rc0p80_ri0p70.csv` |

#### WS1（単一セッション）
単一セッションの自己評価データに対して適応型シミュレーションを実施します：
```bash
python scripts/run_ws1_sim.py \
  --data_dir data/sample/ws1 \
  --rc 0.80 \
  --ri 0.70 \
  --k 5 \
  --output outputs/results_csv/ws1/sim_results/ws1_results_rc0p80_ri0p70_date.csv
```
#### WS2（２回分の自己評価）
２回分の自己評価データに対して適応型シミュレーションを実施します：
```bash
python scripts/run_ws2_sim.py \
  --data_dir data/sample/ws2 \
  --rc 0.80 \
  --ri 0.70 \
  --k 5 \
  --output outputs/results_csv/ws2/sim_results/ws2_results_rc0p80_ri0p70_date.csv
```
### 信頼度閾値グリッドリサーチ (RC × RI)
複数の閾値の組み合わせに対してグリッドリサーチでシミュレーションを実行します：

#### Command-line Arguments
以下の引数はスクリプト実行時に指定できます:

| Argument     | Meaning                                           | Example                                                 |
| ------------ | ------------------------------------------------- | ------------------------------------------------------- |
| `--data_dir` | 使用するデータセットのディレクトリ (WS1 or WS2)     | `data/sample/ws1`                                       |
| `--rc_values`       | 項目補完（completion）の信頼度閾値のリスト    | 0.7 0.8 0.9                                    |
| `--ri_values`       | 総合評価推定の信頼度閾値のリスト | 0.6 0.7 0.8                                                 |
| `--k`        | クロスバリデーションの分割数                  | 5                                                   |
| `--output`   | シミュレーション結果（集約結果）を保存する出力ファイルパス                   | outputs/results_csv/ws1/cmp_thresholds/cmp_results.csv |

#### WS1
```bash
python scripts/compare_thresholds_ws1.py \
  --data_dir data/sample/ws1 \
  --rc_values 0.7 0.8 0.9 \
  --ri_values 0.6 0.7 0.8 \
  --k 5 \
  --output outputs/results_csv/ws1/cmp_thresholds/ws1_cmp_rc_ri_grid.csv
```
#### WS2
```bash
python scripts/compare_thresholds_ws2.py \
  --data_dir data/sample/ws2 \
  --rc_values 0.7 0.8 0.9 \
  --ri_values 0.6 0.7 0.8 \
  --k 5 \
  --output outputs/results_csv/ws2/cmp_thresholds/ws2_cmp_rc_ri_grid.csv
```

### ライブラリとしての利用例（Python API）
専用のスクリプトを使わずに、外部のPythonコードから直接シミュレーションを実行することも可能です。：
```python
import pandas as pd
from adaptive_self_assessment import run_ws1_simulation

# データの読み込み
train_df = pd.read_csv("data/sample/ws1/ws1_data_sample.csv")

# データを訓練用とテスト用に分割（例：80%訓練 / 20%テスト）
test_df = train_df.sample(frac=0.2, random_state=42)
train_df = train_df.drop(test_df.index)

# シミュレーションの実行
results = run_ws1_simulation(
    RC_THRESHOLD=0.80,
    RI_THRESHOLD=0.70,
    train_df=train_df,
    test_df=test_df,
    model_type="logistic_regression",
)

print(results)
```

## テスト
単体テスト・結合テストを実行:
```bash
pytest -s
```
テストは以下に配置されています:
```bash
tests/
```

## ライセンス
MIT License