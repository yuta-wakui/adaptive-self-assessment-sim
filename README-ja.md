# adaptive-self-assessment-sim

## 概要
本リポジトリは、**機械学習に基づく適応型自己評価（Adaptive Self-Assessment）システム**のシミュレーションフレームワークを提供します。  
回答パターンに応じてチェックリスト項目の提示・補完を動的に行うことで、**全項目を使用した場合と同等の評価精度を維持しつつ、回答項目数の削減を実現すること**を目的としています。

本フレームワークは以下の 2 つの自己評価形態に対応しています：
- **WS1 — 単一セッション自己評価**
- **WS2 — 2 回分の自己評価（前回 + 今回）を活用した自己評価**

---

## 背景と目的
ルーブリックやチェックリストによる**間主観的評価（inter-subjective assessment**は、学習者の自己理解を促し、継続的な成長を支援する上で有効です。  
しかし教育の現場では、**評価項目の多さによる評価負担・評価疲れ**が大きな課題となり、モチベーションや回答の質の低下を引き起こします。

そこで本プロジェクトでは、自己評価者の回答パターンに応じて適応的に提示項目を選択し、
**「必要最低限の回答で、評価精度を維持する」**  
ことを目指した適応型自己評価アルゴリズムのシミュレーションを行います。

本システムは将来的に、教育現場・研修・自己評価ツールなどへの応用を想定しています。

---

## 適応型自己評価アルゴリズム
未回答項目の補完・総合評価の予測の精度評価は、以下2つの信頼度閾値によって制御されます：

| 記号 | 役割 |
|------|-----|
| **Rc** | 未回答項目の補完（completion）を実施するための信頼度閾値 |
| **Ri** | 最終的な総合評価を確定するための信頼度閾値 |
|------|-----|

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
        ci = select_next_item(C)        # （現段階ではランダム選択）
        ans = query_response(ci)
        Ca[ci] = ans
        remove ci from C

        (pred_C, conf_C) = completion_model.predict(C, Ca, Pra, Pca)

        for each item j in C:
            if conf_C[j] ≥ Rc:
                Ca[j] = pred_C[j]
                remove j from C

    (pred_R, conf_R) = overall_estimator.predict(Ca, Pra, Pca)

    if conf_R < Ri:
        pred_R = request_manual_rating()    # 信頼度不足の場合は追加の評価が必要

Return pred_R, Ca
```

## Requirement
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
install all required dependencies:
```bash
pip install -r requirements.txt
```
Or install via the package configuration:
```bash
pip install -e .
```
## 使い方
### WS1（単一セッション）
```bash
python scripts/run_ws1_sim.py \
  --data_dir data/sample/ws1 \
  --rc 0.80 \
  --ri 0.70 \
  --k 5 \
  --output outputs/results_csv/ws1/sim_results/ws1_results_rc0p80_ri0p70_date.csv
```
### WS2（２回分の自己評価）
```bash
python scripts/run_ws2_sim.py \
  --data_dir data/sample/ws2 \
  --rc 0.80 \
  --ri 0.70 \
  --k 5 \
  --output outputs/results_csv/ws2/sim_results/ws2_results_rc0p80_ri0p70_date.csv
```
### 信頼度閾値グリッドリサーチ (RC × RI)
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