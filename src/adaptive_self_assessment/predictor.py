import numpy as np
import pandas as pd
from typing import List, Tuple, Dict

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from adaptive_self_assessment.spec import SPEC_WS1, SPEC_WS2, get_spec_cols

def predict_item_ws1(
        Ca: Dict[str, int],
        C: List[str],
        df_train: pd.DataFrame,
        random_state: int = 42
) -> Tuple[Dict[str, int], Dict[str, float]]:
    """
    1回分の学習者データを使って、未回答項目Cを項目ごとにロジスティック回帰で推定
    Parameters:
    ----------
        Ca: Dict[str, int]
            回答済みor補完済み項目
        C: List[str]
            残りの質問項目
        df_train: pd.DataFrame 
            予測に使う訓練データ
        random_state: int
            乱数シード
    Returns:
    -------
        preds: Dict[str, int]
            予測された質問回答結果
        confidences: Dict[str, float]
            各予測の信頼度
    """
    preds: Dict[str, int] = {}
    confidences: Dict[str, float] = {}

    # Caにあるキーのうち、df_trainに存在する列のみを使用
    ca_cols_exist: List[str] = [c for c in Ca.keys() if c in df_train.columns]

    # 各項目の予測
    for item in C:
        try:
            # 学習に使う特徴量の列
            x_cols: List[str] = [c for c in ca_cols_exist if c != item]
            # 使える特徴量がない or 訓練データに対象項目が存在しない場合はエラー
            if len(x_cols) == 0 or item not in df_train.columns:
                raise ValueError("No usable features or target column missing.")
            
            # 学習データの作成
            X_train = df_train[x_cols]
            y_train = df_train[item]

            # モデルの定義
            model = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(
                    max_iter=5000, 
                    class_weight="balanced",
                    random_state=random_state
                ))
            ])

            # モデルの学習
            model.fit(X_train, y_train)

            # 特徴量のうちCaにないものを確認
            missing_features = [c for c in x_cols if c not in Ca]
            if missing_features:
                raise ValueError(f"Missing features in Ca: {missing_features}")

            # チェック項目を推定
            # 予測用データの作成
            x_pred = pd.DataFrame([[Ca[c] for c in x_cols]], columns=x_cols)

            # 項目の予測
            proba = model.predict_proba(x_pred)[0]
            classes = model.named_steps["clf"].classes_
            pred_idx = int(np.argmax(proba))
            preds[item] = int(classes[pred_idx])
            confidences[item] = float(proba[pred_idx])
        except Exception as e:
            print(f"[predict_item_ws1 error] target={item}: {e}")
            preds[item] = 1
            confidences[item] = 0.5

    return preds, confidences

def predict_item_ws2(
        Pra: int, 
        Pca: Dict[str, int], 
        Ca: Dict[str, int], 
        C: List[str], 
        df_train: pd.DataFrame,
        random_state: int = 42
    ) -> Tuple[Dict[str, int], Dict[str, float]]:
    """
    2回分の学習者データを使って、未回答項目Cを項目ごとにロジスティック回帰で推定
    Parameters:
    ----------
        Pra: int
            過去の総合評価の値
        Pca: Dict[str, int]
            過去のチェックリスト結果
        Ca: Dict[str, int]
            回答済みor補完済み項目
        C: List
            残りの質問項目
        df_train: pd.DataFrame 
            予測に使う訓練データ
        random_state: int
            乱数シード
    Returns:
    -------
        preds: Dict[str, int]
            予測された質問回答結果
        confidences: Dict[str, float]
            各予測の信頼度
    """

    preds: Dict[str, int] = {}
    confidences: Dict[str, float] = {}

    # 過去の総合評価と過去のチェック項目列名を取得
    pra_col, pca_cols, _, _, _ = get_spec_cols(df_train, SPEC_WS2)

    # Caにあるキーのうち、df_trainに存在する列のみを使用
    ca_cols_exist = [c for c in Ca.keys() if c in df_train.columns]

    # 各項目の予測
    for item in C:
        try:
            # 学習に使う特徴量の列
            x_cols: List[str] = [pra_col] + pca_cols + [c for c in ca_cols_exist if c != item]
            if len(x_cols) == 0 or item not in df_train.columns:
                raise ValueError("No usable features or target column missing.")
            
            # 学習データの作成
            X_train = df_train[x_cols]
            y_train = df_train[item]

            # モデルの定義
            model = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(
                    max_iter=5000, 
                    class_weight="balanced",
                    random_state=random_state
                ))
            ])

            # モデルの学習
            model.fit(X_train, y_train)

            # 特徴量のうちPra, Pca, Caにないものを確認
            missing_features = []
            missing_features += [c for c in pca_cols if c not in Pca]
            missing_features += [c for c in x_cols if c not in Ca and c not in pca_cols and c != pra_col]
            if Pra is None or missing_features:
                raise ValueError(f"Missing features in Pra, Pca, Ca: {missing_features}")
            
            # チェック項目を推定
            # 予測用データの作成
            row_vals = []
            for c in x_cols:
                if c == pra_col:
                    row_vals.append(Pra)
                elif c in pca_cols:
                    row_vals.append(Pca[c])
                else:
                    row_vals.append(Ca[c])
            x_pred = pd.DataFrame([row_vals], columns=x_cols)

            # 項目の予測
            proba = model.predict_proba(x_pred)[0]
            classes = model.named_steps["clf"].classes_
            pred_idx = int(np.argmax(proba))
            preds[item] = int(classes[pred_idx])
            confidences[item] = float(proba[pred_idx])
        except Exception as e:
            print(f"[predict_item_ws2 error] target={item}: {e}")
            preds[item] = 1
            confidences[item] = 0.5

    return preds, confidences

def predict_overall_ws1(
        Ca: Dict[str, int],
        df_train: pd.DataFrame,
        model_name: str = "logistic_regression",
        random_state: int = 42
) -> Tuple[int, float]:
    """
    1回分の学習者データを使って、総合評価を選択されたモデルで推定
    Parameters:
    ----------
        Ca: Dict[str, int]
            回答済みor補完済み項目
        df_train: pd.DataFrame 
            予測に使う訓練データ
        model_name: str
            使用するモデル名 (現状"logistic_regression"のみ対応)
        random_state: int
            乱数シード
    Returns:
    -------
        pred: int
            予測された総合評価結果
        confidence: float
            予測の信頼度
    """
    # 列名の取得
    _, _, ca_cols, ra_col, _ = get_spec_cols(df_train, SPEC_WS1)

    # 学習データの作成
    for c in ca_cols + [ra_col]:
        if c not in df_train.columns:
            raise ValueError("Required columns missing in training data.")

    X_train = df_train[ca_cols]
    y_train = df_train[ra_col]

    # モデルの定義
    if model_name == "logistic_regression":
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=5000,
                class_weight="balanced",
                random_state=random_state
            ))
        ])
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    # モデルの学習
    model.fit(X_train, y_train)

    # 特徴量のうちCaにないものを確認
    missing_features = [c for c in ca_cols if c not in Ca]
    if missing_features:
        raise ValueError(f"Missing features in Ca: {missing_features}")
    # 総合評価を推定
    # 予測用データの作成
    x_pred = pd.DataFrame([[Ca[c] for c in ca_cols]], columns=ca_cols)

    # 総合評価の予測
    proba = model.predict_proba(x_pred)[0]
    classes = model.named_steps["clf"].classes_
    pred_idx = int(np.argmax(proba))
    pred = int(classes[pred_idx])
    confidence = float(proba[pred_idx])

    return pred, confidence

def predict_overall_ws2(
        Pra: int,
        Pca: Dict[str, int],
        Ca: Dict[str, int],
        df_train: pd.DataFrame,
        model_name: str = "logistic_regression",
        random_state: int = 42
) -> Tuple[int, float]:
    """
    2回分の学習者データを使って、総合評価を選択されたモデルで推定
    Parameters:
    ----------
        Pra: int
            過去の総合評価の値
        Pca: Dict[str, int]
            過去のチェックリスト結果
        Ca: Dict[str, int]
            回答済みor補完済み項目
        df_train: pd.DataFrame 
            予測に使う訓練データ
        model_name: str
            使用するモデル名 (現状"logistic_regression"のみ対応)
        random_state: int
            乱数シード
    Returns:
    -------
        pred: int
            予測された総合評価結果
        confidence: float
            予測の信頼度
    """
    # 列名の取得
    pra_col, pca_cols, ca_cols, ra_col, _ = get_spec_cols(df_train, SPEC_WS2)

    # 学習データの作成
    for c in [pra_col] + pca_cols + ca_cols + [ra_col]:
        if c not in df_train.columns:
            raise ValueError("Required columns missing in training data.")
    
    X_cols = [pra_col] + pca_cols + ca_cols
    X_train = df_train[X_cols]
    y_train = df_train[ra_col]

    if model_name == "logistic_regression":
        # モデルの定義
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=5000,
                class_weight="balanced",
                random_state=random_state
            ))
        ])
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    # モデルの学習
    model.fit(X_train, y_train)

    # 特徴量のうちPra, Pca, Caにないものを確認
    missing_features = []
    missing_features += [c for c in pca_cols if c not in Pca]
    missing_features += [c for c in ca_cols if c not in Ca]
    if Pra is None or missing_features:
        raise ValueError(f"Missing features in Pra, Pca, Ca: {missing_features}")

    # 総合評価を推定
    # 予測用データの作成
    row_vals = []
    for c in X_cols:
        if c == pra_col:
            row_vals.append(Pra)
        elif c in pca_cols:
            row_vals.append(Pca[c])
        else:
            row_vals.append(Ca[c])
    x_pred = pd.DataFrame([row_vals], columns=X_cols)

    # 総合評価の予測
    proba = model.predict_proba(x_pred)[0]
    classes = model.named_steps["clf"].classes_
    pred_idx = int(np.argmax(proba))
    pred = int(classes[pred_idx])
    confidence = float(proba[pred_idx])
    
    return pred, confidence