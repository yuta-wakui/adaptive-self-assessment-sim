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
    1回分の学習者データを使って、未回答のチェック項目回答結果の推定を行う関数
    Parameters:
    ----------
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

    # Caにあるキーのうち、df_trainに存在する列のみを使用
    ca_cols_exist = [c for c in Ca.keys() if c in df_train.columns]

    # 各項目の予測
    for item in C:
        try:
            # 学習に使う特徴量の列
            x_cols = [c for c in ca_cols_exist if c != item]
            if len(x_cols) == 0 or item not in df_train.columns:
                raise ValueError("No usable features or target column missing.")
            
            # 学習データの作成
            X_train = df_train[x_cols]
            y_train = df_train[item]

            # 列名を文字列型に変換 
            X_train_columns = X_train.columns.astype(str)
            x_cols = list(X_train_columns)

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

            # チェック項目を推定
            x_pred = pd.DataFrame([[Ca.get(c, -1) for c in x_cols]], columns=x_cols)
            x_pred.columns = x_pred.columns.astype(str)

            proba = model.predict_proba(x_pred)[0]
            classes = model.named_steps["clf"].classes_
            pred_idx = int(np.argmax(proba))
            preds[item] = int(classes[pred_idx])
            confidences[item] = float(proba[pred_idx])
        except Exception as e:
            print(f"[predict_items error] {item}: {e}")
            preds[item] = 1
            confidences[item] = 0.5

    return preds, confidences

def predict_item_ws2(Pra: int, Pca: Dict[str, int], Ca: Dict[str, int], C: List[str], df_train: pd.DataFrame, spec: Dict) -> Tuple[Dict[str, int], Dict[str, float]]:
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
        spec: Dict
            データの列名情報
    Returns:
    -------
        preds: Dict[str, int]
            予測された質問回答結果
        confidences: Dict[str, float]
            各予測の信頼度
    """

    preds: Dict[str, int] = {}
    confidences: Dict[str, float] = {}

    # 列仕様の取得
    pra_col, pca_cols, _, _, _ = get_spec_cols(df_train, spec)

    # 特徴量の作成
    features = [pra_col] + pca_cols + list(Ca.keys())

    # 各項目の予測
    for item in C:
        try:
            X_train = df_train[features]
            y_train = df_train[item]
            X_train.columns = X_train.columns.astype(str)
            x_pred = pd.DataFrame([[Pra] + list(Pca.values()) + list(Ca.values())], columns=features)
            x_pred.columns = x_pred.columns.astype(str)     

            # モデルの定義
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=5000, random_state=42))
            ])

            # モデルの学習
            pipe.fit(X_train, y_train)

            # チェック項目を推定
            proba = pipe.predict_proba(x_pred)[0]
            pred_class = int(pipe.classes_[np.argmax(proba)])
            confidence = np.max(proba)
            preds[item] = pred_class
            confidences[item] = confidence
        except Exception as e:
            print(f"[predict_items error] {item}: {e}")
            preds[item] = 1
            confidences[item] = 0.5

    return preds, confidences


# def predict_assessment(Pra, Pca, Ca, train_df, target_id, model_type="logistic"):
#     """
#     総合評価の予測を行う関数
#     Args:
#         Pra (float): 過去の総合評価
#         Pca (dict): 過去のチェックリスト結果
#         Ca (dict): 補完された項目
#         train_df (pd.DataFrame): 予測に使うデータ
#         target_id (int): ターゲットユーザーのID
#         model_type (str): 使用するモデルのタイプ（"logistic", "svm", "rf"）
#     Returns:
#         pred_class (int): 予測された総合評価
#         confidence (float): 予測の信頼度
#     """

#     try: 
#         features, target = _create_feature_df(
#             train_df,
#             Pra_col="w3-assessment-result",
#             Pca_cols=[k for k in train_df.columns if k.startswith("w3-") and k not in ["w3-assessment-result"]],
#             Ca_cols=Ca.keys(),
#             target_col="w4-assessment-result"
#         )
        
#         # モデル選
#         if model_type == "logistic":
#             model = LogisticRegression(max_iter=5000, class_weight="balanced")
#         elif model_type == "svm":
#             model = SVC(probability=True, random_state=42)  # probability=True で predict_proba 利用
#         elif model_type == "gd":
#             model = GradientBoostingClassifier(random_state=42)
#         elif model_type == "rf":
#             model = RandomForestClassifier(random_state=42)
#         else:
#             raise ValueError("Invalid model_type. Choose 'logistic', 'svm', or 'rf'.")

#         # モデルの学習
#         model.fit(features, target)

#         # 総合評価を推定
#         x_pred = pd.DataFrame([[Pra] + list(Pca.values()) + list(Ca.values())], columns=features.columns)
        
#         # 予測と信頼度の取得
#         proba = model.predict_proba(x_pred)[0]
#         pred_class = int(model.classes_[np.argmax(proba)])
#         confidence = float(np.max(proba))
#         return pred_class, confidence


#     except Exception as e:
#         print(f"[Error] predict_assessment failed: {e}")
#         return 2, 0.5
