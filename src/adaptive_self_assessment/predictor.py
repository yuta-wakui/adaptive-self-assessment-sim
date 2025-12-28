import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def _get_model_config(cfg: Dict[str, Any], kind: str) -> Tuple[str, Dict[str, Any]]:
    """
    設定ファイルからモデル設定を読み込む内部関数
    Parameters:
    ----------
        cfg: Dict[str, Any]
            設定辞書
        kind: str
            モデル種類（"item" or "overall"）
    Returns:
    -------
        mode_type: str
            モデルの種類（例："logistic_regression"）
        params: Dict[str, Any]
            モデルのパラメータ設定
    """
    if kind not in ["item", "overall"]:
        raise ValueError(f"Unsupported model kind: {kind}")

    model_cfg = cfg.get("model", {})

    sub_cfg = model_cfg.get(kind, {})
    model_type = sub_cfg.get("type", "logistic_regression")
    params = sub_cfg.get("params", {})

    return model_type, params

def _build_model(model_name: str, params: Dict[str, Any], random_state: int = 42) -> Pipeline:
    """
    モデルタイプとパラメーターからsklearnのPipelineモデルを構築する内部関数
    Parameters:
    ----------
        model_name: str
            モデルの種類（例："logistic_regression"）
        params: Dict[str, Any]
            モデルのパラメータ設定
        random_state: int
            乱数シード
    Returns:
    -------
        model: Pipeline
            構築されたモデルのパイプライン
    """  
    if model_name == "logistic_regression":
        base_params = {
            "max_iter": 5000,
            "class_weight": "balanced",
            "random_state": random_state,
        }
        base_params.update(params) # パラメータの上書き
        clf = LogisticRegression(**base_params)

        model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", clf)
        ])
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")
    
    return model

def predict_item_ws1(
        Ca: Dict[str, int],
        C: List[str],
        df_train: pd.DataFrame,
        cfg: Dict[str, Any],
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
        cfg: Dict[str, Any]
            設定辞書
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

    # configがNoneの場合
    if cfg is None:
        raise ValueError("cfg must be provided.")

    # configからモデル設定を取得
    model_type, model_params = _get_model_config(cfg=cfg, kind="item")

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
            X_train = df_train[x_cols].copy()
            y_train = df_train[item].copy()

            # モデルの定義
            model = _build_model(
                model_name=model_type,
                params=model_params,
                random_state=random_state
            )

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
        cfg: Dict[str, Any],
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
        cfg: Dict[str, Any]
            設定辞書
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

    # configがNoneの場合
    if cfg is None:
        raise ValueError("cfg must be provided.")

    # 過去の総合評価と過去のチェック項目列名を取得
    data_cfg = cfg.get("data", {})
    pra_col: str = data_cfg.get("ws2", {}).get("pra_col")
    pca_cols: List[str] = data_cfg.get("ws2", {}).get("pca_cols", [])

    if pra_col is None or not pca_cols:
        raise ValueError("pra_col and pca_cols must be specified in config for WS2.")

    # configからモデル設定を取得
    model_type, model_params = _get_model_config(cfg=cfg, kind="item")

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
            X_train = df_train[x_cols].copy()
            y_train = df_train[item].copy()

            # モデルの定義
            model = _build_model(
                model_name=model_type,
                params=model_params,
                random_state=random_state
            )

            # モデルの学習
            model.fit(X_train, y_train)

            # 特徴量のうちPra, Pca, Caにないものを確認
            missing_features = []
            missing_features += [c for c in pca_cols if c not in Pca]
            missing_features += [c for c in x_cols if c not in Ca and c not in pca_cols and c != pra_col]
            missing_features += [c for c in x_cols if c == pra_col and Pra is None]
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
        cfg: Dict[str, Any],
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
        cfg: Dict[str, Any]
            設定辞書
        random_state: int
            乱数シード
    Returns:
    -------
        pred: int
            予測された総合評価結果
        confidence: float
            予測の信頼度
    """
    # configがNoneの場合
    if cfg is None:
        raise ValueError("cfg must be provided.")

    # 列名の取得
    data_cfg = cfg.get("data", {})
    ca_cols: List[str] = data_cfg.get("ws1", {}).get("ca_cols", [])
    ra_col: str = data_cfg.get("ws1", {}).get("ra_col")

    if ra_col is None or not ca_cols:
        raise ValueError("ra_col and ca_cols must be specified in config for WS1.")

    # 学習データの作成
    for c in ca_cols + [ra_col]:
        if c not in df_train.columns:
            raise ValueError("Required columns missing in training data.")

    X_train = df_train[ca_cols].copy()
    y_train = df_train[ra_col].copy()

    # configからモデル設定を取得
    model_type, model_params = _get_model_config(cfg=cfg, kind="overall")

    # モデルの定義
    model = _build_model(
        model_name=model_type,
        params=model_params,
        random_state=random_state
    )

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
        cfg: Dict[str, Any],
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
        cfg: Dict[str, Any]
            設定辞書
        random_state: int
            乱数シード
    Returns:
    -------
        pred: int
            予測された総合評価結果
        confidence: float
            予測の信頼度
    """
    # configがNoneの場合
    if cfg is None:
        raise ValueError("cfg must be provided.")

    # 列名の取得
    data_cfg = cfg.get("data", {})
    pra_col: str = data_cfg.get("ws2", {}).get("pra_col")
    pca_cols: List[str] = data_cfg.get("ws2", {}).get("pca_cols", [])
    ca_cols: List[str] = data_cfg.get("ws2", {}).get("ca_cols", [])
    ra_col: str = data_cfg.get("ws2", {}).get("ra_col")

    if pra_col is None or not pca_cols or not ca_cols or ra_col is None:
        raise ValueError("pra_col, pca_cols, ca_cols and ra_col must be specified in config for WS2.")

    # 学習データの作成
    for c in [pra_col] + pca_cols + ca_cols + [ra_col]:
        if c not in df_train.columns:
            raise ValueError("Required columns missing in training data.")
    
    X_cols = [pra_col] + pca_cols + ca_cols
    X_train = df_train[X_cols].copy()
    y_train = df_train[ra_col].copy()

    # configからモデル設定を取得
    model_type, model_params = _get_model_config(cfg=cfg, kind="overall")

    # モデルの定義
    model = _build_model(
        model_name=model_type,
        params=model_params,
        random_state=random_state
    )

    # モデルの学習
    model.fit(X_train, y_train)

    # 特徴量のうちPra, Pca, Caにないものを確認
    missing_features = []
    missing_features += [c for c in pca_cols if c not in Pca]
    missing_features += [c for c in ca_cols if c not in Ca]
    missing_features += [c for c in X_cols if c == pra_col and Pra is None]
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

def predict_overall_ws1_with_model(
        model,
        Ca: Dict[str, int],
        ca_cols: List[str]
) -> Tuple[int, float]:
    """
    1回分の学習者データを使って、総合評価を選択されたモデルで推定
    Parameters:
    ----------
        model: Pipeline
            予測に使うモデル
        Ca: Dict[str, int]
            回答済みor補完済み項目
        ca_cols: List[str]
            使用するチェック項目列名
    Returns:
    -------
        pred: int
            予測された総合評価結果
        confidence: float
            予測の信頼度
    """
    x_pred = pd.DataFrame([[Ca[c] for c in ca_cols]], columns=ca_cols)
    proba = model.predict_proba(x_pred)[0]
    classes = model.named_steps["clf"].classes_
    pred_idx = int(np.argmax(proba))
    return int(classes[pred_idx]), float(proba[pred_idx])