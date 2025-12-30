import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional, Hashable

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from adaptive_self_assessment.model_store import ModelStore

FrozenParams = Tuple[Tuple[str, Hashable], ...]

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

def _predict_with_model(
        model: Pipeline,
        x_cols: List[str],
        x_values: List[Any]
) -> Tuple[int, float]:
    """
    与えられたモデルで予測を行う内部関数
    Parameters:
    ----------
        model: Pipeline
            予測に使うモデル
        x_cols: List[str]
            使用する特徴量列名
        x_values: List[Any]
            予測用の特徴量データ
    Returns:
    -------
        pred: int
            予測されたクラスラベル
        confidence: float
            予測の信頼度
    """
    x_pred = pd.DataFrame([x_values], columns=x_cols)
    proba = model.predict_proba(x_pred)[0]
    classes = model.named_steps["clf"].classes_
    pred_idx = int(np.argmax(proba))
    return int(classes[pred_idx]), float(proba[pred_idx])

def predict_overall_ws1(
        Ca: Dict[str, int],
        df_train: pd.DataFrame,
        cfg: Dict[str, Any],
        fold: int,
        store: Optional[ModelStore] = None,
        random_state: int = 42,
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
        fold: int
            交差検証のfold番号
        store: Optional[ModelStore]
            モデルストア（キャッシュ用）
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
    
    for c in ca_cols + [ra_col]:
        if c not in df_train.columns:
            raise ValueError("Required columns missing in training data.")

    missing_ca = [c for c in ca_cols if c not in Ca]
    if missing_ca:
        raise ValueError(f"Missing features in Ca: {missing_ca}")

    # configからモデル設定を取得
    model_type, model_params = _get_model_config(cfg=cfg, kind="overall")

    # モデルストアからモデルを取得
    key = (
        "ws1",
        "overall",
        int(fold),
        tuple(ca_cols),
        str(model_type),
    )
    model = store.get(key) if store is not None else None

    # モデルがキャッシュにない場合は新規作成・学習
    if model is None:
        # 学習データの作成
        X_train = df_train[ca_cols]
        y_train = df_train[ra_col]
        model = _build_model(model_name=model_type, params=model_params, random_state=random_state)
        model.fit(X_train, y_train)
        if store is not None:
            store.set(key, model)

    # 総合評価を推定
    x_values = [Ca[c] for c in ca_cols]
    return _predict_with_model(model, ca_cols, x_values)

def predict_overall_ws2(
        Pra: int,
        Pca: Dict[str, int],
        Ca: Dict[str, int],
        df_train: pd.DataFrame,
        cfg: Dict[str, Any],
        fold: int,
        store: Optional[ModelStore] = None,
        random_state: int = 42,
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
        fold: int
            交差検証のfold番号
        store: Optional[ModelStore]
            モデルストア（キャッシュ用）
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
    ws2_cfg = data_cfg.get("ws2", {})
    pra_col: str = ws2_cfg.get("pra_col")
    pca_cols: List[str] = ws2_cfg.get("pca_cols", [])
    ca_cols: List[str] = ws2_cfg.get("ca_cols", [])
    ra_col: str = ws2_cfg.get("ra_col")

    if pra_col is None or not pca_cols or not ca_cols or ra_col is None:
        raise ValueError("pra_col, pca_cols, ca_cols and ra_col must be specified in config for WS2.")
    
    required = [pra_col] + pca_cols + ca_cols + [ra_col]
    for c in required:
        if c not in df_train.columns:
            raise ValueError(f"Required columns missing in training data: {c}")
    
    if Pra is None:
        raise ValueError("Pra must be provided.")
    missing_pca = [c for c in pca_cols if c not in Pca]
    missing_ca = [c for c in ca_cols if c not in Ca]
    if missing_pca or missing_ca:
        raise ValueError(f"Missing features in Pca/Ca: {missing_pca + missing_ca}")

    # configからモデル設定を取得
    X_cols = [pra_col] + pca_cols + ca_cols
    model_type, model_params = _get_model_config(cfg=cfg, kind="overall")
    key  = (
        "ws2",
        "overall",
        int(fold),
        tuple(X_cols),
        str(model_type),
    )            

    model = store.get(key) if store is not None else None

    # モデルがキャッシュにない場合は新規作成・学習
    if model is None:
        # 学習データの作成
        X_train = df_train[X_cols]
        y_train = df_train[ra_col]
        model = _build_model(model_name=model_type, params=model_params, random_state=random_state)
        model.fit(X_train, y_train)
        if store is not None:
            store.set(key, model)
    
    # 総合評価を推定
    x_values: List[int] = []
    for c in X_cols:
        if c == pra_col:
            x_values.append(Pra)
        elif c in pca_cols:
            x_values.append(Pca[c])
        else:
            x_values.append(Ca[c])

    return _predict_with_model(model, X_cols, x_values)

def predict_item_ws1(
        Ca: Dict[str, int], 
        C: List[str], 
        df_train: pd.DataFrame,
        cfg: Dict[str, Any],
        fold: int,
        store: Optional[ModelStore] = None,
        random_state: int = 42
    ) -> Tuple[Dict[str, int], Dict[str, float]]:
    """
    1回分の学習者データを使って、未回答項目Cを項目ごとに推定
    Parameters:
    ----------
        Ca: Dict[str, int]
            回答済みor補完済み項目
        C: List
            残りの質問項目
        df_train: pd.DataFrame 
            予測に使う訓練データ
        cfg: Dict[str, Any]
            設定辞書
        fold: int
            交差検証のfold番号
        store: Optional[ModelStore]
            モデルストア（キャッシュ用）
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
    ca_cols_exist = [c for c in Ca.keys() if c in df_train.columns]

    # 各項目の予測
    for item in C:
        try:
            # 学習に使う特徴量の列
            x_cols: List[str] = [c for c in ca_cols_exist if c != item]
            if not x_cols or item not in df_train.columns:
                raise ValueError("No usable features or target column missing.")
            
            missing = [c for c in x_cols if c not in Ca]
            if missing:
                raise ValueError(f"Missing features in Ca: {missing}")
            
            # モデルストアからモデルを取得
            key = (
                "ws1",
                "item",
                int(fold),
                str(item),
                tuple(x_cols),
                str(model_type),
            )

            model = store.get(key) if store is not None else None
            # モデルがキャッシュにない場合は新規作成・学習
            if model is None:
                # 学習データの作成
                X_train = df_train[x_cols]
                y_train = df_train[item]
                model = _build_model(
                    model_name=model_type,
                    params=model_params,
                    random_state=random_state
                )
                model.fit(X_train, y_train)
                if store is not None:
                    store.set(key, model)
                
            # チェック項目を推定
            x_values = [Ca[c] for c in x_cols]
            pred, confidence = _predict_with_model(model, x_cols, x_values)
            preds[item] = pred
            confidences[item] = confidence

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
        fold: int,
        store: Optional[ModelStore] = None,
        random_state: int = 42
    ) -> Tuple[Dict[str, int], Dict[str, float]]:
    """
    2回分の学習者データを使って、未回答項目Cを項目ごとに推定
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
        fold: int
            交差検証のfold番号
        store: Optional[ModelStore]
            モデルストア（キャッシュ用）
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
    if Pra is None:
        raise ValueError("Pra must be provided.")
    missing_pca = [c for c in pca_cols if c not in Pca]
    if missing_pca:
        raise ValueError(f"Missing features in Pca: {missing_pca}")

    # configからモデル設定を取得
    model_type, model_params = _get_model_config(cfg=cfg, kind="item")

    # Caにあるキーのうち、df_trainに存在する列のみを使用
    ca_cols_exist = [c for c in Ca.keys() if c in df_train.columns]

    # 各項目の予測
    for item in C:
        try:
            # 学習に使う特徴量の列
            x_cols: List[str] = [pra_col] + pca_cols + [c for c in ca_cols_exist if c != item]
            if not x_cols or item not in df_train.columns:
                raise ValueError("No usable features or target column missing.")
            
            missing = []
            missing = [c for c in x_cols if (c != pra_col and c not in pca_cols and c not in Ca)]
            missing += [c for c in x_cols if (c in pca_cols and c not in Pca)]
            if missing:
                raise ValueError(f"Missing features in Pra/Pca/Ca: {missing}")
            
            # モデルストアからモデルを取得
            key = (
                "ws2",
                "item",
                int(fold),
                str(item),
                tuple(x_cols),
                str(model_type),
            )

            model = store.get(key) if store is not None else None
            # モデルがキャッシュにない場合は新規作成・学習
            if model is None:
                # 学習データの作成
                X_train = df_train[x_cols]
                y_train = df_train[item]
                model = _build_model(
                    model_name=model_type,
                    params=model_params,
                    random_state=random_state
                )
                model.fit(X_train, y_train)
                if store is not None:
                    store.set(key, model)
                
            # チェック項目を推定
            x_values: List[int] = []
            for c in x_cols:
                if c == pra_col:
                    x_values.append(Pra)
                elif c in pca_cols:
                    x_values.append(Pca[c])
                else:
                    x_values.append(Ca[c])

            pred, confidence = _predict_with_model(model, x_cols, x_values)
            preds[item] = pred
            confidences[item] = confidence

        except Exception as e:
            print(f"[predict_item_ws2 error] target={item}: {e}")
            preds[item] = 1
            confidences[item] = 0.5
        
    return preds, confidences