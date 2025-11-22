import os
import glob
import numpy as np
import pandas as pd

def correct_typos(data):
    """
    列名に含まれる誤字を修正する

    Parameters:
    ----------
    data : pd.DataFrame
        入力データフレーム
    
    Returns:
    -------
    修正後のデータフレーム
    """
    # 誤字と正しい綴りの対応
    typo_corrections = {
        "assessemnt": "assessment",
    }

    # 列名修正
    new_columns = []
    for col in data.columns:
        for typo, correct in typo_corrections.items():
            col = col.replace(typo, correct)
        new_columns.append(col)
    data.columns = new_columns

    return data


def adjust_values(data):
    """
    2回実施データの値を変換する
    変換ルール：
    ・総合評価: 負の値は0とする。正の値は小数点第一位四捨五入して整数とする。4を超える値は4とする。
    ・振り返り文字数: 負の値は0とする。正の値は小数点第一位四捨五入して整数とする。
    ・チェックリスト評価結果: 負の値は0とする。正の値は小数点第一位四捨五入して整数とする。2を超える値は2とする

    Parameters:
    ----------
    data: pd.DataFrame
        入力データフレーム
    Returns:
    -------
    変換後のデータフレーム
    """
    # 総合評価の調整（w3とw4の両方）
    for col in ['w3-assessment-result', 'w4-assessment-result']:
        if col in data.columns:
            data[col] = data[col].apply(lambda x: 0 if x < 0 else round(min(x, 4)))

    # 振り返り文字数の調整（w3とw4の両方）
    for col in ['w3-reflection-length', 'w4-reflection-length']:
        if col in data.columns:
            data[col] = data[col].apply(lambda x: 0 if x < 0 else round(x))

    # チェックリスト評価結果の調整（w3とw4の両方）
    for col in data.columns:
        if (col.startswith('w3-') or col.startswith('w4-')) and col[3:].isdigit():
            data[col] = data[col].apply(lambda x: 0 if x < 0 else round(min(x, 2)))

    return data

def preprocess_data(input_folder, output_folder):
    """
    指定したフォルダ内のCSVファイルに前処理を適用し、新しいフォルダに保存する
    Parameters:
    ----------
    input_folder : str
        入力フォルダのパス
    output_folder : str
        出力フォルダのパス
    """
    csv_files = glob.glob(os.path.join(input_folder, '*.csv'))

    for file_path in csv_files:
        try:
            # CSVファイルの読み込み
            data = pd.read_csv(file_path)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue

        # 誤字修正
        data = correct_typos(data)

        # Unnamed列を削除
        data = data.loc[:, ~data.columns.str.contains('^Unnamed')]

        # 値の変換処理
        data_adjusted = adjust_values(data)

        # 欠損値の補完
        # 総合評価は最頻値で補完
        for col in ['w3-assessment-result', 'w4-assessment-result']:
            if col in data_adjusted.columns:
                mode_value = data_adjusted[col].mode()[0]
                data_adjusted[col] = data_adjusted[col].fillna(mode_value)

        # 振り返り文字数は0で補完
        for col in ['w3-reflection-length', 'w4-reflection-length']:
            if col in data_adjusted.columns:
                data_adjusted[col] = data_adjusted[col].fillna(0)

        # チェックリスト評価結果は各時点の最頻値で補完
        # w3のチェックリスト項目
        w3_items = [col for col in data_adjusted.columns if col.startswith('w3-') and col[3:].isdigit()]
        if w3_items:
            w3_modes = data_adjusted[w3_items].mode().iloc[0]
            for col in w3_items:
                data_adjusted[col] = data_adjusted[col].fillna(w3_modes[col])

        # w4のチェックリスト項目
        w4_items = [col for col in data_adjusted.columns if col.startswith('w4-') and col[3:].isdigit()]
        if w4_items:
            w4_modes = data_adjusted[w4_items].mode().iloc[0]
            for col in w4_items:
                data_adjusted[col] = data_adjusted[col].fillna(w4_modes[col])

        # 出力フォルダに保存
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        save_path = os.path.join(output_folder, f"{base_name}_processed.csv")
        data_adjusted.to_csv(save_path, index=False)
        print(f"Processed and saved: {save_path}")

if __name__ == '__main__':
    # 入力フォルダと出力フォルダの相対パス
    input_folders = [
        '../../data/synthetic/ws2_synthetic_20250326_130',
        '../../data/synthetic/ws2_synthetic_20250326_1300'
    ]

    output_folders = [
        '../../data/processed/w2-synthetic_20250326_130_processed',
        '../../data/processed/w2-synthetic_20250326_1300_processed'
    ]

    # 出力フォルダの作成
    for output_folder in output_folders:
        os.makedirs(output_folder, exist_ok=True)
        print(f"Created output directory: {output_folder}")

    # 各データセットに対して前処理を実行
    for input_folder, output_folder in zip(input_folders, output_folders):
        print(f"\nProcessing data from {input_folder}")
        print(f"Output will be saved to {output_folder}")
        preprocess_data(input_folder, output_folder)
        print(f"Completed processing for {input_folder}")