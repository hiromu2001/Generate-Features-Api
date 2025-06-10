import pandas as pd
import numpy as np
import jpholiday
import lightgbm as lgb
import joblib
import os
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# === 設定 ===
MODEL_PATH = "models/discount_model.lgb"
DATA_FILE = "uploaded/sales_data.xlsx"  # または .csv に置き換え可
os.makedirs("models", exist_ok=True)

# === データ読み込み（ExcelまたはCSV） ===
def load_and_preprocess(file_path: str) -> pd.DataFrame:
    ext = os.path.splitext(file_path)[-1].lower()
    if ext == ".csv":
        df = pd.read_csv(file_path, encoding="utf-8")
    else:
        df = pd.read_excel(file_path, header=1)

    print("\n[読み込み完了] shape:", df.shape)

    df.drop(columns=df.columns[2:5], inplace=True)
    df = df[df[df.columns[1]].notna()]
    df.rename(columns={df.columns[0]: "商品コード", df.columns[1]: "商品名"}, inplace=True)

    def melt_and_clean(cols, value_name):
        d = pd.melt(df, id_vars=["商品コード", "商品名"], value_vars=cols, var_name="日付_raw", value_name=value_name)
        d["日付"] = d["日付_raw"].str.extract(r'(\d{4}年\d{2}月\d{2}日)')[0]
        d["日付"] = pd.to_datetime(d["日付"], format="%Y年%m月%d日", errors='coerce')
        return d.drop(columns=["日付_raw"])

    df_qty = melt_and_clean([c for c in df.columns if "販売数量" in c], "販売数量")
    df_amt = melt_and_clean([c for c in df.columns if "販売金額" in c], "販売金額")
    df_rate = melt_and_clean([c for c in df.columns if "売変合計率" in c], "売変合計率")

    merged = df_qty.merge(df_amt, on=["商品コード", "商品名", "日付"])\
                   .merge(df_rate, on=["商品コード", "商品名", "日付"])
    merged.dropna(subset=["日付"], inplace=True)

    # 欠損値補完
    merged["販売数量"].fillna(0, inplace=True)
    merged["販売金額"].fillna(0, inplace=True)
    merged["売変合計率"].fillna(merged["売変合計率"].mean(), inplace=True)

    print("[前処理完了] レコード数:", len(merged))
    return merged

# === 特徴量エンジニアリング ===
def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["month"] = df["日付"].dt.month
    df["week"] = df["日付"].dt.isocalendar().week.astype(int)
    df["day_of_week"] = df["日付"].dt.dayofweek
    df["day_of_month"] = df["日付"].dt.day
    df["is_weekend"] = df["day_of_week"].isin([5,6]).astype(int)
    df["zero_day_flag"] = df["day_of_month"].isin([10,20,30]).astype(int)
    df["holiday_flag"] = df["日付"].apply(lambda x: int(jpholiday.is_holiday(x)))

    for i in range(7):
        df[f"weekday_{i}"] = (df["day_of_week"] == i).astype(int)

    # 過去7日間平均特徴量
    df = df.sort_values(by=["商品コード", "日付"])
    df["avg_quantity_7d"] = df.groupby("商品コード")["販売数量"].transform(lambda x: x.shift(1).rolling(7, min_periods=1).mean())
    df["avg_amount_7d"] = df.groupby("商品コード")["販売金額"].transform(lambda x: x.shift(1).rolling(7, min_periods=1).mean())
    df["avg_discount_7d"] = df.groupby("商品コード")["売変合計率"].transform(lambda x: x.shift(1).rolling(7, min_periods=1).mean())

    df.dropna(subset=["avg_quantity_7d", "avg_amount_7d", "avg_discount_7d"], inplace=True)
    return df

# === 学習 & 評価 ===
def train_model(df: pd.DataFrame):
    X = df[[
        "販売数量", "month", "week", "day_of_week", "day_of_month", "is_weekend",
        "zero_day_flag", "holiday_flag",
        "avg_quantity_7d", "avg_amount_7d", "avg_discount_7d"
    ] + [f"weekday_{i}" for i in range(7)]]
    y = df["売変合計率"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = lgb.LGBMRegressor(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"\n[ホールドアウト検証] MAE: {mae:.4f}")

    joblib.dump(model, MODEL_PATH)
    print("[モデル保存] →", MODEL_PATH)
    return mae

if __name__ == "__main__":
    print("\n=== モデル学習開始 ===")
    df = load_and_preprocess(DATA_FILE)
    df_feat = create_features(df)
    mae = train_model(df_feat)
    print("\n=== 学習完了 ===")
