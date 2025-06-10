import pandas as pd
import numpy as np
import jpholiday
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from datetime import datetime, timedelta
import lightgbm as lgb
import joblib
import os
from typing import Union
import uvicorn

app = FastAPI()

MODEL_PATH = "models/discount_model.lgb"
UPLOAD_DIR = "uploaded"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# グローバルにデータ保持
data_df = None
model = None

# ==== データ前処理関数（CSV/Excel） ====
def preprocess_excel_or_csv(file_path: str) -> pd.DataFrame:
    ext = os.path.splitext(file_path)[-1].lower()
    if ext == ".csv":
        df = pd.read_csv(file_path, encoding="utf-8")
    else:
        df = pd.read_excel(file_path, header=1)

    # ✅ 列名をすべて文字列に変換（int型列名エラー防止）
    df.columns = df.columns.map(str)

    # ✅ ログ出力：列名確認（Renderログでデバッグ用）
    print("▶ 読み込んだ列名:", df.columns.tolist())

    # C〜E列削除（index 2, 3, 4）
    df.drop(columns=df.columns[2:5], inplace=True)

    # 商品名が欠損している行は削除
    df = df[df[df.columns[1]].notna()]

    # 列名リネーム（先頭2列）
    df.rename(columns={df.columns[0]: "商品コード", df.columns[1]: "商品名"}, inplace=True)

    # ==== 各項目の縦持ち化 ====
    def melt_and_clean(cols, value_name):
        d = pd.melt(df, id_vars=["商品コード", "商品名"], value_vars=cols, var_name="日付_raw", value_name=value_name)
        d["日付"] = d["日付_raw"].str.extract(r'(\d{4}年\d{2}月\d{2}日)')[0]
        d["日付"] = pd.to_datetime(d["日付"], format="%Y年%m月%d日", errors='coerce')
        return d.drop(columns=["日付_raw"])

    df_qty = melt_and_clean([c for c in df.columns if "販売数量" in c], "販売数量")
    df_amt = melt_and_clean([c for c in df.columns if "販売金額" in c], "販売金額")
    df_rate = melt_and_clean([c for c in df.columns if "売変合計率" in c], "売変合計率")

    # 結合
    merged = df_qty.merge(df_amt, on=["商品コード", "商品名", "日付"])\
                   .merge(df_rate, on=["商品コード", "商品名", "日付"])

    # 日付欠損除外
    merged.dropna(subset=["日付"], inplace=True)

    # 欠損処理
    merged["販売数量"].fillna(0, inplace=True)
    merged["販売金額"].fillna(0, inplace=True)
    merged["売変合計率"].fillna(merged["売変合計率"].mean(), inplace=True)

    return merged


# ==== 特徴量生成 ====
def generate_features(df: pd.DataFrame, item: Union[str, int], date: str, quantity: int) -> pd.DataFrame:
    date = pd.to_datetime(date)
    latest = df[df["商品コード"].astype(str).eq(str(item)) | df["商品名"].eq(str(item))]
    if latest.empty:
        raise ValueError("指定された商品が見つかりません")

    week_ago = date - timedelta(days=7)
    df_recent = latest[(latest["日付"] >= week_ago) & (latest["日付"] < date)]

    feature = {
        "販売数量": quantity,
        "month": date.month,
        "week": date.isocalendar().week,
        "day_of_week": date.dayofweek,
        "day_of_month": date.day,
        "is_weekend": int(date.dayofweek >= 5),
        "zero_day_flag": int(date.day in [10, 20, 30]),
        "holiday_flag": int(jpholiday.is_holiday(date)),
        "avg_quantity_7d": df_recent["販売数量"].mean() if not df_recent.empty else 0,
        "avg_amount_7d": df_recent["販売金額"].mean() if not df_recent.empty else 0,
        "avg_discount_7d": df_recent["売変合計率"].mean() if not df_recent.empty else 0
    }

    for i in range(7):
        feature[f"weekday_{i}"] = 1 if date.dayofweek == i else 0

    print("[特徴量生成]", feature)
    return pd.DataFrame([feature])

# ==== モデル読み込み ====
def load_model():
    global model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("モデルファイルが存在しません")
    model = joblib.load(MODEL_PATH)
    print("[モデル読み込み完了]")

# ==== リクエストモデル ====
class PredictRequest(BaseModel):
    item: Union[str, int]
    date: str  # YYYY-MM-DD
    quantity: int

@app.post("/upload")
def upload_file(file: UploadFile = File(...)):
    global data_df
    print("✅ /upload にリクエストが届いた")
    try:
        print("▶ アップロード開始")

        # ファイル読み込み＆ログ出力
        contents = file.file.read()
        print(f"▶ ファイル名: {file.filename}")
        print(f"▶ ファイルサイズ: {len(contents)} bytes")

        # ファイル保存
        path = os.path.join(UPLOAD_DIR, file.filename)
        with open(path, "wb") as f:
            f.write(contents)
        print(f"▶ 保存完了: {path}")

        # 前処理実行
        data_df = preprocess_excel_or_csv(path)
        print("▶ 前処理完了")
        print("▶ データ件数:", len(data_df))
        print("▶ カラム:", data_df.columns.tolist())
        print(data_df.head())

        return {"message": "アップロードと前処理が完了しました。"}
    except Exception as e:
        print("❌ アップロード処理でエラー:", e)
        return {"error": f"アップロード処理中にエラーが発生しました: {str(e)}"}


@app.post("/predict")
def predict(req: PredictRequest):
    global data_df, model
    if data_df is None:
        return {"error": "先に /upload で販売実績ファイルをアップロードしてください。"}
    if model is None:
        load_model()

    try:
        X = generate_features(data_df, req.item, req.date, req.quantity)
        y_pred = model.predict(X)[0]
        response = {
            "result": f"『{req.item}』について分析した結果、{req.date} に {req.quantity}個販売すると、売変合計率はおおよそ {round(y_pred * 100, 2)}% と予測されます。\nモデルはホールドアウト検証で平均誤差が約±◯% でした。",
            "predicted_discount_rate": round(float(y_pred), 4)
        }
        print("[予測結果]", response)
        return response
    except Exception as e:
        print("[エラー]", e)
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
