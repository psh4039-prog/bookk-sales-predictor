
# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.express as px
from io import BytesIO

st.set_page_config(page_title="부크크 매출 예측기", layout="wide")
st.title("📉 부크크 매출 예측기")

if "library_data" not in st.session_state:
    st.session_state.library_data = None

# -----------------------------
# 🔧 엑셀 파일 업로드
# -----------------------------
st.sidebar.header("1️⃣ 엑셀 업로드")
uploaded_file = st.sidebar.file_uploader("매출 데이터를 포함한 엑셀 파일을 업로드하세요", type=["xlsx"])

# -----------------------------
# 🔧 예측 기간 선택
# -----------------------------
st.sidebar.header("2️⃣ 예측 기간 선택")
start_date = st.sidebar.date_input("예측 시작일", pd.to_datetime("today"))
end_date = st.sidebar.date_input("예측 종료일", pd.to_datetime("2025-12-31"))

# -----------------------------
# 🔧 엑셀 전처리 함수
# -----------------------------
def preprocess_excel(uploaded_file):
    df = pd.read_excel(uploaded_file, sheet_name=0, header=1)
    df.columns = df.columns.str.strip()

    date_col = None
    for col in df.columns:
        if '일자' in col or '날짜' in col or 'date' in col.lower():
            date_col = col
            break
    if date_col is None:
        raise ValueError("엑셀에 '일자' 또는 '날짜' 열이 없습니다.")

    df[date_col] = pd.to_datetime(df[date_col])
    clients = df.columns.drop(date_col)
    df_melted = df.melt(id_vars=date_col, value_vars=clients, var_name='거래처', value_name='매출액')
    df_melted.dropna(subset=['매출액'], inplace=True)
    df_melted.rename(columns={date_col: 'ds', '매출액': 'y'}, inplace=True)
    df_melted['y'] = pd.to_numeric(df_melted['y'], errors='coerce')
    df_melted.dropna(subset=['y'], inplace=True)
    df_melted['y'] = df_melted['y'].round()
    return df_melted

# -----------------------------
# 🔮 예측 함수
# -----------------------------
def predict_sales(df, start_date, end_date):
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    all_forecasts = []

    if '거래처' in df.columns:
        group_cols = df['거래처'].unique()
    else:
        group_cols = [None]

    for client in group_cols:
        if client is not None:
            df_client = df[df['거래처'] == client][['ds', 'y']].copy()
        else:
            df_client = df[['ds', 'y']].copy()

        df_client = df_client.sort_values('ds')
        if len(df_client) < 2:
            continue

        try:
            model = Prophet()
            model.fit(df_client)

            last_date = pd.to_datetime(df_client['ds'].max())
            period_days = max((end_date - last_date).days, 1)
            future_df = model.make_future_dataframe(periods=period_days, freq='D')
            forecast_monthly = model.predict(future_df)
            result = forecast_monthly[['ds', 'yhat']].copy()
            result = result[result['ds'].between(start_date, end_date)]
            result['yhat'] = result['yhat'].round().astype(int)
            if client is not None:
                result['거래처'] = client
            all_forecasts.append(result)
        except Exception as e:
            continue

    if not all_forecasts:
        return pd.DataFrame()

    df_result = pd.concat(all_forecasts).reset_index(drop=True)
    df_result.rename(columns={"yhat": "yhat_final"}, inplace=True)
    return df_result

# -----------------------------
# 📊 요약 테이블 출력
# -----------------------------
def display_summary_table(forecast_df):
    if forecast_df.empty:
        st.warning("예측 결과가 없습니다. 데이터 확인이 필요합니다.")
        return

    st.subheader("📊 예측 요약 (일자별 × 거래처별)")

    pivot_df = forecast_df.pivot_table(index='ds', columns='거래처', values='yhat_final', aggfunc='sum').fillna(0)
    pivot_df['총합'] = pivot_df.sum(axis=1)
    display_df = pivot_df.copy()
    for col in display_df.columns:
        display_df[col] = display_df[col].apply(lambda x: f"{int(x):,}")

    st.dataframe(display_df.reset_index().rename(columns={"ds": "날짜"}), use_container_width=True)

    st.markdown("### 📌 거래처별 예측 매출 합계")
    client_sums = pivot_df.sum().drop("총합")
    total_sum = pivot_df["총합"].sum()
    for client, amount in client_sums.items():
        st.markdown(f"- **{client}**: {int(amount):,} 원")
    st.markdown(f"### ✅ 전체 합계: **{int(total_sum):,} 원**")

    # 다운로드 기능
    csv = pivot_df.reset_index().to_csv(index=False).encode("utf-8-sig")
    st.download_button(label="📥 예측 결과 다운로드", data=csv, file_name="예측결과_거래처별.csv", mime="text/csv")

# -----------------------------
# 🚀 실행
# -----------------------------
if uploaded_file:
    df_new = preprocess_excel(uploaded_file)

    if st.session_state.library_data is None:
        st.session_state.library_data = df_new
    else:
        st.session_state.library_data = pd.concat([st.session_state.library_data, df_new], ignore_index=True)
        st.session_state.library_data.drop_duplicates(subset=['ds', '거래처'], keep='last', inplace=True)

    st.success("✅ 데이터 업로드 및 누적 학습 완료")
    df_all = st.session_state.library_data.copy()
    forecast_df = predict_sales(df_all, start_date, end_date)
    display_summary_table(forecast_df)
else:
    st.info("👈 왼쪽 사이드바에서 엑셀 파일을 업로드하세요.")
