# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
from prophet import Prophet
from io import BytesIO
import plotly.express as px
from datetime import datetime

st.set_page_config(page_title="부크크 매출 예측기", layout="wide")
st.title("📈 부크크 매출 예측기")

# --- 초기 세션 상태 ---
if "library_data" not in st.session_state:
    st.session_state.library_data = None

# --- 엑셀 업로드 ---
st.sidebar.header("1️⃣ 엑셀 업로드")
uploaded_file = st.sidebar.file_uploader("매출 데이터를 포함한 엑셀 파일 업로드", type=["xlsx"])

# --- 예측 기간 ---
st.sidebar.header("2️⃣ 예측 기간 선택")
start_date = st.sidebar.date_input("예측 시작일", datetime(2025, 9, 9))
end_date = st.sidebar.date_input("예측 종료일", datetime(2025, 12, 31))

# --- 엑셀 전처리 ---
def preprocess_excel(uploaded_file):
    df = pd.read_excel(uploaded_file, sheet_name=0, header=1)
    df.columns = df.columns.str.strip()
    date_col = [col for col in df.columns if '일자' in col or '날짜' in col or 'date' in col.lower()][0]
    df[date_col] = pd.to_datetime(df[date_col])
    clients = df.columns.drop(date_col)
    df_melted = df.melt(id_vars=date_col, value_vars=clients, var_name='거래처', value_name='매출액')
    df_melted.dropna(subset=['매출액'], inplace=True)
    df_melted.rename(columns={date_col: 'ds', '매출액': 'y'}, inplace=True)
    df_melted['y'] = pd.to_numeric(df_melted['y'], errors='coerce').round()
    df_melted.dropna(subset=['y'], inplace=True)
    return df_melted

# --- 예측 함수 ---
def predict_sales(df: pd.DataFrame, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    all_forecasts = []

    if '거래처' in df.columns:
        clients = df['거래처'].unique()
    else:
        clients = [None]

    for client in clients:
        df_client = df[df['거래처'] == client] if client else df.copy()
        if df_client.empty:
            continue

        if client == "교보문고":
            df_client['ds'] = pd.to_datetime(df_client['ds'])
            df_client['ds'] = df_client['ds'].dt.to_period("M").dt.to_timestamp()
            df_monthly = df_client.groupby('ds').agg({'y': 'sum'}).reset_index()

            last_train_date = df_monthly['ds'].max()
            period_months = (end_date.to_period("M") - last_train_date.to_period("M")).n + 1
            if period_months <= 0:
                continue

            model = Prophet()
            model.fit(df_monthly)
            future_df = model.make_future_dataframe(periods=period_months, freq='MS')
            forecast = model.predict(future_df)
        else:
            df_client = df_client.sort_values('ds')
            last_date = df_client['ds'].max()
            period_days = max((end_date - last_date).days, 1)

            model = Prophet()
            model.fit(df_client)
            future_df = model.make_future_dataframe(periods=period_days, freq='D')
            forecast = model.predict(future_df)

        forecast_filtered = forecast[forecast['ds'].between(start_date, end_date)][['ds', 'yhat']].copy()
        forecast_filtered['yhat'] = forecast_filtered['yhat'].round().astype(int)
        if client:
            forecast_filtered['거래처'] = client
        all_forecasts.append(forecast_filtered)

    if not all_forecasts:
        return pd.DataFrame(columns=['ds', 'yhat', '거래처'])

    return pd.concat(all_forecasts).reset_index(drop=True)

    # 과거 매출은 엑셀 데이터로 덮어쓰기
    df_merged = df_result.merge(df, on=["ds", "거래처"], how="left", suffixes=("_pred", ""))
    df_merged["yhat_final"] = df_merged["y"].combine_first(df_merged["yhat_pred"])
    df_merged.drop(columns=["y", "yhat_pred"], inplace=True)

    return df_merged

# --- 일자별 요약 표시 ---
def display_summary_table(forecast_df):
    st.subheader("📊 예측 요약")
    pivot_df = forecast_df.pivot_table(index='ds', columns='거래처', values='yhat_final', aggfunc='sum').fillna(0)
    display_df = pivot_df.copy().astype(int).applymap(lambda x: f"{x:,}")
    st.dataframe(display_df.reset_index().rename(columns={'ds': '날짜'}), use_container_width=True)

    st.markdown("---")
    st.markdown("### ✅ 거래처별 및 전체 합계")
    totals = pivot_df.sum()
    for client, val in totals.items():
        st.markdown(f"- **{client}**: {int(val):,} 원")
    st.markdown(f"### 📌 전체 합계: **{int(totals.sum()):,} 원**")

# --- 실행 ---
if uploaded_file:
    df_new = preprocess_excel(uploaded_file)
    if st.session_state.library_data is None:
        st.session_state.library_data = df_new
    else:
        st.session_state.library_data = pd.concat([st.session_state.library_data, df_new], ignore_index=True)
        st.session_state.library_data.drop_duplicates(subset=['ds', '거래처'], keep='last', inplace=True)

    df_all = st.session_state.library_data
    forecast_df = predict_sales(df_all, start_date, end_date)
    display_summary_table(forecast_df)

    # 다운로드 기능
    csv = forecast_df.to_csv(index=False).encode('utf-8-sig')
    st.download_button(label="📥 예측 결과 다운로드", data=csv, file_name="sales_forecast.csv", mime="text/csv")

else:
    st.info("👈 왼쪽 사이드바에서 엑셀 파일을 업로드하세요.")
