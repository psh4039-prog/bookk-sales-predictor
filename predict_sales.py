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
def predict_sales(df, start_date, end_date):
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    all_forecasts = []
    client_list = df['거래처'].unique()

    for client in client_list:
        df_group = df[df['거래처'] == client][['ds', 'y']].copy().sort_values('ds')

        if client == "교보문고":
            df_group['월'] = df_group['ds'].dt.to_period("M")
            monthly_sum = df_group.groupby('월')['y'].sum().reset_index()
            monthly_sum['ds'] = monthly_sum['월'].dt.to_timestamp()
            prophet_df = monthly_sum[['ds', 'y']]
            model = Prophet()
            model.fit(prophet_df)
            future_months = pd.date_range(start=prophet_df['ds'].max(), end=end_date, freq='MS')
            future_df = pd.DataFrame({'ds': future_months})
            forecast_monthly = model.predict(future_df)
            forecast_monthly = forecast_monthly[['ds', 'yhat']].copy()
            forecast_monthly['yhat'] = forecast_monthly['yhat'].round()

            # 월 예측값 → 평일 일수로 분배
            all_days = pd.date_range(start=start_date, end=end_date, freq='D')
            for _, row in forecast_monthly.iterrows():
                month = row['ds'].strftime('%Y-%m')
                days_in_month = [d for d in all_days if d.strftime('%Y-%m') == month and d.weekday() < 5]
                if not days_in_month:
                    continue
                daily_yhat = int(row['yhat']) // len(days_in_month)
                for d in days_in_month:
                    all_forecasts.append({'ds': d, 'yhat': daily_yhat, '거래처': client})
        else:
            model = Prophet()
            model.fit(df_group)
            last_date = df_group['ds'].max()
            period_days = max((end_date - last_date).days, 1)
            future = model.make_future_dataframe(periods=period_days, freq='D')
            forecast = model.predict(future)
            result = forecast[['ds', 'yhat']]
            result = result[result['ds'].between(start_date, end_date)]
            result['yhat'] = result['yhat'].round()
            result['거래처'] = client
            all_forecasts.extend(result.to_dict(orient='records'))

    df_result = pd.DataFrame(all_forecasts)

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
