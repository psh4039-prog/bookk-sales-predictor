# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.express as px
from io import BytesIO

st.set_page_config(page_title="부크크 매출 예측기", layout="wide")
st.title("📈 부크크 매출 예측기")

# --- 초기 상태 설정 ---
if "library_data" not in st.session_state:
    st.session_state.library_data = None

# --- 엑셀 파일 업로드 ---
st.sidebar.header("1️⃣ 엑셀 업로드")
uploaded_file = st.sidebar.file_uploader("매출 데이터를 포함한 엑셀 파일을 업로드하세요", type=["xlsx"])

# --- 예측 기간 선택 ---
st.sidebar.header("2️⃣ 예측 기간 선택")
start_date = st.sidebar.date_input("예측 시작일", pd.to_datetime("today"))
end_date = st.sidebar.date_input("예측 종료일", pd.to_datetime("2025-12-31"))

# --- 데이터 전처리 함수 ---
def preprocess_excel(uploaded_file):
    df = pd.read_excel(uploaded_file, sheet_name=0, header=1)
    df.columns = df.columns.str.strip()
    date_col = None
    for col in df.columns:
        if '일자' in col or '날짜' in col or 'date' in col.lower():
            date_col = col
            break
    if date_col is None:
        raise ValueError("엑셀 파일에 '일자' 또는 '날짜'라는 이름의 열이 존재하지 않습니다.")
    df[date_col] = pd.to_datetime(df[date_col])
    clients = df.columns.drop(date_col)
    df_melted = df.melt(id_vars=date_col, value_vars=clients, var_name='거래처', value_name='매출액')
    df_melted.dropna(subset=['매출액'], inplace=True)
    df_melted.rename(columns={date_col: 'ds', '매출액': 'y'}, inplace=True)
    df_melted['y'] = pd.to_numeric(df_melted['y'], errors='coerce')
    df_melted.dropna(subset=['y'], inplace=True)
    df_melted['y'] = df_melted['y'].round()
    return df_melted

# --- 예측 함수 ---
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
            df_group = df[df['거래처'] == client][['ds', 'y']].copy()
        else:
            df_group = df[['ds', 'y']].copy()
        df_group = df_group.sort_values('ds')
        model = Prophet()
        model.fit(df_group)
        last_date = pd.to_datetime(df_group['ds'].max())
        period_days = max((end_date - last_date).days, 1)
        future = model.make_future_dataframe(periods=period_days, freq='D')
        forecast = model.predict(future)
        result = forecast[['ds', 'yhat']].copy()
        result = result[result['ds'].between(start_date, end_date)]
        result.dropna(subset=['yhat'], inplace=True)
        result['yhat'] = result['yhat'].round().astype(int)
        if client is not None:
            result['거래처'] = client
        all_forecasts.append(result)
    df_result = pd.concat(all_forecasts).reset_index(drop=True)
    return df_result

# --- 요약 출력 함수 ---
def display_daily_summary(df_result):
    st.subheader("📊 예측 요약 (일자별)")
    df_result = df_result.dropna(subset=['yhat']).sort_values("ds")
    pivot_df = df_result.pivot_table(index='ds', columns='거래처', values='yhat', aggfunc='sum').fillna(0)
    pivot_df['합계'] = pivot_df.loc[:, pivot_df.columns != '합계'].sum(axis=1)
    display_df = pivot_df.copy()
    for col in display_df.columns:
        display_df[col] = display_df[col].apply(lambda x: f"{int(x):,}")
    st.dataframe(display_df.reset_index().rename(columns={'ds': '날짜'}), use_container_width=True)

    # 하단 합계
    st.markdown("### 📌 거래처별 예측 매출 합계")
    total_by_client = pivot_df.loc[:, pivot_df.columns != '합계'].sum()
    total_all = total_by_client.sum()
    for client, total in total_by_client.items():
        st.markdown(f"- **{client}**: {int(total):,} 원")
    st.markdown(f"### ✅ 전체 합계: **{int(total_all):,} 원**")

# --- 시각화 함수 ---
def plot_forecast(forecast):
    forecast = forecast.dropna(subset=['yhat'])
    forecast['month'] = forecast['ds'].dt.to_period('M')
    monthly = forecast.groupby(['month', '거래처'])['yhat'].sum().reset_index()
    monthly['month'] = monthly['month'].astype(str)
    fig = px.bar(monthly, x='month', y='yhat', color='거래처', barmode='group',
                 labels={'month': '월', 'yhat': '예측 매출액', '거래처': '거래처'}, title="월별 거래처별 예측 추이")
    fig.update_layout(width=1000, height=400)
    st.plotly_chart(fig)

# --- 실행 영역 ---
if uploaded_file:
    df_new = preprocess_excel(uploaded_file)
    if st.session_state.library_data is None:
        st.session_state.library_data = df_new
    else:
        st.session_state.library_data = pd.concat([st.session_state.library_data, df_new], ignore_index=True)
        st.session_state.library_data.drop_duplicates(subset=['ds', '거래처'], keep='last', inplace=True)

    st.success("✅ 데이터 업로드 및 누적 학습 완료")
    df_library = st.session_state.library_data.copy()
    forecast = predict_sales(df_library, start_date, end_date)

    # 예측 이전 기간이면 안내
    if forecast.empty:
        st.warning("⚠️ 선택한 기간에 예측 데이터가 없습니다. 예측 가능한 날짜 이후를 선택해 주세요.")
    else:
        display_daily_summary(forecast)
        st.subheader("📈 예측 그래프")
        plot_forecast(forecast)
else:
    st.info("👈 왼쪽 사이드바에서 엑셀 파일을 업로드하세요.")
