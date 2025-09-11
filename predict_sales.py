# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.express as px

# 📌 PAGE SETUP
st.set_page_config(page_title="Bookk Sales Predictor", layout="wide")
st.title("📈 Bookk Sales Forecasting App")  # 타이틀은 영문 사용

# 📌 Session State 초기화
if "library_data" not in st.session_state:
    st.session_state.library_data = None

# 📌 Sidebar 입력 영역
st.sidebar.header("1️⃣ Upload Excel")
uploaded_file = st.sidebar.file_uploader("Upload Excel with daily sales data", type=["xlsx"])

st.sidebar.header("2️⃣ Select Forecast Period")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("today"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2025-12-31"))

# 📌 Excel 전처리 함수
def preprocess_excel(file):
    df = pd.read_excel(file, sheet_name=0, header=1)
    df.columns = df.columns.str.strip()

    date_col = next((col for col in df.columns if '일자' in col or '날짜' in col or 'date' in col.lower()), None)
    if date_col is None:
        raise ValueError("No '일자' or '날짜' column found in Excel.")

    df[date_col] = pd.to_datetime(df[date_col])
    clients = df.columns.drop(date_col)
    df_melted = df.melt(id_vars=date_col, value_vars=clients, var_name='Client', value_name='Sales')
    df_melted.dropna(subset=['Sales'], inplace=True)
    df_melted.rename(columns={date_col: 'ds', 'Sales': 'y'}, inplace=True)
    df_melted['y'] = pd.to_numeric(df_melted['y'], errors='coerce').round()
    df_melted.dropna(subset=['y'], inplace=True)
    return df_melted

# 📌 예측 함수
def predict_sales(df, start_date, end_date):
    all_forecasts = []
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    clients = df['Client'].unique() if 'Client' in df.columns else [None]

    for client in clients:
        sub_df = df[df['Client'] == client][['ds', 'y']] if client else df[['ds', 'y']]
        model = Prophet()
        model.fit(sub_df)

        future = model.make_future_dataframe(periods=max((end_date - sub_df['ds'].max()).days, 1), freq='D')
        forecast = model.predict(future)
        forecast = forecast[['ds', 'yhat']]
        forecast = forecast[forecast['ds'].between(start_date, end_date)]
        forecast['yhat'] = forecast['yhat'].round().astype(int)
        forecast['Client'] = client
        all_forecasts.append(forecast)

    return pd.concat(all_forecasts).reset_index(drop=True)

# 📌 예측 요약 표시 함수
def display_summary(df_result):
    st.subheader("📊 Forecast Summary (Daily)")

    df_result = df_result.sort_values("ds")
    pivot = df_result.pivot_table(index='ds', columns='Client', values='yhat', aggfunc='sum').fillna(0)
    pivot['Total'] = pivot.sum(axis=1)

    display_df = pivot.applymap(lambda x: f"{int(x):,}")
    st.dataframe(display_df.reset_index().rename(columns={'ds': 'Date'}), use_container_width=True)

    st.markdown("### 📌 Total Forecast by Client")
    totals = pivot.sum().drop("Total")
    total_all = pivot["Total"].sum()

    for client, amount in totals.items():
        st.markdown(f"- **{client}**: {int(amount):,} KRW")
    st.markdown(f"### ✅ Overall Total: **{int(total_all):,} KRW**")

# 📌 그래프 함수
def plot_forecast(df_result):
    df_result['Month'] = df_result['ds'].dt.to_period('M').astype(str)
    monthly = df_result.groupby(['Month', 'Client'])['yhat'].sum().reset_index()

    col1, col2 = st.columns(2)
    with col1:
        fig1 = px.line(df_result, x='ds', y='yhat', color='Client', title='Daily Forecast')
        fig1.update_layout(width=600, height=400)
        st.plotly_chart(fig1)
    with col2:
        fig2 = px.bar(monthly, x='Month', y='yhat', color='Client', title='Monthly Forecast')
        fig2.update_layout(width=600, height=400)
        st.plotly_chart(fig2)

# 📌 실행
if uploaded_file:
    df_new = preprocess_excel(uploaded_file)
    if st.session_state.library_data is None:
        st.session_state.library_data = df_new
    else:
        st.session_state.library_data = pd.concat([
            st.session_state.library_data, df_new
        ], ignore_index=True).drop_duplicates(subset=['ds', 'Client'], keep='last')

    st.success("✅ Data uploaded and appended successfully.")
    df_all = st.session_state.library_data.copy()
    forecast = predict_sales(df_all, start_date, end_date)
    forecast['Formatted'] = forecast['yhat'].map("{:,}".format)

    # 표 + 요약
    display_summary(forecast)

    # 그래프
    st.subheader("📈 Forecast Charts")
    plot_forecast(forecast)
else:
    st.info("👈 Please upload an Excel file to begin.")
