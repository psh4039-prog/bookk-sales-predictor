
# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.express as px
from io import BytesIO

st.set_page_config(page_title="부크크 매출 예측기", layout="wide")
st.title("📈 부크크 매출 예측기")

# 초기 상태 설정
if "library_data" not in st.session_state:
    st.session_state.library_data = None

# 엑셀 업로드
st.sidebar.header("1️⃣ 엑셀 업로드")
uploaded_file = st.sidebar.file_uploader("매출 데이터를 포함한 엑셀 파일을 업로드하세요", type=["xlsx"])

# 예측 기간 선택
st.sidebar.header("2️⃣ 예측 기간 선택")
start_date = st.sidebar.date_input("예측 시작일", pd.to_datetime("today"))
end_date = st.sidebar.date_input("예측 종료일", pd.to_datetime("2025-12-31"))

# 엑셀 전처리
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

# 예측
def predict_sales(df, start_date, end_date):
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    all_forecasts = []
    group_cols = df['거래처'].unique() if '거래처' in df.columns else [None]
    for client in group_cols:
        df_client = df[df['거래처'] == client][['ds', 'y']].copy() if client else df[['ds', 'y']].copy()
        df_client = df_client.sort_values('ds')
        df_client = df_client.dropna(subset=['y'])
        if df_client.shape[0] < 2:
            st.warning(f"❗ 거래처 '{client}'는 유효한 데이터가 2개 미만이라 예측에서 제외됩니다.")
            continue
        model = Prophet()
        model.fit(df_client)
        last_date = pd.to_datetime(df_client['ds'].max())
        period_days = max((end_date - last_date).days, 1)
        future = model.make_future_dataframe(periods=period_days, freq='D')
        forecast = model.predict(future)
        result = forecast[['ds', 'yhat']].copy()
        result = result[result['ds'].between(start_date, end_date)]
        result['yhat'] = result['yhat'].round().astype(int)
        if client:
            result['거래처'] = client
        all_forecasts.append(result)
    return pd.concat(all_forecasts).reset_index(drop=True) if all_forecasts else pd.DataFrame()

# 시각화
def plot_forecast(forecast):
    fig1 = px.line(forecast, x='ds', y='yhat', title='일별 매출 예측', labels={'ds': '날짜', 'yhat': '예측 매출액'})
    fig1.update_layout(width=600, height=400)
    forecast['month'] = forecast['ds'].dt.to_period('M')
    monthly = forecast.groupby('month')['yhat'].sum().reset_index()
    monthly['month'] = monthly['month'].astype(str)
    fig2 = px.bar(monthly, x='month', y='yhat', title='월별 매출 예측', labels={'month': '월', 'yhat': '예측 매출액'})
    fig2.update_layout(width=600, height=400)
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig1)
    with col2:
        st.plotly_chart(fig2)

# 요약 테이블
def display_daily_summary(df_result):
    st.subheader("📊 예측 요약 (일자별 + 거래처별)")
    df_result = df_result.sort_values("ds")
    pivot_df = df_result.pivot_table(index='ds', columns='거래처', values='yhat', aggfunc='sum').fillna(0)
    pivot_df['예측 매출'] = pivot_df.sum(axis=1)
    display_df = pivot_df.copy()
    for col in display_df.columns:
        display_df[col] = display_df[col].apply(lambda x: f"{int(x):,}")
    st.dataframe(display_df.reset_index().rename(columns={'ds': '날짜'}), use_container_width=True)
    st.markdown("### 📌 거래처별 예측 매출 합계")
    total_by_client = pivot_df.sum().drop("예측 매출")
    total_all = pivot_df["예측 매출"].sum()
    for client, total in total_by_client.items():
        st.markdown(f"- **{client}**: {int(total):,} 원")
    st.markdown(f"### ✅ 전체 합계: **{int(total_all):,} 원**")

    # 🔽 다운로드 버튼
    output_excel = BytesIO()
    pivot_df.to_excel(output_excel, engine='openpyxl')
    st.download_button("📥 예측 결과 다운로드 (Excel)", data=output_excel.getvalue(), file_name="forecast_result.xlsx")

# 실행
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
    if not forecast.empty:
        forecast['예측 매출'] = forecast['yhat'].astype(int).map("{:,}".format)
        display_daily_summary(forecast)
        st.subheader("📈 예측 그래프")
        plot_forecast(forecast)
    else:
        st.warning("예측할 수 있는 데이터가 없습니다.")
else:
    st.info("👈 왼쪽 사이드바에서 엑셀 파일을 업로드하세요.")
