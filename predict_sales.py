
import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("📈 부크크 매출 예측 프로그램 (Prophet 기반)")

uploaded_file = st.file_uploader("📁 엑셀 파일 업로드 (일자별 매출)", type=["xlsx"])

if uploaded_file:
    sheet = pd.ExcelFile(uploaded_file).sheet_names[0]
    df = pd.read_excel(uploaded_file, sheet_name=sheet)

    df.columns = df.iloc[0]
    df = df.drop(index=0)
    df = df.rename(columns={pd.NaT: '일자', '일': '요일'})
    df['일자'] = pd.to_datetime(df['일자'])
    df['합계'] = pd.to_numeric(df['합계'], errors='coerce')

    df_prophet = df[['일자', '합계']].dropna().rename(columns={'일자': 'ds', '합계': 'y'})

    st.success(f"✅ {len(df_prophet)}건의 데이터가 로드되었습니다.")

    start_date = st.date_input("예측 시작일", value=pd.to_datetime("2025-09-09"))
    end_date = st.date_input("예측 종료일", value=pd.to_datetime("2025-12-31"))

    if start_date >= end_date:
        st.warning("⚠️ 예측 종료일은 시작일보다 이후여야 합니다.")
    else:
        model = Prophet()
        model.fit(df_prophet)

        future = model.make_future_dataframe(
    periods=(pd.to_datetime(end_date) - pd.to_datetime(df_prophet['ds'].max())).days + 1,
    freq='D'
        )

        forecast = model.predict(future)
        forecast_range = forecast[(forecast['ds'] >= pd.to_datetime(start_date)) & (forecast['ds'] <= pd.to_datetime(end_date))]

        st.subheader("📊 예측 요약")
        st.dataframe(forecast_range[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].rename(columns={
            'ds': '날짜', 'yhat': '예측 매출', 'yhat_lower': '하한값', 'yhat_upper': '상한값'
        }).round(0), use_container_width=True)

        st.subheader("📈 예측 그래프")
        fig1 = model.plot(forecast)
        st.pyplot(fig1)

        st.subheader("📉 트렌드 구성 요소")
        fig2 = model.plot_components(forecast)
        st.pyplot(fig2)
