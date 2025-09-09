
import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("📈 부크크 매출 예측 프로그램 (거래처별 상세 리포트 포함)")

uploaded_file = st.file_uploader("📁 엑셀 파일 업로드 (일자별 매출)", type=["xlsx"])

if uploaded_file:
    sheet = pd.ExcelFile(uploaded_file).sheet_names[0]
    df = pd.read_excel(uploaded_file, sheet_name=sheet)

    df.columns = df.iloc[0]
    df = df.drop(index=0)
    df = df.rename(columns={pd.NaT: '일자', '일': '요일'})
    df['일자'] = pd.to_datetime(df['일자'])

    거래처컬럼 = ['PG사', '예스24', '교보문고', '알라딘', '영풍']
    for col in 거래처컬럼:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df[['일자'] + 거래처컬럼]

    st.success(f"✅ {len(df)}건의 데이터가 로드되었습니다.")

    start_date = st.date_input("예측 시작일", value=pd.to_datetime("2025-09-09"))
    end_date = st.date_input("예측 종료일", value=pd.to_datetime("2025-12-31"))

    if start_date >= end_date:
        st.warning("⚠️ 예측 종료일은 시작일보다 이후여야 합니다.")
    else:
        forecasts = []
        last_date = df['일자'].max()
        total_periods = (pd.to_datetime(end_date) - pd.to_datetime(last_date)).days + 1

        for col in 거래처컬럼:
            df_prophet = df[['일자', col]].dropna().rename(columns={'일자': 'ds', col: 'y'})
            model = Prophet()
            model.fit(df_prophet)
            future = model.make_future_dataframe(periods=total_periods)
            forecast = model.predict(future)
            forecast = forecast[['ds', 'yhat']].rename(columns={'yhat': col})
            forecasts.append(forecast)

        result = forecasts[0]
        for f in forecasts[1:]:
            result = pd.merge(result, f, on='ds', how='outer')
        result = result[result['ds'].between(pd.to_datetime(start_date), pd.to_datetime(end_date))]
        result['합계'] = result[거래처컬럼].sum(axis=1)
        result_display = result.copy()
        for col in 거래처컬럼 + ['합계']:
            result_display[col] = result_display[col].fillna(0).apply(lambda x: f"{int(x/1000):,}")

        st.subheader("📊 예측 요약 (일자별)")
        st.dataframe(result_display.rename(columns={'ds': '날짜'}), use_container_width=True)

        # 📅 월별 합산
        result_monthly = result.copy()
        result_monthly['월'] = result_monthly['ds'].dt.to_period('M')
        monthly_summary = result_monthly.groupby('월')[거래처컬럼 + ['합계']].sum().reset_index()
        monthly_display = monthly_summary.copy()
        for col in 거래처컬럼 + ['합계']:
            monthly_display[col] = monthly_display[col].fillna(0).apply(lambda x: f"{int(x/1000):,}")

        st.subheader("📅 월별 예측 요약")
        st.dataframe(monthly_display.rename(columns={'월': '예측 월'}), use_container_width=True)

        # 📈 시각화 - 거래처별 일별/월별 추이
        st.subheader("📈 거래처별 예측 추이 그래프")

        fig, ax = plt.subplots(figsize=(10, 4))
        for col in 거래처컬럼:
            ax.plot(result['ds'], result[col], label=col)
        ax.set_title('일별 거래처별 매출 추이', fontsize=14)
        ax.set_xlabel('날짜')
        ax.set_ylabel('예측 매출 (₩)')
        ax.legend()
        st.pyplot(fig)

        fig2, ax2 = plt.subplots(figsize=(10, 4))
        for col in 거래처컬럼:
            ax2.plot(monthly_summary['월'].astype(str), monthly_summary[col], label=col)
        ax2.set_title('월별 거래처별 매출 추이', fontsize=14)
        ax2.set_xlabel('월')
        ax2.set_ylabel('예측 매출 (₩)')
        ax2.legend()
        st.pyplot(fig2)
