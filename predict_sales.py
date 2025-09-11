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
# 🔧 엑셀 업로드
# -----------------------------
st.sidebar.header("1️⃣ 엑셀 업로드")
uploaded_file = st.sidebar.file_uploader("매출 데이터를 포함한 엑셀 파일을 업로드하세요", type=["xlsx"])

# -----------------------------
# 🔧 예측 기간 선택
# -----------------------------
st.sidebar.header("2️⃣ 예측 기간 선택")
start_date = st.sidebar.date_input("예측 시작일", pd.to_datetime("today"))
end_date = st.sidebar.date_input("예측 종료일", pd.to_datetime("2025-12-31"))
start_date = pd.to_datetime(start_date)
end_date = pd.to_datetime(end_date)

# -----------------------------
# 📂 엑셀 전처리
# -----------------------------
def preprocess_excel(file):
    df = pd.read_excel(file, sheet_name=0, header=1)
    df.columns = df.columns.str.strip()
    date_col = next((col for col in df.columns if '일자' in col or '날짜' in col), None)
    df[date_col] = pd.to_datetime(df[date_col])
    clients = df.columns.drop(date_col)
    df_melt = df.melt(id_vars=date_col, value_vars=clients, var_name='거래처', value_name='매출액')
    df_melt.dropna(subset=['매출액'], inplace=True)
    df_melt.rename(columns={date_col: 'ds', '매출액': 'y'}, inplace=True)
    df_melt['y'] = pd.to_numeric(df_melt['y'], errors='coerce').round()
    df_melt.dropna(subset=['y'], inplace=True)
    return df_melt

# -----------------------------
# 🔮 예측 함수
# -----------------------------
def predict_sales(df, start_date, end_date):
    future_df_all = []

    for client in df['거래처'].unique():
        df_client = df[df['거래처'] == client].copy()

        # 4-1: 교보문고는 월 단위로 학습
        if '교보문고' in client:
            df_client['ds_month'] = df_client['ds'].dt.to_period('M').dt.to_timestamp()
            df_monthly = df_client.groupby('ds_month')['y'].sum().reset_index().rename(columns={'ds_month': 'ds'})
            model = Prophet()
            model.fit(df_monthly)

            future_month = pd.date_range(start=start_date, end=end_date, freq='MS')
            future_df = pd.DataFrame({'ds': future_month})
            forecast = model.predict(future_df)[['ds', 'yhat']]
            forecast['yhat'] = forecast['yhat'].clip(lower=0)

            # 일 단위로 분배
            result = []
            for _, row in forecast.iterrows():
                month = row['ds']
                days_in_month = pd.date_range(month, month + pd.offsets.MonthEnd(0), freq='D')
                daily_value = row['yhat'] / len(days_in_month)
                for day in days_in_month:
                    if start_date <= day <= end_date:
                        result.append({'ds': day, '거래처': client, 'yhat_final': round(daily_value)})
            forecast_df = pd.DataFrame(result)

        else:
            model = Prophet()
            model.fit(df_client[['ds', 'y']].sort_values('ds'))

            future = model.make_future_dataframe(periods=(end_date - df_client['ds'].max()).days, freq='D')
            forecast = model.predict(future)[['ds', 'yhat']]
            forecast = forecast[(forecast['ds'] >= start_date) & (forecast['ds'] <= end_date)]
            forecast['yhat_final'] = forecast['yhat'].clip(lower=0).round()
            forecast['거래처'] = client
            forecast_df = forecast[['ds', '거래처', 'yhat_final']]

        future_df_all.append(forecast_df)

    return pd.concat(future_df_all)

# -----------------------------
# 📊 요약 출력
# -----------------------------
def display_summary_table(predicted, original, start_date, end_date):
    st.subheader("📊 예측 요약 (일자별 × 거래처별)")

    # 과거 구간은 원본에서, 미래 구간은 예측에서 병합
    original_range = original[(original['ds'] >= start_date) & (original['ds'] <= end_date)].copy()
    original_range.rename(columns={'y': 'yhat_final'}, inplace=True)
    merged = pd.concat([original_range[['ds', '거래처', 'yhat_final']], predicted])

    pivot = merged.pivot_table(index='ds', columns='거래처', values='yhat_final', aggfunc='sum').fillna(0)

    display = pivot.copy().astype(int).applymap(lambda x: f"{x:,}")
    st.dataframe(display.reset_index().rename(columns={"ds": "날짜"}), use_container_width=True)

    # 거래처별 합계
    st.markdown("### 📌 거래처별 예측 매출 합계")
    total_by_client = pivot.sum()
    for client, amount in total_by_client.items():
        st.markdown(f"- **{client}**: {int(amount):,} 원")
    st.markdown(f"### ✅ 전체 합계: **{int(total_by_client.sum()):,} 원**")

    # 다운로드 버튼
    csv = pivot.reset_index().to_csv(index=False).encode("utf-8-sig")
    st.download_button("📥 예측 결과 다운로드", data=csv, file_name="예측결과.csv", mime="text/csv")

# -----------------------------
# 📉 그래프 출력
# -----------------------------
def display_graphs(df):
    st.subheader("📈 거래처별 매출 예측 그래프")

    pivot = df.pivot_table(index='ds', columns='거래처', values='yhat_final', aggfunc='sum').fillna(0)
    total_sum = pivot.sum().sum()

    cols = st.columns(2)
    for i, client in enumerate(pivot.columns):
        with cols[i % 2]:
            fig = px.bar(pivot[client].reset_index(), x='ds', y=client,
                         title=f"{client} 매출 예측",
                         labels={'ds': '날짜', client: '매출'},
                         text=pivot[client].apply(lambda x: f"{(x/total_sum*100):.1f}%" if total_sum > 0 else ""))
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)

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

    original_data = st.session_state.library_data.copy()
    forecast_data = predict_sales(original_data, start_date, end_date)
    display_summary_table(forecast_data, original_data, start_date, end_date)
    display_graphs(forecast_data)
else:
    st.info("👈 왼쪽 사이드바에서 엑셀 파일을 업로드하세요.")
