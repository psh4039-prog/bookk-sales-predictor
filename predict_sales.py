# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.express as px
from io import BytesIO

st.set_page_config(page_title="📊 부크크 매출 예측기", layout="wide")
st.title("📈 부크크 매출 예측기")

# 세션 초기화
if "library_data" not in st.session_state:
    st.session_state.library_data = None

# --- 사이드바 설정 ---
st.sidebar.header("1️⃣ 엑셀 업로드")
uploaded_file = st.sidebar.file_uploader("매출 데이터를 포함한 엑셀 파일을 업로드하세요", type=["xlsx"])

st.sidebar.header("2️⃣ 예측 기간 설정")
start_date = st.sidebar.date_input("예측 시작일", pd.to_datetime("today"))
end_date = st.sidebar.date_input("예측 종료일", pd.to_datetime("2025-12-31"))

# --- 엑셀 전처리 ---
def preprocess_excel(uploaded_file):
    df = pd.read_excel(uploaded_file, sheet_name=0, header=1)
    df.columns = df.columns.str.strip()
    date_col = next((col for col in df.columns if "일자" in col or "날짜" in col or "date" in col.lower()), None)
    if date_col is None:
        raise ValueError("날짜 또는 일자 컬럼을 찾을 수 없습니다.")

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.melt(id_vars=[date_col], var_name="거래처", value_name="매출액")
    df.dropna(subset=["매출액"], inplace=True)
    df.rename(columns={date_col: "ds", "매출액": "y"}, inplace=True)
    df["y"] = pd.to_numeric(df["y"], errors="coerce").round()
    df.dropna(subset=["y"], inplace=True)
    return df

# --- 예측 함수 ---
def predict_sales(df, start_date, end_date):
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    if "거래처" in df.columns:
        groups = df["거래처"].unique()
    else:
        groups = [None]

    all_forecasts = []
    for client in groups:
        df_client = df[df["거래처"] == client][["ds", "y"]] if client else df[["ds", "y"]]
        df_client = df_client.sort_values("ds")
        if len(df_client) < 2:
            continue

        model = Prophet()
        model.fit(df_client)

        last_date = pd.to_datetime(df_client["ds"].max())
        period_days = max((end_date - last_date).days, 1)
        future = model.make_future_dataframe(periods=period_days, freq="D")

        forecast = model.predict(future)
        forecast = forecast[forecast["ds"].between(start_date, end_date)].copy()
        forecast["yhat"] = forecast["yhat"].round().astype(int)
        forecast = forecast[["ds", "yhat"]]
        forecast["거래처"] = client
        all_forecasts.append(forecast)

    if not all_forecasts:
        return pd.DataFrame()

    result = pd.concat(all_forecasts, ignore_index=True)
    return result

# --- 시각화 ---
def plot_forecast(forecast_df):
    fig1 = px.line(forecast_df, x="ds", y="yhat", color="거래처", title="📅 일별 매출 예측")
    fig2 = px.bar(forecast_df.groupby(forecast_df["ds"].dt.to_period("M")).sum(numeric_only=True).reset_index().rename(columns={"ds": "월"}), 
                  x="월", y="yhat", title="📊 월별 매출 예측")
    col1, col2 = st.columns(2)
    col1.plotly_chart(fig1, use_container_width=True)
    col2.plotly_chart(fig2, use_container_width=True)

# --- 요약표 표시 ---
def display_summary_table(forecast_df):
    st.markdown("## 📊 예측 요약 (일자별 × 거래처별)")

    forecast_df = forecast_df.copy()
    forecast_df = forecast_df.sort_values("ds")
    pivot_df = forecast_df.pivot_table(index="ds", columns="거래처", values="yhat", aggfunc="sum").fillna(0)
    pivot_df["총합"] = pivot_df.sum(axis=1)

    display_df = pivot_df.copy()
    display_df = display_df.applymap(lambda x: f"{int(x):,}")
    st.dataframe(display_df.reset_index().rename(columns={"ds": "날짜"}), use_container_width=True)

    # 하단 합계
    st.markdown("### 📌 거래처별 예측 매출 합계")
    total_by_client = pivot_df.sum().drop("총합")
    total_all = pivot_df["총합"].sum()
    for client, total in total_by_client.items():
        st.markdown(f"- **{client}**: {int(total):,} 원")
    st.markdown(f"✅ **전체 합계**: **{int(total_all):,} 원**")

# --- 파일 다운로드 함수 ---
def convert_df_to_excel(df):
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name="Forecast")
    return buffer.getvalue()

# --- 실행 영역 ---
if uploaded_file:
    df_new = preprocess_excel(uploaded_file)
    if st.session_state.library_data is None:
        st.session_state.library_data = df_new
    else:
        st.session_state.library_data = pd.concat([st.session_state.library_data, df_new], ignore_index=True)
        st.session_state.library_data.drop_duplicates(subset=["ds", "거래처"], keep="last", inplace=True)

    original_data = st.session_state.library_data.copy()
    forecast_data = predict_sales(original_data, start_date, end_date)

    if forecast_data.empty:
        st.warning("❌ 예측 가능한 데이터가 부족합니다.")
    else:
        forecast_data["예측 매출"] = forecast_data["yhat"].map("{:,}".format)
        display_summary_table(forecast_data)
        st.markdown("## 📥 예측 결과 다운로드")
        excel_bytes = convert_df_to_excel(forecast_data)
        st.download_button("📥 예측 결과 다운로드", data=excel_bytes, file_name="forecast_result.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        plot_forecast(forecast_data)
else:
    st.info("👈 왼쪽에서 엑셀 파일을 업로드해주세요.")
