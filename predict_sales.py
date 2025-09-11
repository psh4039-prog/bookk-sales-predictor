
# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.express as px
from io import BytesIO
from datetime import timedelta

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
    date_col = next((col for col in df.columns if "일자" in col or "날짜" in col or "date" in col.lower()), None)
    if date_col is None:
        raise ValueError("엑셀 파일에 '일자' 또는 '날짜'라는 열이 존재하지 않습니다.")
    df[date_col] = pd.to_datetime(df[date_col])
    clients = df.columns.drop(date_col)
    df_melted = df.melt(id_vars=date_col, value_vars=clients, var_name="거래처", value_name="매출액")
    df_melted.dropna(subset=["매출액"], inplace=True)
    df_melted.rename(columns={date_col: "ds", "매출액": "y"}, inplace=True)
    df_melted["y"] = pd.to_numeric(df_melted["y"], errors="coerce")
    df_melted.dropna(subset=["y"], inplace=True)
    df_melted["y"] = df_melted["y"].round()
    return df_melted

# --- 예측 함수 ---
def predict_sales(df, start_date, end_date):
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    all_forecasts = []
    clients = df["거래처"].unique()

    for client in clients:
        df_client = df[df["거래처"] == client][["ds", "y"]].sort_values("ds").copy()
        last_actual_date = df_client["ds"].max()

        if client == "교보문고":
            df_client["month"] = df_client["ds"].dt.to_period("M")
            df_monthly = df_client.groupby("month")["y"].sum().reset_index()
            df_monthly["ds"] = df_monthly["month"].dt.to_timestamp()
            df_monthly = df_monthly[["ds", "y"]]
            model = Prophet()
            model.fit(df_monthly)
            months = pd.date_range(start=df_monthly["ds"].max(), end=end_date, freq="MS")
            future = pd.DataFrame({"ds": months})
            forecast = model.predict(future)
            forecast["yhat"] = forecast["yhat"].round().astype(int)

            spread_result = []
            for _, row in forecast.iterrows():
                month_start = row["ds"]
                month_end = (month_start + pd.offsets.MonthEnd(1)).date()
                days_in_month = pd.date_range(month_start, month_end)
                daily_value = int(row["yhat"] / len(days_in_month))
                for day in days_in_month:
                    spread_result.append({"ds": day, "yhat": daily_value, "거래처": client})
            result = pd.DataFrame(spread_result)
        else:
            model = Prophet()
            model.fit(df_client)
            period_days = max((end_date - df_client["ds"].max()).days, 1)
            future = model.make_future_dataframe(periods=period_days, freq="D")
            forecast = model.predict(future)
            result = forecast[["ds", "yhat"]]
            result["yhat"] = result["yhat"].round().astype(int)
            result = result[result["ds"].between(start_date, end_date)]
            result["거래처"] = client

        df_actual = df_client[df_client["ds"].between(start_date, end_date)][["ds", "y"]].copy()
        df_actual["거래처"] = client
        df_actual.rename(columns={"y": "yhat"}, inplace=True)
        all_forecasts.append(pd.concat([df_actual, result[result["ds"] > df_actual["ds"].max()]]))

    return pd.concat(all_forecasts).reset_index(drop=True)

# --- 시각화 함수 ---
def plot_forecast(forecast):
    col1, col2 = st.columns(2)

    with col1:
        fig = px.line(forecast, x="ds", y="yhat", color="거래처", labels={"ds": "날짜", "yhat": "예측 매출액"}, title="일별 매출 추이")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        by_client = forecast.groupby("거래처")["yhat"].sum().reset_index()
        fig2 = px.pie(by_client, names="거래처", values="yhat", title="거래처별 전체 매출 비중")
        st.plotly_chart(fig2, use_container_width=True)

# --- 요약표 함수 ---
def display_daily_summary(df_result):
    st.subheader("📊 예측 요약 (일자별)")
    df_result = df_result.sort_values("ds")
    pivot_df = df_result.pivot_table(index="ds", columns="거래처", values="yhat", aggfunc="sum").fillna(0)
    if "Total" in pivot_df.columns:
        pivot_df.drop(columns="Total", inplace=True)
    pivot_df["예측 매출"] = pivot_df.sum(axis=1)

    display_df = pivot_df.copy()
    for col in display_df.columns:
        display_df[col] = display_df[col].apply(lambda x: f"{int(x):,}")
    st.dataframe(display_df.reset_index().rename(columns={"ds": "날짜"}), use_container_width=True)

    st.markdown("### 📌 거래처별 예측 매출 합계")
    total_by_client = pivot_df.sum().drop("예측 매출")
    total_all = pivot_df["예측 매출"].sum()
    for client, total in total_by_client.items():
        st.markdown(f"- **{client}**: {int(total):,} 원")
    st.markdown(f"### ✅ 전체 합계: **{int(total_all):,} 원**")

# --- 실행 영역 ---
if uploaded_file:
    df_new = preprocess_excel(uploaded_file)
    if st.session_state.library_data is None:
        st.session_state.library_data = df_new
    else:
        st.session_state.library_data = pd.concat([st.session_state.library_data, df_new], ignore_index=True)
        st.session_state.library_data.drop_duplicates(subset=["ds", "거래처"], keep="last", inplace=True)

    st.success("✅ 데이터 업로드 및 누적 학습 완료")
    df_library = st.session_state.library_data.copy()
    forecast = predict_sales(df_library, start_date, end_date)
    forecast["예측 매출"] = forecast["yhat"].astype(int).map("{:,}".format)

    display_daily_summary(forecast)

    st.subheader("📈 예측 그래프")
    plot_forecast(forecast)

    # --- 다운로드 버튼 ---
    st.subheader("📥 예측 결과 다운로드")
    csv = forecast.to_csv(index=False).encode("utf-8-sig")
    st.download_button("Download CSV", csv, "예측_결과.csv", "text/csv")
else:
    st.info("👈 왼쪽 사이드바에서 엑셀 파일을 업로드하세요.")
