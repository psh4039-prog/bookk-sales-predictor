import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.express as px

# 페이지 설정
st.set_page_config(page_title="부크크 매주 예측기", layout="wide")
st.title("\ud83d\udcc8 \ubd80\ud06c\ud06c \ub9e4주 \uc608\uce21\uae30")

# --- 초기 상태 설정 ---
if "library_data" not in st.session_state:
    st.session_state.library_data = None

# --- 엘콠 파일 업로드 ---
st.sidebar.header("1️⃣ 엘콠 업로드")
uploaded_file = st.sidebar.file_uploader("매주 데이터를 포함한 엘콠 파일을 업로드하세요", type=["xlsx"])

# --- 예측 기간 선택 ---
st.sidebar.header("2️⃣ 예측 기간 선택")
start_date = st.sidebar.date_input("예측 시작일", pd.to_datetime("today"))
end_date = st.sidebar.date_input("예측 종료일", pd.to_datetime("2025-12-31"))

# --- 전체 회사 매주정보 전처리 ---
def preprocess_excel(uploaded_file):
    df = pd.read_excel(uploaded_file, sheet_name=0, header=1)
    df.columns = df.columns.str.strip()

    date_col = None
    for col in df.columns:
        if '일자' in col or '날짜' in col or 'date' in col.lower():
            date_col = col
            break

    if date_col is None:
        raise ValueError("엑셀 파일에 '일자' 또는 '날짜'라는 열이 존재하지 않습니다.")

    df[date_col] = pd.to_datetime(df[date_col])
    clients = df.columns.drop(date_col)
    df_melted = df.melt(id_vars=date_col, value_vars=clients, var_name='거래처', value_name='매출액')
    df_melted.dropna(subset=['매출액'], inplace=True)
    df_melted.rename(columns={date_col: 'ds', '매출액': 'y'}, inplace=True)
    df_melted['y'] = pd.to_numeric(df_melted['y'], errors='coerce')
    df_melted.dropna(subset=['y'], inplace=True)
    df_melted['y'] = df_melted['y'].round()
    return df_melted

# --- Prophet 예측 함수 ---
def predict_sales(df, start_date, end_date):
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    all_forecasts = []

    group_cols = df['거래처'].unique() if '거래처' in df.columns else [None]

    for client in group_cols:
        df_group = df[df['거래처'] == client][['ds', 'y']].copy() if client is not None else df[['ds', 'y']].copy()
        df_group = df_group.sort_values('ds')
        model = Prophet()
        model.fit(df_group)

        last_date = pd.to_datetime(df_group['ds'].max())
        period_days = max((end_date - last_date).days, 1)
        future = model.make_future_dataframe(periods=period_days, freq='D')

        forecast = model.predict(future)
        result = forecast[['ds', 'yhat']].copy()
        result = result[result['ds'].between(start_date, end_date)]
        result['yhat'] = result['yhat'].round().astype(int)

        if client is not None:
            result['거래처'] = client

        all_forecasts.append(result)

    df_result = pd.concat(all_forecasts).reset_index(drop=True)
    return df_result

# --- 일자별 예측 통계 표시 ---
def display_daily_summary(df_result):
    st.subheader("\ud83d\udcca \uc608\uce21 \uc694\uc57d (\uc77c\uc790\ubcc4)")

    df_result = df_result.sort_values("ds")
    pivot_df = df_result.pivot_table(index='ds', columns='거래처', values='yhat', aggfunc='sum').fillna(0)
    pivot_df['예측 매출'] = pivot_df.sum(axis=1)

    display_df = pivot_df.copy()
    for col in display_df.columns:
        display_df[col] = display_df[col].apply(lambda x: f"{int(x):,}")

    st.dataframe(display_df.reset_index().rename(columns={'ds': '날짜'}), use_container_width=True)

    st.markdown("### \ud83d\udccc \uac70\ub9ac\ucc28\ubcc4 \uc608\uce21 \ub9e4\uc8fc \ud569\uacc4")
    total_by_client = pivot_df.sum().drop("예측 매출")
    total_all = pivot_df["예측 매출"].sum()

    for client, total in total_by_client.items():
        st.markdown(f"- **{client}**: {int(total):,} 원")
    st.markdown(f"### ✅ \uc804\uccb4 \ud569\uacc4: **{int(total_all):,} 원**")

# --- 그래프 구성 ---
def plot_forecast(forecast):
    forecast['month'] = forecast['ds'].dt.to_period('M')
    monthly = forecast.groupby(['month', '거래처'])['yhat'].sum().reset_index()
    monthly['month'] = monthly['month'].astype(str)

    col1, col2 = st.columns(2)
    with col1:
        fig1 = px.line(forecast, x='ds', y='yhat', color='거래처', title='일별 매출 예측', labels={'ds': '날짜', 'yhat': '예측 매출액'})
        fig1.update_layout(width=600, height=400)
        st.plotly_chart(fig1)
    with col2:
        fig2 = px.bar(monthly, x='month', y='yhat', color='거래처', title='월별 매출 예측', labels={'month': '월', 'yhat': '예측 매출액'})
        fig2.update_layout(width=600, height=400)
        st.plotly_chart(fig2)

# --- 실행 구조 ---
if uploaded_file:
    df_new = preprocess_excel(uploaded_file)

    if st.session_state.library_data is None:
        st.session_state.library_data = df_new
    else:
        st.session_state.library_data = pd.concat([
            st.session_state.library_data, df_new
        ], ignore_index=True)
        st.session_state.library_data.drop_duplicates(subset='ds', keep='last', inplace=True)

    st.success("\u2705 \ub370\uc774\ud130 \uc5c5\ub85c\ub4dc \ubc0f \ub204주 학습 \uc644료")

    df_library = st.session_state.library_data.copy()
    forecast = predict_sales(df_library, start_date, end_date)
    forecast['예측 매출'] = forecast['yhat'].astype(int).map("{:,}".format)

    # ✅ 예측 요약 출력
    display_daily_summary(forecast)

    st.subheader("\ud83d\udcc8 \uc608\uce21 \uadf8\ub798\ud504")
    plot_forecast(forecast)
else:
    st.info("\ud83d\udc48 \uc67c쪽 \uc0ac이드바에서 \uc5d8콠 \ud30c일을 \uc5c5\ub85c\ub4dc하세요.")
