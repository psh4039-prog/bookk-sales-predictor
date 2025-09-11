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
    date_col = next((col for col in df.columns if '일자' in col or '날짜' in col.lower()), None)
    if date_col is None:
        raise ValueError("엑셀 파일에 '일자' 또는 '날짜'라는 이름의 열이 존재하지 않습니다.")
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

    # ✅ 과거 날짜 조회 방지
    if end_date < df['ds'].max():
        st.warning("❌ 예측 종료일이 기존 데이터보다 과거입니다. 예측할 수 없습니다.")
        return pd.DataFrame()

    all_forecasts = []
    group_cols = df['거래처'].unique() if '거래처' in df.columns else [None]

    for client in group_cols:
        df_group = df[df['거래처'] == client][['ds', 'y']].copy()
        df_group = df_group.sort_values('ds')

        if client == '교보문고':
            df_group['ds'] = df_group['ds'].dt.to_period('M').dt.to_timestamp()
            df_month = df_group.groupby('ds')['y'].sum().reset_index()
            model = Prophet()
            model.fit(df_month)
            periods = max(1, (end_date.to_period('M') - df_month['ds'].max().to_period('M')).n + 1)
            future = model.make_future_dataframe(periods=periods, freq='MS')
            forecast = model.predict(future)[['ds', 'yhat']]
            forecast = forecast[forecast['ds'].between(start_date, end_date)]
            forecast['yhat'] = forecast['yhat'].clip(lower=0).round()

            # 월 매출을 일별로 분배
            expanded = []
            for _, row in forecast.iterrows():
                month = row['ds'].to_period('M').to_timestamp()
                days = pd.date_range(month, periods=month.days_in_month, freq='D')
                val_per_day = row['yhat'] / len(days)
                temp = pd.DataFrame({'ds': days, 'yhat': val_per_day})
                expanded.append(temp)
            result = pd.concat(expanded)
            result = result[result['ds'].between(start_date, end_date)]

        else:
            model = Prophet()
            model.fit(df_group)
            periods = max(1, (end_date - df_group['ds'].max()).days + 1)
            future = model.make_future_dataframe(periods=periods, freq='D')
            forecast = model.predict(future)[['ds', 'yhat']]
            result = forecast[forecast['ds'].between(start_date, end_date)]
            result['yhat'] = result['yhat'].clip(lower=0).round()

        result['거래처'] = client
        all_forecasts.append(result)

    df_result = pd.concat(all_forecasts)
    return df_result

# --- 예측 요약 ---
# --- 예측 요약 ---
def display_daily_summary(df_result):
    if df_result.empty:
        return

    st.subheader("📊 예측 요약 (일자별)")
    df_result = df_result.sort_values("ds")

    # 피벗 테이블 생성
    pivot_df = df_result.pivot_table(index='ds', columns='거래처', values='yhat', aggfunc='sum').fillna(0)

    # 정확한 합계 계산 (숫자 기준)
    pivot_df['합계'] = pivot_df.sum(axis=1)

    # 표시용 복사본 생성
    display_df = pivot_df.copy()

    # 숫자 포맷 적용
    for col in display_df.columns:
        display_df[col] = display_df[col].apply(lambda x: f"{int(round(x)):,}")

    # 테이블 출력
    st.dataframe(display_df.reset_index().rename(columns={'ds': '날짜'}), use_container_width=True)

    # 하단 합계 출력
    st.markdown("### 📌 거래처별 예측 매출 합계")

    total_by_client = pivot_df.drop(columns='합계').sum()
    total_all = pivot_df['합계'].sum()

    for client, total in total_by_client.items():
        st.markdown(f"- **{client}**: {int(round(total)):,} 원")

    st.markdown(f"### ✅ 전체 합계: **{int(round(total_all)):,} 원**")


# --- 다운로드 버튼 ---
def download_excel(df_result):
    if df_result.empty:
        return

    df_copy = df_result.copy()
    df_copy['yhat'] = df_copy['yhat'].astype(int)
    df_copy['yhat'] = df_copy['yhat'].map("{:,}".format)
    df_copy.rename(columns={'ds': '날짜', 'yhat': '예측 매출'}, inplace=True)
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df_copy.to_excel(writer, index=False, sheet_name="예측 결과")
    st.download_button(
        label="📥 예측 결과 엑셀 다운로드",
        data=output.getvalue(),
        file_name="예측결과.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# --- 시각화 ---
def plot_forecast(df_result):
    if df_result.empty:
        return

    df_result['날짜'] = pd.to_datetime(df_result['ds'])
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📈 거래처별 일별 추이")
        fig = px.line(df_result, x='날짜', y='yhat', color='거래처',
                      labels={'yhat': '매출액'}, title="일별 추이")
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### 📊 거래처 비중 (월별 총합 기준)")
        df_result['월'] = df_result['날짜'].dt.to_period("M").astype(str)
        monthly = df_result.groupby(['월', '거래처'])['yhat'].sum().reset_index()
        fig2 = px.bar(monthly, x='월', y='yhat', color='거래처',
                      barmode='stack', text_auto=True,
                      labels={'yhat': '매출액'}, title="월별 추이")
        fig2.update_layout(height=350)
        st.plotly_chart(fig2, use_container_width=True)

# --- 실행 영역 ---
if uploaded_file:
    df_new = preprocess_excel(uploaded_file)
    if st.session_state.library_data is None:
        st.session_state.library_data = df_new
    else:
        st.session_state.library_data = pd.concat(
            [st.session_state.library_data, df_new],
            ignore_index=True
        ).drop_duplicates(subset=['ds', '거래처'], keep='last')

    st.success("✅ 데이터 업로드 및 누적 학습 완료")
    df_library = st.session_state.library_data.copy()

    forecast = predict_sales(df_library, start_date, end_date)
    display_daily_summary(forecast)
    plot_forecast(forecast)
    download_excel(forecast)
else:
    st.info("👈 왼쪽 사이드바에서 엑셀 파일을 업로드하세요.")
