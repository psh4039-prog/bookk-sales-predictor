# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
from prophet import Prophet
from io import BytesIO
import plotly.express as px
from datetime import datetime

st.set_page_config(page_title="부크크 매출 예측기", layout="wide")
st.title("📈 부크크 매출 예측기")

# --- 초기 세션 상태 ---
if "library_data" not in st.session_state:
    st.session_state.library_data = None

# --- 엑셀 업로드 ---
st.sidebar.header("1️⃣ 엑셀 업로드")
uploaded_file = st.sidebar.file_uploader("매출 데이터를 포함한 엑셀 파일 업로드", type=["xlsx"])

# --- 예측 기간 ---
st.sidebar.header("2️⃣ 예측 기간 선택")
start_date = st.sidebar.date_input("예측 시작일", datetime(2025, 9, 9))
end_date = st.sidebar.date_input("예측 종료일", datetime(2025, 12, 31))

# --- 엑셀 전처리 ---
def preprocess_excel(uploaded_file):
    df = pd.read_excel(uploaded_file, sheet_name=0, header=1)
    df.columns = df.columns.str.strip()
    date_col = [col for col in df.columns if '일자' in col or '날짜' in col or 'date' in col.lower()][0]
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
    """
    Prophet 모델을 기반으로 입력된 df 데이터에 대해 start_date ~ end_date까지 예측 수행
    - df: DataFrame (컬럼: ds, y, 거래처)
    - start_date, end_date: datetime 형식
    """
    import pandas as pd
    from prophet import Prophet

    # 날짜 형식 보장
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # Prophet 결과 저장 리스트
    all_forecasts = []

    # 전체 혹은 거래처별 그룹
    group_cols = df['거래처'].unique() if '거래처' in df.columns else [None]

    for client in group_cols:
        if client is not None:
            df_group = df[df['거래처'] == client][['ds', 'y']].copy()
        else:
            df_group = df[['ds', 'y']].copy()

        # 2개 미만의 유효 데이터는 예측 불가
        if df_group.dropna().shape[0] < 2:
            continue

        df_group = df_group.sort_values('ds')
        model = Prophet()
        model.fit(df_group)

        # 예측 기간 계산
        last_date = pd.to_datetime(df_group['ds'].max())
        period_days = max((end_date - last_date).days, 1)  # 최소 1일 이상 보장

        # 미래 예측 프레임 생성
        future = model.make_future_dataframe(periods=period_days, freq='D')
        forecast = model.predict(future)

        # 필요한 컬럼 추출
        result = forecast[['ds', 'yhat']].copy()
        result = result[result['ds'].between(start_date, end_date)]
        result['yhat'] = result['yhat'].round().astype(int)

        # ✅ 거래처 정보 반드시 추가
        if client is not None:
            result['거래처'] = client

        all_forecasts.append(result)

    # 예측 결과 통합
    df_result = pd.concat(all_forecasts).reset_index(drop=True) if all_forecasts else pd.DataFrame()

    return df_result


    # 과거 매출은 엑셀 데이터로 덮어쓰기
    df_merged = df_result.merge(df, on=["ds", "거래처"], how="left", suffixes=("_pred", ""))
    df_merged["yhat_final"] = df_merged["y"].combine_first(df_merged["yhat_pred"])
    df_merged.drop(columns=["y", "yhat_pred"], inplace=True)

    return df_merged

# --- 일자별 요약 표시 ---
def display_summary_table(forecast_df):
    st.subheader("📊 예측 요약 (일자별 × 거래처별)")

    if '거래처' not in forecast_df.columns:
        st.warning("⚠ 예측 결과에 '거래처' 정보가 없습니다. 거래처별 요약이 불가능합니다.")
        return

    # 날짜 정렬
    forecast_df = forecast_df.sort_values("ds")

    # 예측값 정수로 변환
    forecast_df['yhat'] = forecast_df['yhat'].round().astype(int)

    # 일자별 × 거래처별 피벗 테이블 생성
    pivot_df = forecast_df.pivot_table(index='ds', columns='거래처', values='yhat', aggfunc='sum').fillna(0)

    # 총합 열 추가
    pivot_df['총합'] = pivot_df.sum(axis=1)

    # 숫자 포맷 적용 (쉼표 단위로)
    display_df = pivot_df.copy()
    display_df = display_df.applymap(lambda x: f"{int(x):,}")

    # 표 출력
    st.dataframe(display_df.reset_index().rename(columns={'ds': '날짜'}), use_container_width=True)

    # 거래처별 합계 출력
    st.markdown("### 📌 거래처별 예측 매출 합계")
    total_by_client = pivot_df.drop(columns='총합').sum()
    total_all = pivot_df['총합'].sum()

    for client, total in total_by_client.items():
        st.markdown(f"- **{client}**: {int(total):,} 원")

    st.markdown(f"### ✅ 전체 합계: **{int(total_all):,} 원**")


# --- 실행 ---
if uploaded_file:
    df_new = preprocess_excel(uploaded_file)
    if st.session_state.library_data is None:
        st.session_state.library_data = df_new
    else:
        st.session_state.library_data = pd.concat([st.session_state.library_data, df_new], ignore_index=True)
        st.session_state.library_data.drop_duplicates(subset=['ds', '거래처'], keep='last', inplace=True)

    df_all = st.session_state.library_data
    forecast_df = predict_sales(df_all, start_date, end_date)
    display_summary_table(forecast_df)

    # 다운로드 기능
    csv = forecast_df.to_csv(index=False).encode('utf-8-sig')
    st.download_button(label="📥 예측 결과 다운로드", data=csv, file_name="sales_forecast.csv", mime="text/csv")

else:
    st.info("👈 왼쪽 사이드바에서 엑셀 파일을 업로드하세요.")
