
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
    # 엑셀 데이터 불러오기
    df = pd.read_excel(uploaded_file, sheet_name=0, header=1)

    # 모든 컬럼 이름에서 공백 제거
    df.columns = df.columns.str.strip()

    # '일자' 또는 '날짜' 컬럼 탐색
    date_col = None
    for col in df.columns:
        if '일자' in col or '날짜' in col or 'date' in col.lower():
            date_col = col
            break

    if date_col is None:
        raise ValueError("엑셀 파일에 '일자' 또는 '날짜'라는 이름의 열이 존재하지 않습니다.")

    # 일자 컬럼을 datetime으로 변환
    df[date_col] = pd.to_datetime(df[date_col])

    # melt 구조로 변환 (일자를 제외한 나머지 컬럼은 거래처)
    clients = df.columns.drop(date_col)
    df_melted = df.melt(id_vars=date_col, value_vars=clients, var_name='거래처', value_name='매출액')

    # 결측값 제거
    df_melted.dropna(subset=['매출액'], inplace=True)

    # 컬럼명 Prophet 형식으로 맞추기
    df_melted.rename(columns={date_col: 'ds', '매출액': 'y'}, inplace=True)
    df_melted['y'] = pd.to_numeric(df_melted['y'], errors='coerce')
df_melted.dropna(subset=['y'], inplace=True)
df_melted['y'] = df_melted['y'].round()
    return df_melted

# --- 예측 함수 ---
def predict_sales(df, start_date, end_date):
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=(end_date - df['ds'].max()).days, freq='D')
    forecast = model.predict(future)
    forecast_filtered = forecast[(forecast['ds'] >= pd.to_datetime(start_date)) & (forecast['ds'] <= pd.to_datetime(end_date))]
    return forecast_filtered[['ds', 'yhat']]

# --- 시각화 함수 ---
def plot_forecast(forecast):
    # 일별 추이
    fig1 = px.line(forecast, x='ds', y='yhat', title='일별 매출 예측', labels={'ds': '날짜', 'yhat': '예측 매출액'})
    fig1.update_layout(width=600, height=400)

    # 월별 추이
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

# --- 실행 영역 ---
if uploaded_file:
    df_new = preprocess_excel(uploaded_file)

    # 최초 업로드라면 세션에 저장, 아니면 누적
    if st.session_state.library_data is None:
        st.session_state.library_data = df_new
    else:
        st.session_state.library_data = pd.concat([st.session_state.library_data, df_new], ignore_index=True)
        st.session_state.library_data.drop_duplicates(subset='ds', keep='last', inplace=True)

    st.success("✅ 데이터 업로드 및 누적 학습 완료")
    df_library = st.session_state.library_data.copy()

    forecast = predict_sales(df_library, start_date, end_date)
    forecast['예측 매출'] = forecast['yhat'].astype(int).map("{:,}".format)

    st.subheader("📊 예측 요약 (일자별)")
    st.dataframe(forecast[['ds', '예측 매출']].rename(columns={'ds': '날짜'}), use_container_width=True)

    st.subheader("📈 예측 그래프")
    plot_forecast(forecast)
else:
    st.info("👈 왼쪽 사이드바에서 엑셀 파일을 업로드하세요.")

# 시트 자동 감지
xls = pd.ExcelFile(uploaded_file)
sheet_name = xls.sheet_names[0]  # 첫 시트 이름 가져오기
df = pd.read_excel(xls, sheet_name=sheet_name)
