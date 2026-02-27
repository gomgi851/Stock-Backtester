import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
from data_loader import get_krx_list, get_us_list


# --- 1. 설정 및 데이터 로드 ---
st.set_page_config(layout="wide", page_title="Stock Backtester")

kr_list = get_krx_list()
us_list = get_us_list()

# --- 2. 사이드바 UI ---
with st.sidebar:
    st.title("⚙️ 설정 및 검색")
    
    st.header("📅 기간 선택")
    today = datetime.now().date()
    sd = st.date_input("시작 날짜", value=today - timedelta(days=365), max_value=today)
    ed = st.date_input("종료 날짜", value=today, max_value=today)
    
    date_error = False
    if sd >= ed:
        st.error("❌ 에러: 시작 날짜가 종료 날짜보다 빨라야 합니다.")
        date_error = True
    else:
        st.caption(f"선택 기간: {(ed-sd).days}일")

    st.markdown("---")

    if not date_error:
        st.header("🇰🇷 한국 주식")
        sel_kr = st.multiselect("종목 선택", options=kr_list['display'].tolist())

        st.header("🇺🇸 미국 주식")
        sel_us = st.multiselect("종목 선택", options=us_list['display'].tolist())
    else:
        sel_kr, sel_us = [], []

# --- 3. 선택 종목 파싱 ---
selected_stocks = []
for item in sel_kr:
    name = item.replace("🇰🇷 ", "").split(" (")[0]
    ticker = item.split(" (")[1].replace(")", "")
    selected_stocks.append({'name': name, 'ticker': ticker, 'region': 'KR'})

for item in sel_us:
    ticker = item.replace("🇺🇸 ", "").split(" - ")[0]
    name = item.split(" - ")[1]
    selected_stocks.append({'name': name, 'ticker': ticker, 'region': 'US'})

# --- 4. 메인 화면 ---
st.title('🚀 Multi-Asset Portfolio Dashboard')

tab1, tab2 = st.tabs(['📊 분석 대시보드', '⚖️ 비중 설정기'])

if not selected_stocks:
    st.info('왼쪽 사이드바에서 종목을 선택해 주세요.')
else:
    # --- Tab 2: 비중 설정기 ---
    with tab2:
        st.subheader("📋 자산별 비중 설정")
        portfolio_weights = {}
        
        for stock in selected_stocks:
            c1, c2, c3, c4 = st.columns([0.5, 2, 1, 2])
            with c1: st.write("🇰🇷" if stock['region'] == 'KR' else "🇺🇸")
            with c2: st.write(f"**{stock['name']}**")
            with c3: st.code(stock['ticker'], language=None)
            with c4:
                weight = st.number_input(
                    "비중", min_value=0.0, max_value=100.0, value=0.0, step=5.0,
                    key=f"w_{stock['ticker']}", label_visibility="collapsed"
                )
                portfolio_weights[stock['ticker']] = weight
            st.markdown("---")
        
        total_w = sum(portfolio_weights.values())
        if total_w != 100:
            st.warning(f"⚠️ 현재 비중 합계: {total_w}% (100%가 되도록 조정하세요)")
        else:
            st.success("✅ 비중 설정 완료")

    # --- Tab 1: 분석 대시보드 ---
    with tab1:
        st.subheader("📈 포트폴리오 성과 비교")
        
        with st.spinner('데이터를 계산 중입니다...'):
            try:
                # 1. 벤치마크와 선택 종목 티커 합치기
                benchmark_tickers = ['^GSPC', '^KS11']
                stock_tickers = [s['ticker'] for s in selected_stocks]
                all_tickers = benchmark_tickers + stock_tickers
                
                # 2. 데이터 다운로드
                raw_data = yf.download(all_tickers, start=sd, end=ed)['Close']
                
                if not raw_data.empty:
                    df = raw_data.copy()
                    df.index = df.index.tz_localize(None)
                    df = df.resample('D').last().ffill().dropna()
                    
                    # 3. 누적 수익률 정규화 (시작점 100)
                    norm_df = (df / df.iloc[0]) * 100
                    
                    total_input_w = sum(portfolio_weights.values())
                    display_df = pd.DataFrame(index=norm_df.index)
                    
                    if '^GSPC' in norm_df.columns:
                        display_df['S&P 500'] = norm_df['^GSPC']
                    if '^KS11' in norm_df.columns:
                        display_df['KOSPI'] = norm_df['^KS11']
                    
                    if total_input_w > 0:
                        weighted_sum = pd.Series(0.0, index=norm_df.index)
                        for stock in selected_stocks:
                            t = stock['ticker']
                            w = portfolio_weights.get(t, 0) / total_input_w
                            weighted_sum += norm_df[t] * w
                        
                        display_df['My Portfolio'] = weighted_sum

                        # 4. Plotly 그래프 (y축 꽉 차게)
                        fig = go.Figure()
                        color_map = {
                            'S&P 500': '#888888',   # 진한 회색
                            'KOSPI':   '#cccccc',   # 연한 회색
                            'My Portfolio': '#ff1616'  # 기본 파란색
                        }

                        for col in display_df.columns:
                            fig.add_trace(go.Scatter(
                                x=display_df.index, y=display_df[col],
                                name=col, mode='lines',
                                line=dict(color=color_map.get(col, '#ff1616'))
                            ))

                        y_min = display_df.min().min()
                        y_max = display_df.max().max()
                        margin = (y_max - y_min) * 0.03

                        fig.update_layout(
                            yaxis=dict(range=[y_min - margin, y_max + margin]),
                            xaxis_title="날짜",
                            yaxis_title="수익률 (100 기준)",
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # 5. 하단 성과 지표
                        st.markdown("---")
                        
                        # 계산을 위한 기초 변수
                        days = (display_df.index[-1] - display_df.index[0]).days
                        # 0일인 경우 에러 방지
                        years = days / 365 if days > 0 else 1
                        
                        m1, m2, m3 = st.columns(3)
                        
                        # 누적 수익률 계산
                        my_final_val = display_df['My Portfolio'].iloc[-1]
                        my_ret = my_final_val - 100
                        
                        # CAGR 계산 (시작값이 100이므로 종료값/100)
                        my_cagr = ((my_final_val / 100) ** (1/years) - 1) * 100
                        
                        sp_ret = display_df['S&P 500'].iloc[-1] - 100 if 'S&P 500' in display_df.columns else 0
                        ko_ret = display_df['KOSPI'].iloc[-1] - 100 if 'KOSPI' in display_df.columns else 0
                        
                        # 지표 출력
                        with m1:
                            st.metric("내 포트폴리오 누적 수익률", f"{my_ret:.2f}%")
                            st.caption(f"📅 연평균 수익률(CAGR): **{my_cagr:.2f}%**")
                        
                        with m2:
                            st.metric("S&P 500 대비", f"{sp_ret:.2f}%", f"{my_ret - sp_ret:.2f}%")
                            st.caption(f"S&P 500 누적 성과")
                            
                        with m3:
                            st.metric("KOSPI 대비", f"{ko_ret:.2f}%", f"{my_ret - ko_ret:.2f}%")
                            st.caption(f"KOSPI 누적 성과")
                    else:
                        st.warning("⚠️ '비중 설정기' 탭에서 비중을 먼저 설정해주세요!")

                        # 비중 없을 때도 Plotly로 출력
                        fig = go.Figure()
                        color_map = {
                            'S&P 500': '#888888',   # 진한 회색
                            'KOSPI':   '#cccccc',   # 연한 회색
                            'My Portfolio': "#ff1616"  # 기본 파란색
                        }

                        for col in display_df.columns:
                            fig.add_trace(go.Scatter(
                                x=display_df.index, y=display_df[col],
                                name=col, mode='lines',
                                line=dict(color=color_map.get(col, '#ff1616'))
                            ))
                        y_min = display_df.min().min()
                        y_max = display_df.max().max()
                        margin = (y_max - y_min) * 0.03

                        fig.update_layout(
                            yaxis=dict(range=[y_min - margin, y_max + margin]),
                            xaxis_title="날짜",
                            yaxis_title="수익률 (100 기준)",
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                        )
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("데이터를 가져오는 데 실패했습니다.")
            except Exception as e:
                st.error(f"분석 중 오류 발생: {e}")

        st.markdown("---")
        st.subheader("📝 선택된 종목 요약")
        cols = st.columns(4)
        for i, stock in enumerate(selected_stocks):
            with cols[i % 4]:
                st.metric(label=f"{stock['ticker']}", value=stock['name'][:15])