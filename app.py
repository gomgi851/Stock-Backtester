import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
from data_loader import get_krx_list, get_us_list

# --- 1. 설정 및 데이터 캐싱 로직 ---
st.set_page_config(layout="wide", page_title="PR vs TR Backtester")

@st.cache_data(ttl=3600)
def get_stock_data(ticker, start_date, end_date):
    """주가 및 배당 데이터를 가져와 캐싱함"""
    obj = yf.Ticker(ticker)
    # 분석 기간용 데이터
    hist = obj.history(start=start_date, end=end_date)
    # 배당 히스토리용 (최대 15년)
    full_divs = obj.dividends
    return hist, full_divs

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
st.title('🚀 Multi-Asset PR vs TR Dashboard')
tab1, tab2, tab3 = st.tabs(['📊 분석 대시보드', '⚖️ 비중 설정기', '💰 배당 히스토리'])

if not selected_stocks:
    st.info('왼쪽 사이드바에서 종목을 선택해 주세요.')
else:
    # --- Tab 2: 비중 설정기 ---
    with tab2:
        st.subheader("📋 자산별 비중 설정")
        portfolio_weights = {}
        for stock in selected_stocks:
            c1, c2, c3, c4 = st.columns([0.5, 2, 1, 2])
            c1.write("🇰🇷" if stock['region'] == 'KR' else "🇺🇸")
            c2.write(f"**{stock['name']}**")
            c3.code(stock['ticker'], language=None)
            weight = c4.number_input("비중", 0.0, 100.0, 0.0, 5.0, key=f"w_{stock['ticker']}", label_visibility="collapsed")
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
        with st.spinner('데이터 계산 중...'):
            try:
                bench_data = yf.download(['^GSPC', '^KS11'], start=sd, end=ed, progress=False)['Close']
                bench_data.index = bench_data.index.tz_localize(None)
                
                pr_df = pd.DataFrame(index=bench_data.index)
                tr_df = pd.DataFrame(index=bench_data.index)
                
                for stock in selected_stocks:
                    raw, _ = get_stock_data(stock['ticker'], sd, ed)
                    if not raw.empty:
                        raw.index = raw.index.tz_localize(None)
                        pr_df[stock['ticker']] = (raw['Close'] / raw['Close'].iloc[0]) * 100
                        daily_tr_ratio = (raw['Close'] + raw['Dividends']) / raw['Close'].shift(1)
                        daily_tr_ratio.iloc[0] = 1.0
                        tr_df[stock['ticker']] = daily_tr_ratio.cumprod() * 100
                
                combined = pd.concat([bench_data, pr_df.add_suffix('_PR'), tr_df.add_suffix('_TR')], axis=1).ffill().dropna()
                norm_df = (combined / combined.iloc[0]) * 100
                
                display_df = pd.DataFrame(index=norm_df.index)
                if '^GSPC' in norm_df.columns: display_df['S&P 500'] = norm_df['^GSPC']
                if '^KS11' in norm_df.columns: display_df['KOSPI'] = norm_df['^KS11']
                
                total_input_w = sum(portfolio_weights.values())
                if total_input_w > 0:
                    pf_pr = pd.Series(0.0, index=norm_df.index)
                    pf_tr = pd.Series(0.0, index=norm_df.index)
                    for t, w in portfolio_weights.items():
                        weight_factor = w / total_input_w
                        if f"{t}_PR" in norm_df.columns:
                            pf_pr += norm_df[f"{t}_PR"] * weight_factor
                            pf_tr += norm_df[f"{t}_TR"] * weight_factor
                    display_df['My Portfolio (PR)'] = pf_pr
                    display_df['My Portfolio (TR)'] = pf_tr

                fig = go.Figure()
                colors = {'S&P 500': '#888888', 'KOSPI': '#cccccc', 'My Portfolio (PR)': '#00d2ff', 'My Portfolio (TR)': '#ff1616'}
                for col in display_df.columns:
                    is_main = 'My Portfolio' in col
                    fig.add_trace(go.Scatter(
                        x=display_df.index, y=display_df[col], name=col,
                        line=dict(color=colors.get(col, '#ff1616'), width=3 if is_main else 1.5,
                                  dash='dot' if 'PR' in col and is_main else None)
                    ))
                fig.update_layout(template="plotly_white", hovermode="x unified", legend=dict(orientation="h", y=1.05, x=1, xanchor="right"))
                st.plotly_chart(fig, use_container_width=True)

                if total_input_w == 100:
                    st.markdown("---")
                    st.subheader("📋 성과 요약 (Price Return 기준)")
                    m1, m2, m3 = st.columns(3)
                    my_pr_val = display_df['My Portfolio (PR)'].iloc[-1]
                    my_tr_val = display_df['My Portfolio (TR)'].iloc[-1]
                    years = (display_df.index[-1] - display_df.index[0]).days / 365.25
                    my_pr_ret = my_pr_val - 100
                    my_pr_cagr = ((my_pr_val/100)**(1/years)-1)*100 if years > 0 else 0
                    div_only_ret = (my_tr_val / my_pr_val - 1) * 100 

                    m1.metric("내 포트폴리오 (PR)", f"{my_pr_ret:.2f}%", f"CAGR {my_pr_cagr:.2f}%")
                    st.write(f"🎁 배당 재투자 추가 수익률: **{div_only_ret:.2f}%**")
                    m2.metric("S&P 500 대비 (PR)", f"{display_df['S&P 500'].iloc[-1]-100:.2f}%")
                    m3.metric("KOSPI 대비 (PR)", f"{display_df['KOSPI'].iloc[-1]-100:.2f}%")

            except Exception as e:
                st.error(f"오류 발생: {e}")

    # --- Tab 3: 배당 히스토리 (연도별 15년) ---
    with tab3:
        st.subheader("💰 연도별 배당금 합계 (최근 15년)")
        target_stock = st.selectbox("종목 선택", options=[s['name'] for s in selected_stocks], key="div_year_selector")
        target_ticker = next(s['ticker'] for s in selected_stocks if s['name'] == target_stock)
        
        try:
            _, full_divs = get_stock_data(target_ticker, sd, ed) # 캐시된 전체 배당 데이터 사용
            
            if not full_divs.empty:
                full_divs.index = full_divs.index.tz_localize(None)
                start_year = datetime.now().year - 14
                filtered_divs = full_divs[full_divs.index.year >= start_year]

                if not filtered_divs.empty:
                    df_div = filtered_divs.reset_index()
                    df_div.columns = ['Date', 'Dividends']
                    df_div['Year'] = df_div['Date'].dt.year
                    yearly_summary = df_div.groupby('Year')['Dividends'].sum().reset_index()
                    yearly_summary['Year_Str'] = yearly_summary['Year'].astype(str)

                    fig_div = go.Figure(go.Bar(
                        x=yearly_summary['Year_Str'], y=yearly_summary['Dividends'],
                        marker_color='#2ECC71', text=yearly_summary['Dividends'].round(2), textposition='auto'
                    ))
                    fig_div.update_xaxes(type='category')
                    fig_div.update_layout(title=f"📅 {target_stock} 연간 배당 추이", template="plotly_white", height=500)
                    st.plotly_chart(fig_div, use_container_width=True)
                else:
                    st.warning("최근 15년 내 배당 데이터가 없습니다.")
            else:
                st.warning("배당 정보가 없는 종목입니다.")
        except Exception as e:
            st.error(f"데이터 로드 오류: {e}")