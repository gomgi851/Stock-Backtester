import pandas as pd
import os
import requests
import streamlit as st

def get_download_stock(market_type=None):
    """시장 타입에 따른 KRX 종목 리스트 다운로드 (User-Agent 추가)"""
    stock_type = {'kospi': 'stockMkt', 'kosdaq': 'kosdaqMkt'}
    market_url = stock_type[market_type]
    download_link = f'http://kind.krx.co.kr/corpgeneral/corpList.do?method=download&marketType={market_url}'
    
    # 💡 브라우저처럼 보이게 헤더 추가 (차단 방지)
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    
    try:
        response = requests.get(download_link, headers=headers, timeout=10)
        response.encoding = 'cp949'
        
        # 💡 lxml이나 html5lib이 없어도 동작하도록 엔진 지정 시도
        df_list = pd.read_html(response.text, header=0)
        if not df_list:
            return pd.DataFrame()
        return df_list[0]
    except Exception as e:
        print(f"네트워크 오류 또는 라이브러리 부재: {e}")
        return pd.DataFrame()

@st.cache_data
def get_krx_list():
    """코스피(.KS)와 코스닥(.KQ)을 구분하여 통합 리스트 생성"""
    file_path = 'krw_list.csv'
    
    if os.path.exists(file_path):
        return pd.read_csv(file_path, dtype={'ticker': str})
    
    try:
        # 1. 코스피 처리
        kospi_df = get_download_stock('kospi')
        if not kospi_df.empty:
            kospi_df['ticker'] = kospi_df['종목코드'].map('{:06d}.KS'.format)
        
        # 2. 코스닥 처리
        kosdaq_df = get_download_stock('kosdaq')
        if not kosdaq_df.empty:
            kosdaq_df['ticker'] = kosdaq_df['종목코드'].map('{:06d}.KQ'.format)
        
        # 3. 데이터 통합 확인
        if kospi_df.empty and kosdaq_df.empty:
            st.error("KRX 데이터를 가져오지 못했습니다. 인터넷 연결을 확인하세요.")
            return pd.DataFrame(columns=['name', 'ticker', 'display'])

        code_df = pd.concat([kospi_df, kosdaq_df])
        df = code_df[['회사명', 'ticker']].copy()
        df.columns = ['name', 'ticker']
        df['display'] = "🇰🇷 " + df['name'] + " (" + df['ticker'] + ")"
        
        # 4. 저장
        df.to_csv(file_path, index=False, encoding='utf-8-sig')
        return df
        
    except Exception as e:
        st.error(f"한국 주식 리스트 처리 중 에러: {e}")
        return pd.DataFrame(columns=['name', 'ticker', 'display'])

@st.cache_data
def get_us_list():
    """SEC 공식 데이터를 사용하여 미국 주식 리스트 생성 (User-Agent는 그대로 유지)"""
    file_path = 'us_stocks.csv'
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    
    try:
        url = "https://www.sec.gov/files/company_tickers.json"
        headers = {"User-Agent": "MyStockApp contact@example.com"}
        res = requests.get(url, headers=headers, timeout=10)
        data = res.json()
        raw_df = pd.DataFrame.from_dict(data, orient="index")
        df = raw_df[['title', 'ticker']].copy()
        df.columns = ['name', 'ticker']
        df['display'] = "🇺🇸 " + df['ticker'] + " - " + df['name']
        df.to_csv(file_path, index=False, encoding='utf-8-sig')
        return df
    except:
        return pd.DataFrame(columns=['name', 'ticker', 'display'])