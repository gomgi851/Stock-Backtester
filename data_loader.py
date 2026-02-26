import pandas as pd
import os
import streamlit as st

@st.cache_data
def get_krx_list():
    """제공해주신 최적화 로직: 코스피(.KS)와 코스닥(.KQ) 통합 리스트 생성"""
    file_path = 'krx_list.csv'
    
    # 기존 파일이 있으면 즉시 로드
    if os.path.exists(file_path):
        return pd.read_csv(file_path, dtype={'ticker': str})
    
    try:
        url = 'https://kind.krx.co.kr/corpgeneral/corpList.do'
        
        # 1. 코스닥 및 코스피 데이터 읽기 (euc-kr 인코딩 적용)
        kosdaq = pd.read_html(url + "?method=download&marketType=kosdaqMkt", encoding='euc-kr')[0]
        kospi = pd.read_html(url + "?method=download&marketType=stockMkt", encoding='euc-kr')[0]

        # 2. 종목코드 6자리 맞추고 접미사 붙이기 (.KQ / .KS)
        kosdaq['ticker'] = kosdaq['종목코드'].astype(str).str.zfill(6) + '.KQ'
        kospi['ticker'] = kospi['종목코드'].astype(str).str.zfill(6) + '.KS'

        # 3. 데이터 통합 및 정리
        df = pd.concat([kosdaq, kospi], ignore_index=True)
        df = df[['회사명', 'ticker']].copy()
        df.columns = ['name', 'ticker']
        
        # 4. Streamlit 검색용 display 컬럼 생성
        df['display'] = "🇰🇷 " + df['name'] + " (" + df['ticker'] + ")"
        
        # 5. CSV 저장 (한글 깨짐 방지 utf-8-sig)
        df.to_csv(file_path, index=False, encoding='utf-8-sig')
        
        return df
        
    except Exception as e:
        st.error(f"한국 주식 리스트 로드 실패: {e}")
        return pd.DataFrame(columns=['name', 'ticker', 'display'])

@st.cache_data
def get_us_list():
    """SEC 데이터를 사용한 미국 주식 리스트 생성 (기존 유지)"""
    file_path = 'us_stocks.csv'
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        if 'display' in df.columns: return df
    
    import requests
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