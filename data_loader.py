import pandas as pd
import os
import streamlit as st
import requests

@st.cache_data
def get_krx_list():
    file_path = 'krx_list.csv'
    
    if os.path.exists(file_path):
        return pd.read_csv(file_path, dtype={'ticker': str})
    
    try:
        # 핵심: 브라우저인 척 속이는 헤더 추가
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        url = 'https://kind.krx.co.kr/corpgeneral/corpList.do'
        
        # 코스닥 데이터 가져오기
        res_kosdaq = requests.get(url + "?method=download&marketType=kosdaqMkt", headers=headers)
        kosdaq = pd.read_html(res_kosdaq.text)[0]
        
        # 코스피 데이터 가져오기
        res_kospi = requests.get(url + "?method=download&marketType=stockMkt", headers=headers)
        kospi = pd.read_html(res_kospi.text)[0]

        kosdaq['ticker'] = kosdaq['종목코드'].astype(str).str.zfill(6) + '.KQ'
        kospi['ticker'] = kospi['종목코드'].astype(str).str.zfill(6) + '.KS'

        df = pd.concat([kosdaq, kospi], ignore_index=True)
        df = df[['회사명', 'ticker']].copy()
        df.columns = ['name', 'ticker']
        df['display'] = "🇰🇷 " + df['name'] + " (" + df['ticker'] + ")"
        
        # 서버 환경에서도 다음에 안 불러오도록 저장
        df.to_csv(file_path, index=False, encoding='utf-8-sig')
        return df
        
    except Exception as e:
        # 에러 발생 시 앱이 멈추지 않게 빈 데이터프레임이라도 반환
        st.error(f"한국 주식 리스트 로드 실패 (403 방지 필요): {e}")
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