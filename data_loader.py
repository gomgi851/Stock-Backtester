import pandas as pd
import os
import streamlit as st

import pandas as pd
import os
import streamlit as st
import requests
from io import BytesIO

@st.cache_data
def get_krx_list():
    # [수정 1] 파일 경로를 현재 파일(data_loader.py) 위치 기준으로 절대 경로화
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(BASE_DIR, 'krx_list.csv')
    
    # [수정 2] 파일이 있으면 무조건 파일부터 읽기 (403 방지)
    if os.path.exists(file_path):
        try:
            return pd.read_csv(file_path, dtype={'ticker': str})
        except Exception as e:
            st.warning(f"파일은 있으나 읽기 실패: {e}")

    # [수정 3] 파일이 없을 때만 크롤링 시도 (headers 추가로 403 우회)
    try:
        url = 'https://kind.krx.co.kr/corpgeneral/corpList.do'
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36'
        }
        
        dfs = []
        for m_type in ["stockMkt", "kosdaqMkt"]:
            params = {'method': 'download', 'marketType': m_type}
            res = requests.get(url, params=params, headers=headers, timeout=10)
            # 403 에러 발생 시 여기서 멈추지 않고 에러 메시지를 명확히 함
            res.raise_for_status() 
            
            temp_df = pd.read_html(BytesIO(res.content), encoding='euc-kr')[0]
            suffix = '.KS' if m_type == "stockMkt" else '.KQ'
            temp_df['ticker'] = temp_df['종목코드'].astype(str).str.zfill(6) + suffix
            dfs.append(temp_df)

        df = pd.concat(dfs, ignore_index=True)
        df = df[['회사명', 'ticker']].copy()
        df.columns = ['name', 'ticker']
        df['display'] = "🇰🇷 " + df['name'] + " (" + df['ticker'] + ")"
        
        # 성공하면 다음에 안 막히게 저장
        df.to_csv(file_path, index=False, encoding='utf-8-sig')
        return df
        
    except Exception as e:
        # 최종 실패 시 403 에러가 나더라도 앱이 멈추지 않게 기본 데이터 반환
        st.error(f"한국 주식 리스트 로드 실패: {e}")
        return pd.DataFrame({
            'name': ['삼성전자'], 
            'ticker': ['005930.KS'], 
            'display': ['🇰🇷 삼성전자 (005930.KS)']
        })

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