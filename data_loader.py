import pandas as pd
import os
import streamlit as st
import requests
from io import BytesIO

@st.cache_data
def get_krx_list():
    file_path = 'krx_list.csv'
    
    if os.path.exists(file_path):
        return pd.read_csv(file_path, dtype={'ticker': str})
    
    try:
        url = 'https://kind.krx.co.kr/corpgeneral/corpList.do'
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36'
        }
        
        # 1. 코스닥/코스피 데이터를 각각 가져오기
        # marketType: stockMkt(유가증권), kosdaqMkt(코스닥)
        dfs = []
        for m_type in ["stockMkt", "kosdaqMkt"]:
            params = {
                'method': 'download',
                'marketType': m_type
            }
            # verify=False는 SSL 인증서 에러 방지용 (필요시)
            res = requests.get(url, params=params, headers=headers, timeout=10)
            
            # [핵심] read_html에 StringIO 대신 BytesIO를 사용하여 인코딩 문제를 방지하고
            # flavor='bs4' 또는 'lxml'을 명시합니다.
            try:
                # KRX 테이블 데이터는 'euc-kr'인 경우가 많습니다.
                temp_df = pd.read_html(BytesIO(res.content), encoding='euc-kr')[0]
                
                # 티커 포맷 정리
                suffix = '.KS' if m_type == "stockMkt" else '.KQ'
                temp_df['ticker'] = temp_df['종목코드'].astype(str).str.zfill(6) + suffix
                dfs.append(temp_df)
            except Exception as inner_e:
                print(f"{m_type} 로드 중 내부 오류: {inner_e}")
                continue

        if not dfs:
            raise ValueError("데이터를 하나도 가져오지 못했습니다. (No tables found)")

        # 2. 통합 및 정리
        df = pd.concat(dfs, ignore_index=True)
        df = df[['회사명', 'ticker']].copy()
        df.columns = ['name', 'ticker']
        df['display'] = "🇰🇷 " + df['name'] + " (" + df['ticker'] + ")"
        
        # 3. CSV 저장
        df.to_csv(file_path, index=False, encoding='utf-8-sig')
        return df
        
    except Exception as e:
        st.error(f"한국 주식 리스트 로드 최종 실패: {e}")
        # 최후의 수단: 앱이 멈추지 않도록 샘플 데이터 반환
        return pd.DataFrame({
            'name': ['삼성전자', 'SK하이닉스'],
            'ticker': ['005930.KS', '000660.KS'],
            'display': ['🇰🇷 삼성전자 (005930.KS)', '🇰🇷 SK하이닉스 (000660.KS)']
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