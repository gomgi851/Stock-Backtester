import pandas as pd

url = 'https://kind.krx.co.kr/corpgeneral/corpList.do'
kosdaq = pd.read_html(url+"?method=download&marketType=kosdaqMkt", encoding='euc-kr')[0]
kospi = pd.read_html(url+"?method=download&marketType=stockMkt", encoding='euc-kr')[0]

# 종목코드 6자리 맞추고 접미사 붙이기
kosdaq['종목코드'] = kosdaq['종목코드'].astype(str).str.zfill(6) + '.KQ'
kospi['종목코드'] = kospi['종목코드'].astype(str).str.zfill(6) + '.KS'
print("코스닥:", kosdaq.head())
print("코스피:", kospi.head()) 
print("코스닥 종목 수:", len(kosdaq))