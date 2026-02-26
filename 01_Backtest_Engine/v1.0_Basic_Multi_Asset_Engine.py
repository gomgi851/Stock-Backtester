import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import requests

# 오늘의 환율 가져오기 (USD to KRW)
def get_today_exchange_rate():
    url = "https://api.exchangerate-api.com/v4/latest/USD"
    response = requests.get(url)
    data = response.json()
    return data['rates']['KRW']  # USD 대비 KRW 환율

# 포트폴리오 구성 (한국 주식은 원화, 미국 주식은 USD)
portfolio = {'138040.KS': 100, '139480.KS': 100, 'HOOD': 100, 'NVDA' : 50}
start_date = '2023-01-01'
end_date = '2025-01-01'

# 오늘의 환율 적용
exchange_rate = get_today_exchange_rate()

# 포트폴리오 가격 데이터 가져오기
portfolio_prices = pd.DataFrame()
for ticker, shares in portfolio.items():
    data = yf.download(ticker, start=start_date, end=end_date)['Close']
    portfolio_prices[ticker] = data

# 환율 적용 (미국 주식은 USD를 KRW로 변환)
portfolio_prices['HOOD'] *= exchange_rate
portfolio_prices['NVDA'] *= exchange_rate

# 포트폴리오 가격 데이터를 주식 수로 곱하여 포트폴리오 가치 계산
portfolio_values = portfolio_prices.mul(portfolio)

# 포트폴리오 총 가치 계산
portfolio_values['Total'] = portfolio_values.sum(axis=1)

# S&P 500과 KOSPI 지수 데이터 가져오기
sp500 = yf.download('^GSPC', start=start_date, end=end_date)['Close'].squeeze()
kospi = yf.download('^KS11', start=start_date, end=end_date)['Close'].squeeze()

# 모든 데이터를 하나의 DataFrame으로 합치기
comparison = pd.DataFrame({
    'Portfolio': portfolio_values['Total'],
    'S&P 500': sp500,
    'KOSPI': kospi
})

# 한국과 미국의 휴장일을 고려하여 데이터 정렬
# 두 시장 모두 열린 날짜만 선택
comparison = comparison.dropna()  # 누락된 데이터 제거

# 데이터 정규화 (첫 날 값을 100으로 설정)
normalized_comparison = comparison / comparison.iloc[0] * 100

# 그래프 그리기
plt.figure(figsize=(14, 7))
plt.plot(normalized_comparison['Portfolio'], label='Portfolio')
plt.plot(normalized_comparison['S&P 500'], label='S&P 500')
plt.plot(normalized_comparison['KOSPI'], label='KOSPI')
plt.title('Portfolio vs S&P 500 vs KOSPI (환율 고려, 휴장일 제외)')
plt.xlabel('Date')
plt.ylabel('Normalized Price')
plt.legend()
plt.grid()
plt.show()