import pandas as pd
import yfinance as yf
import requests
import matplotlib.pyplot as plt
import koreanize_matplotlib
from datetime import datetime
import numpy as np
from scipy.interpolate import make_interp_spline

class PortfolioBacktester:
    def __init__(self):
        self.portfolio = pd.DataFrame(columns=['ticker', 'shares'])
        self.url = "http://kind.krx.co.kr/corpgeneral/corpList.do?method=download"
        self.krx_list = pd.read_html(self.url, header=0, encoding='euc-kr')[0]  # 인코딩 지정
        self.krx_list['종목코드'] = self.krx_list['종목코드'].astype(str).str.zfill(6)
        print("---------- 한국거래소 종목코드 데이터 로드 완료 ----------")
        self.exchange_rate = self.get_exchange_rate()
        print("----------         환율 정보 로드 완료          ----------")

    def get_exchange_rate(self):
        """USD to KRW 환율을 API로부터 가져옵니다."""
        url = "https://api.exchangerate-api.com/v4/latest/USD"
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            return data['rates']['KRW']
        except Exception as e:
            print(f"환율 정보를 가져오는 데 실패했습니다: {e}")
            return None

    def get_korean_ticker(self, stock_name):
        """
        한국 주식의 종목명을 입력하면 티커를 반환합니다.
        """
        try:
            match = self.krx_list[self.krx_list['회사명'] == stock_name]
            if not match.empty:
                return match['종목코드'].values[0]
            else:
                print(f"한국 주식 {stock_name}의 티커를 찾을 수 없습니다.")
                return None
        except Exception as e:
            print(f"한국 주식 티커를 가져오는 데 실패했습니다: {e}")
            return None

    def get_us_ticker(self, stock_name):
        """
        미국 주식의 종목명을 입력하면 티커를 반환합니다.
        """
        try:
            search_result = yf.Ticker(stock_name)
            if search_result.info:
                return stock_name
            else:
                print(f"미국 주식 {stock_name}의 티커를 찾을 수 없습니다.")
                return None
        except Exception as e:
            print(f"미국 주식 티커 검색 중 오류 발생: {e}")
            return None

    def add_stock(self, ticker, shares):
        """포트폴리오에 주식을 추가합니다."""
        new_stock = pd.DataFrame({'ticker': [ticker], 'shares': [shares]})
        self.portfolio = pd.concat([self.portfolio, new_stock], ignore_index=True)

    def calculate_tr_index(self, price_data, dividend_data=None):
        """배당금을 재투자한 TR 지수를 계산합니다."""
        tr_index = price_data.copy()
        tr_index.iloc[0] = 1000  # 초기값 설정

        for i in range(1, len(tr_index)):
            # 배당금이 있는지 확인
            if dividend_data is not None and tr_index.index[i] in dividend_data.index:
                dividend = dividend_data.loc[tr_index.index[i]]
                tr_index.iloc[i] = tr_index.iloc[i-1] * (price_data.iloc[i] + dividend) / price_data.iloc[i-1]
            else:
                tr_index.iloc[i] = tr_index.iloc[i-1] * price_data.iloc[i] / price_data.iloc[i-1]

        return tr_index

    def run_backtest(self, start_date, end_date):
        """백테스팅을 실행하고 결과를 시각화합니다."""
        if self.exchange_rate is None:
            print("환율 정보가 없어 백테스팅을 진행할 수 없습니다.")
            return

        # 포트폴리오 가격 데이터 가져오기
        portfolio_prices = pd.DataFrame()
        portfolio_tr = pd.DataFrame()

        for _, stock in self.portfolio.iterrows():
            ticker = stock['ticker']
            shares = stock['shares']
            try:
                data = yf.download(ticker, start=start_date, end=end_date)
                price_data = data['Close']
                # 배당금 데이터가 있는지 확인
                if 'Dividends' in data.columns:
                    dividend_data = data['Dividends']
                else:
                    dividend_data = None  # 배당금 데이터가 없는 경우

                # TR 지수 계산
                tr_index = self.calculate_tr_index(price_data, dividend_data)

                portfolio_prices[ticker] = price_data
                portfolio_tr[ticker] = tr_index
            except Exception as e:
                print(f"{ticker}의 데이터를 가져오는 데 실패했습니다: {e}")
                continue


        # 환율 적용 (미국 주식은 USD를 KRW로 변환)
        for ticker in self.portfolio['ticker']:
            if not ticker.endswith('.KS'):  # 한국 주식이 아닌 경우 (미국 주식)
                portfolio_prices[ticker] *= self.exchange_rate
                portfolio_tr[ticker] *= self.exchange_rate

        # 포트폴리오 가격 데이터를 주식 수로 곱하여 포트폴리오 가치 계산
        portfolio_values = portfolio_prices.mul(self.portfolio.set_index('ticker')['shares'])
        portfolio_tr_values = portfolio_tr.mul(self.portfolio.set_index('ticker')['shares'])

        # 포트폴리오 총 가치 계산
        portfolio_values['Total'] = portfolio_values.sum(axis=1)
        portfolio_tr_values['Total'] = portfolio_tr_values.sum(axis=1)

        # S&P 500과 KOSPI 지수 데이터 가져오기
        try:
            sp500 = yf.download('^GSPC', start=start_date, end=end_date)['Close'].squeeze()
            kospi = yf.download('^KS11', start=start_date, end=end_date)['Close'].squeeze()
        except Exception as e:
            print(f"지수 데이터를 가져오는 데 실패했습니다: {e}")
            return

        # 모든 데이터를 하나의 DataFrame으로 합치기
        comparison = pd.DataFrame({
            'Portfolio': portfolio_values['Total'],
            'Portfolio_TR': portfolio_tr_values['Total'],
            'S&P 500': sp500,
            'KOSPI': kospi
        })

        # 누락된 데이터 제거
        print(comparison)
        comparison = comparison.dropna()
        print(comparison)
        # 데이터 정규화 (첫 날 값을 1000으로 설정)
        normalized_comparison = comparison / comparison.iloc[0] * 1000

        # 그래프 그리기
        plt.figure(figsize=(14, 7))

        # 날짜를 숫자로 변환 (x축)
        x = np.arange(len(normalized_comparison.index))  # 날짜를 숫자로 변환
        x_new = np.linspace(x.min(), x.max(), 300)  # 300개의 점으로 보간

        # 포트폴리오 (가격)
        spl_portfolio = make_interp_spline(x, normalized_comparison['Portfolio'], k=3)
        y_portfolio_smooth = spl_portfolio(x_new)
        plt.plot(pd.to_datetime(normalized_comparison.index[x_new.astype(int)]), y_portfolio_smooth, label='Portfolio (Price, Smoothed)', linewidth=3, color='skyblue')  # 보간된 데이터

        # 포트폴리오 (TR)
        spl_portfolio_tr = make_interp_spline(x, normalized_comparison['Portfolio_TR'], k=3)
        y_portfolio_tr_smooth = spl_portfolio_tr(x_new)
        plt.plot(pd.to_datetime(normalized_comparison.index[x_new.astype(int)]), y_portfolio_tr_smooth, label='Portfolio (TR, Smoothed)', linewidth=3, color='blue')  # 보간된 데이터

        # S&P 500
        spl_sp500 = make_interp_spline(x, normalized_comparison['S&P 500'], k=3)
        y_sp500_smooth = spl_sp500(x_new)
        plt.plot(pd.to_datetime(normalized_comparison.index[x_new.astype(int)]), y_sp500_smooth, label='S&P 500 (Smoothed)', linewidth=2, color='darkgray')  # 보간된 데이터

        # KOSPI
        spl_kospi = make_interp_spline(x, normalized_comparison['KOSPI'], k=3)
        y_kospi_smooth = spl_kospi(x_new)
        plt.plot(pd.to_datetime(normalized_comparison.index[x_new.astype(int)]), y_kospi_smooth, label='KOSPI (Smoothed)', linewidth=2, color='lightgrey')  # 보간된 데이터

        # 제목과 레이블 설정
        plt.title('Portfolio vs S&P 500 vs KOSPI (환율 고려, 휴장일 제외)', fontsize=16, fontweight='light', pad=20)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Normalized Price', fontsize=12)

        # 범례 설정
        plt.legend(fontsize=12, loc='upper left')

        # 그리드 설정
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # 레이아웃 조정
        plt.tight_layout()
        plt.savefig('portfolio_performance.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    backtester = PortfolioBacktester()

    # 백테스팅 기간 입력
    start_date = input("백테스팅 시작일을 입력하세요 (YYYY-MM-DD): ")
    end_date = input("백테스팅 종료일을 입력하세요 (YYYY-MM-DD): ")

    # 주식 정보 입력
    while True:
        stock_input = input("주식 이름과 수량을 입력하세요 (예: 메리츠금융지주 20 KRW), 종료하려면 -1을 입력하세요: ")
        if stock_input == "-1":
            break
        try:
            stock_name, shares, currency = stock_input.rsplit(' ')
            shares = int(shares)
            # 한국 주식인지 미국 주식인지 확인
            if currency.lower() == 'usd':  # 미국 주식
                ticker = backtester.get_us_ticker(stock_name)
            else:  # 한국 주식
                ticker = backtester.get_korean_ticker(stock_name)
                if ticker:
                    ticker += ".KS"  # 한국 주식 티커에 .KS 추가
            if ticker:
                backtester.add_stock(ticker, shares)
            else:
                print(f"{stock_name}에 해당하는 티커를 찾을 수 없습니다. 다시 입력해주세요.")
        except ValueError:
            print("잘못된 입력 형식입니다. 다시 입력해주세요.")

    # 백테스팅 실행
    backtester.run_backtest(start_date, end_date)

if __name__ == "__main__":
    main()