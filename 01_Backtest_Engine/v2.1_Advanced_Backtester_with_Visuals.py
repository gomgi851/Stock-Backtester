import pandas as pd
import yfinance as yf
import requests
import matplotlib.pyplot as plt
import koreanize_matplotlib
from datetime import datetime
import numpy as np
from scipy.interpolate import make_interp_spline
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import re

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
                # 비슷한 이름의 주식을 추천 (2.3)
                similar_stocks = self.krx_list[self.krx_list['회사명'].str.contains(stock_name, case=False)]
                if not similar_stocks.empty:
                    print(f"'{stock_name}'에 해당하는 주식을 찾을 수 없습니다. 비슷한 주식: {similar_stocks['회사명'].values}")
                else:
                    print(f"'{stock_name}'에 해당하는 주식을 찾을 수 없습니다.")
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
                print(f"미국 주식 '{stock_name}'의 티커를 찾을 수 없습니다.")
                return None
        except Exception as e:
            print(f"미국 주식 티커 검색 중 오류 발생: {e}")
            return None

    def add_stock(self, ticker, shares):
        """포트폴리오에 주식을 추가합니다."""
        new_stock = pd.DataFrame({'ticker': [ticker], 'shares': [shares]})
        self.portfolio = pd.concat([self.portfolio, new_stock], ignore_index=True)

    def run_backtest(self, start_date, end_date):
        """백테스팅을 실행하고 결과를 시각화합니다."""
        if self.exchange_rate is None:
            print("환율 정보가 없어 백테스팅을 진행할 수 없습니다. 직접 환율을 입력하세요.")
            self.exchange_rate = float(input("USD to KRW 환율을 입력하세요: "))  # 5.2

        # 포트폴리오 가격 데이터 가져오기
        portfolio_prices = pd.DataFrame()
        for _, stock in self.portfolio.iterrows():
            ticker = stock['ticker']
            shares = stock['shares']
            try:
                print(f"{ticker} 데이터를 가져오는 중...")  # 4.1
                data = yf.download(ticker, start=start_date, end=end_date)['Close']
                portfolio_prices[ticker] = data
            except Exception as e:
                print(f"{ticker}의 데이터를 가져오는 데 실패했습니다: {e}")
                continue

        # 환율 적용 (미국 주식은 USD를 KRW로 변환)
        for ticker in self.portfolio['ticker']:
            if not ticker.endswith('.KS'):  # 한국 주식이 아닌 경우 (미국 주식)
                portfolio_prices[ticker] *= self.exchange_rate

        # 포트폴리오 가격 데이터를 주식 수로 곱하여 포트폴리오 가치 계산
        portfolio_values = portfolio_prices.mul(self.portfolio.set_index('ticker')['shares'])

        # 포트폴리오 총 가치 계산
        portfolio_values['Total'] = portfolio_values.sum(axis=1)

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
            'S&P 500': sp500,
            'KOSPI': kospi
        })

        # 누락된 데이터 제거
        comparison = comparison.dropna()

        # 데이터 정규화 (첫 날 값을 1000으로 설정)
        normalized_comparison = comparison / comparison.iloc[0] * 1000

        # 그래프 그리기
        plt.figure(figsize=(14, 7))

        # 날짜를 숫자로 변환 (x축)
        x = np.arange(len(normalized_comparison.index))  # 날짜를 숫자로 변환
        x_new = np.linspace(x.min(), x.max(), 300)  # 300개의 점으로 보간

        # 포트폴리오
        spl_portfolio = make_interp_spline(x, normalized_comparison['Portfolio'], k=3)
        y_portfolio_smooth = spl_portfolio(x_new)
        plt.plot(normalized_comparison.index, normalized_comparison['Portfolio'], label='Portfolio', linewidth=3, color='skyblue', linestyle='-', alpha=0.3)  # 원본 데이터 (투명하게)
        plt.plot(pd.to_datetime(normalized_comparison.index[x_new.astype(int)]), y_portfolio_smooth, label='Portfolio (Smoothed)', linewidth=3, color='skyblue')  # 보간된 데이터

        # S&P 500
        spl_sp500 = make_interp_spline(x, normalized_comparison['S&P 500'], k=3)
        y_sp500_smooth = spl_sp500(x_new)
        plt.plot(normalized_comparison.index, normalized_comparison['S&P 500'], label='S&P 500', linewidth=2, color='red', linestyle='--', alpha=0.3)  # 원본 데이터 (투명하게)
        plt.plot(pd.to_datetime(normalized_comparison.index[x_new.astype(int)]), y_sp500_smooth, label='S&P 500 (Smoothed)', linewidth=2, color='red')  # 보간된 데이터

        # KOSPI
        spl_kospi = make_interp_spline(x, normalized_comparison['KOSPI'], k=3)
        y_kospi_smooth = spl_kospi(x_new)
        plt.plot(normalized_comparison.index, normalized_comparison['KOSPI'], label='KOSPI', linewidth=2, color='green', linestyle='-.', alpha=0.3)  # 원본 데이터 (투명하게)
        plt.plot(pd.to_datetime(normalized_comparison.index[x_new.astype(int)]), y_kospi_smooth, label='KOSPI (Smoothed)', linewidth=2, color='green')  # 보간된 데이터

        # 누적 수익률 표시 (1.4)
        final_values = normalized_comparison.iloc[-1]
        returns = (final_values - 1000) / 1000 * 100
        plt.text(1.02, 0.95,  # x=1.02 (그래프 오른쪽 밖), y=0.95 (상단)
         f"누적 수익률:\n포트폴리오: {returns['Portfolio']:.2f}%\nS&P 500: {returns['S&P 500']:.2f}%\nKOSPI: {returns['KOSPI']:.2f}%",
         transform=plt.gca().transAxes,  # 그래프 축 기준으로 위치 설정
         fontsize=12,
         bbox=dict(facecolor='white', alpha=0.8),
         verticalalignment='top')  # 텍스트를 상단 정렬

        # 제목과 레이블 설정
        plt.title('Portfolio vs S&P 500 vs KOSPI (환율 고려, 휴장일 제외)', fontsize=16, fontweight='light', pad=20)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Normalized Price', fontsize=12)

        # 범례 설정
        plt.legend(fontsize=12, loc='upper left')  # 범례를 오른쪽 상단으로 이동

        # 그리드 설정
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # 레이아웃 조정
        plt.tight_layout()

        # 그래프 저장 옵션 제공 (4.2)
        save_option = input("그래프를 저장하시겠습니까? (Y/N): ").strip().lower()
        if save_option == 'y':
            filename = input("저장할 파일 이름을 입력하세요 (예: portfolio.png): ").strip()
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"그래프가 '{filename}'로 저장되었습니다.")

        plt.show()

def main():
    backtester = PortfolioBacktester()

    # 백테스팅 기간 입력
    while True:
        start_date = input("백테스팅 시작일을 입력하세요 (YYYY-MM-DD): ").strip()
        end_date = input("백테스팅 종료일을 입력하세요 (YYYY-MM-DD): ").strip()
        try:
            datetime.strptime(start_date, "%Y-%m-%d")
            datetime.strptime(end_date, "%Y-%m-%d")
            break
        except ValueError:
            print("잘못된 날짜 형식입니다. 다시 입력해주세요.")

    # 주식 정보 입력
    while True:
        stock_input = input("주식 이름과 수량을 입력하세요 (예: 메리츠금융지주 20 KRW), 종료하려면 -1을 입력하세요: ").strip()
        if stock_input == "-1":
            break
        try:
            # 정규표현식으로 입력 형식 검사 (2.3)
            match = re.match(r"(\D+)\s+(\d+)\s+(\w+)", stock_input)
            if not match:
                print("잘못된 입력 형식입니다. 다시 입력해주세요.")
                continue
            stock_name, shares, currency = match.groups()
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