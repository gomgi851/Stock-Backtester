import pandas as pd
import yfinance as yf
import requests
import matplotlib.pyplot as plt
import koreanize_matplotlib
from datetime import datetime, timedelta
import numpy as np
import os


class PortfolioBacktester:
    def __init__(self):
        self.portfolio = pd.DataFrame(columns=['ticker', 'shares'])
        self.krx_list = self.load_krx_list()
        print('\033[32m')
        print("-----------------<  한국거래소 종목코드 데이터 로드 완료  >-----------------")
        self.exchange_rate = self.get_exchange_rate()
        print("-------------------------<  환율 정보 로드 완료  >--------------------------")
        print('\33[0m')

    def load_krx_list(self):
        """KRX 종목코드를 파일에서 불러오거나, 파일이 없으면 웹에서 가져와 파일로 저장합니다."""
        file_path = 'krx_list.csv'
        if os.path.exists(file_path):
            # 파일이 존재하면 파일에서 읽어옴
            krx_list = pd.read_csv(file_path, dtype={'종목코드': str})
        else:
            # 파일이 없으면 웹에서 가져와서 파일로 저장
            url = "http://kind.krx.co.kr/corpgeneral/corpList.do?method=download"
            krx_list = pd.read_html(url, header=0, encoding='euc-kr')[0]
            krx_list['종목코드'] = krx_list['종목코드'].astype(str).str.zfill(6)
            krx_list.to_csv(file_path, index=False)
        return krx_list

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
            if '.ks' in stock_name.lower():
                return stock_name[0:6]
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

    def calculate_returns(self, data, quantity):
        """
        주식 데이터와 수량을 입력하면 TR과 PR을 계산합니다.
        """
        data['PR'] = data['Close']
        data['TR'] = data['Close'] + data['Dividends'].cumsum()
        data['PR'] *= quantity
        data['TR'] *= quantity
        return data

    def calculate_annualized_return(self, initial_value, final_value, start_date, end_date):
        """
        초기 값과 최종 값, 시작 날짜와 종료 날짜를 입력받아 연 평균 수익률을 계산합니다.
        """
        # 백테스팅 기간 계산 (년 단위)
        delta = (end_date - start_date).days / 365.25
        # 연 평균 수익률 계산
        annualized_return = (final_value / initial_value) ** (1 / delta) - 1
        return annualized_return

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
                # yf.Ticker().history()를 사용하여 배당금 정보 포함 데이터 가져오기
                stock_data = yf.Ticker(ticker).history(start=start_date, end=end_date)
                stock_data.index = stock_data.index.tz_localize(None)
                price_data = stock_data['Close']
                dividend_data = stock_data['Dividends']  # 배당금 데이터


                # TR과 PR 계산
                data = self.calculate_returns(stock_data, shares)

                portfolio_prices[ticker] = data['PR']
                portfolio_tr[ticker] = data['TR']
            except Exception as e:
                print(f"{ticker}의 데이터를 가져오는 데 실패했습니다: {e}")
                continue

        # 환율 적용 (미국 주식은 USD를 KRW로 변환)
        for ticker in self.portfolio['ticker']:
            if not ticker.endswith('.KS'):  # 한국 주식이 아닌 경우 (미국 주식)
                portfolio_prices[ticker] *= self.exchange_rate
                portfolio_tr[ticker] *= self.exchange_rate
        portfolio_prices = portfolio_prices.dropna().round(1)
        portfolio_tr = portfolio_tr.dropna().round(1)
        # 포트폴리오 총 가치 계산
        portfolio_prices['Total'] = portfolio_prices.sum(axis=1)
        portfolio_tr['Total'] = portfolio_tr.sum(axis=1)
        portfolio_prices.to_csv('portfolio_prices.csv', index=True, encoding='utf-8-sig')
        portfolio_tr.to_csv('portfolio_tr.csv', index=True, encoding='utf-8-sig')

        # S&P 500과 KOSPI 지수 데이터 가져오기
        try:
            sp500 = yf.Ticker('^GSPC').history(start=start_date, end=end_date)['Close'].squeeze()
            sp500.index = sp500.index.tz_localize(None)
            kospi = yf.Ticker('^KS11').history(start=start_date, end=end_date)['Close'].squeeze()
            kospi.index = kospi.index.tz_localize(None)
        except Exception as e:
            print(f"지수 데이터를 가져오는 데 실패했습니다: {e}")
            return

        # 모든 데이터를 하나의 DataFrame으로 합치기
        comparison = pd.DataFrame({
            'Portfolio_PR': portfolio_prices['Total'],
            'Portfolio_TR': portfolio_tr['Total'],
            'S&P 500': sp500,
            'KOSPI': kospi
        })

        # 시간대 통일 (UTC로 변환)
        # comparison.index = comparison.index.tz_localize(None)  # 시간대 제거
        # comparison = comparison.resample('D').last()  # 일별 마지막 데이터 선택

        # comparison.to_csv('portfolio_comparison.csv', index=True, encoding='utf-8-sig')
        # print("포트폴리오 비교 데이터가 'portfolio_comparison.csv' 파일로 저장되었습니다.")

        # 누락된 데이터 제거
        comparison.replace(0, np.nan, inplace=True)
        comparison = comparison.dropna()

        # 데이터 정규화 (첫 날 값을 1000으로 설정)
        normalized_comparison = comparison / comparison.iloc[0] * 1000

        # MDD 계산
        mdd, drawdown = self.calculate_mdd(normalized_comparison['Portfolio_PR'])

        # Bloomberg Terminal 스타일로 그래프 그리기
        plt.style.use('dark_background')  # 어두운 배경
        plt.figure(figsize=(14, 7))


        # 포트폴리오 (TR)
        plt.plot(normalized_comparison.index, normalized_comparison['Portfolio_TR'], label='Portfolio (TR)', linewidth=2, color='blue')
        # 포트폴리오 (PR)
        plt.plot(normalized_comparison.index, normalized_comparison['Portfolio_PR'], label='Portfolio (PR)', linewidth=2, color='cyan')
        # S&P 500
        plt.plot(normalized_comparison.index, normalized_comparison['S&P 500'], label='S&P 500', linewidth=1.5, color='green', alpha = 0.5)
        # KOSPI
        plt.plot(normalized_comparison.index, normalized_comparison['KOSPI'], label='KOSPI', linewidth=1.5, color='red', alpha = 0.5)



        # 누적 수익률 표시 (그래프 밖에 배치)
        final_values = normalized_comparison.iloc[-1]
        returns = (final_values - 1000) / 1000 * 100

        # 연 평균 수익률 계산
        start_date_dt = pd.to_datetime(start_date)
        end_date_dt = pd.to_datetime(end_date)
        annualized_returns = {
            'Portfolio_TR': self.calculate_annualized_return(1000, final_values['Portfolio_TR'], start_date_dt, end_date_dt),
            'Portfolio_PR': self.calculate_annualized_return(1000, final_values['Portfolio_PR'], start_date_dt, end_date_dt),
            'S&P 500': self.calculate_annualized_return(1000, final_values['S&P 500'], start_date_dt, end_date_dt),
            'KOSPI': self.calculate_annualized_return(1000, final_values['KOSPI'], start_date_dt, end_date_dt)
        }

        plt.text(
            0.14, 0.97,  # x=1.02 (그래프 오른쪽 밖), y=0.95 (상단)
            f"누적 수익률:\n"
            f"{'P/F TR   :'} {returns['Portfolio_TR']:>7.2f}%\n"
            f"{'P/F PR   :'} {returns['Portfolio_PR']:>7.2f}%\n"
            f"{'S&P500:'} {returns['S&P 500']:>7.2f}%\n"
            f"{'KOSPI   :'} {returns['KOSPI']:>7.2f}%\n\n"
            f"연 평균 수익률:\n"
            f"{'P/F TR   :'} {annualized_returns['Portfolio_TR']*100:>7.2f}%\n"
            f"{'P/F PR   :'} {annualized_returns['Portfolio_PR']*100:>7.2f}%\n"
            f"{'S&P500:'} {annualized_returns['S&P 500']*100:>7.2f}%\n"
            f"{'KOSPI   :'} {annualized_returns['KOSPI']*100:>7.2f}%",
            transform=plt.gca().transAxes,  # 그래프 축 기준으로 위치 설정
            fontsize=12,
            bbox=dict(facecolor='black', alpha=0.8, edgecolor='white'),  # 검은색 배경, 하얀색 테두리
            verticalalignment='top'  # 텍스트를 상단 정렬
        )

        # 제목과 레이블 설정
        plt.title('Portfolio vs S&P 500 vs KOSPI (환율 고려, 휴장일 제외)', fontsize=16, fontweight='light', pad=20, color='white')
        plt.xlabel('Date', fontsize=12, color='white')
        plt.ylabel('Normalized Price', fontsize=12, color='white')

        # 범례 설정
        plt.legend(fontsize=12, loc='upper left', facecolor='black', edgecolor='white')

        # 그리드 설정
        plt.grid(axis='y', linestyle='--', alpha=0.5, color='gray')

        # 레이아웃 조정
        plt.rc('font', family = "NanumGothic")
        plt.tight_layout()
        #plt.savefig('portfolio_performance_bloomberg.png', dpi=600, bbox_inches='tight', facecolor='black')
        plt.show()

        # 두 번째 플롯: MDD 그래프
        plt.figure(figsize=(14, 7))
        plt.style.use('dark_background')  # 어두운 배경

        # Drawdown 그래프
        plt.plot(drawdown.index, drawdown * 100, label='Drawdown', linewidth=2, color='red')
        plt.fill_between(drawdown.index, drawdown * 100, 0, color='red', alpha=0.3)

        # MDD 값 표시
        plt.axhline(y=mdd * 100, color='white', linestyle='--', linewidth=1, label=f'MDD: {mdd*100:.2f}%')

        # 제목과 레이블 설정
        plt.title('Maximum Drawdown (MDD)', fontsize=16, fontweight='light', pad=20, color='white')
        plt.xlabel('Date', fontsize=12, color='white')
        plt.ylabel('Drawdown (%)', fontsize=12, color='white')

        # 범례 설정
        plt.legend(fontsize=12, loc='upper left', facecolor='black', edgecolor='white')

        # 그리드 설정
        plt.grid(axis='y', linestyle='--', alpha=0.5, color='gray')

        # 레이아웃 조정
        plt.rc('font', family = "NanumGothic")
        plt.tight_layout()
        #plt.savefig('portfolio_mdd.png', dpi=600, bbox_inches='tight', facecolor='black')
        plt.show()

    def calculate_mdd(self, pr_series):
        """
        PR(Price Return) 시리즈를 입력받아 MDD(Maximum Drawdown)를 계산합니다.
        """
        # 누적 최고점 계산
        cumulative_max = pr_series.cummax()
        # Drawdown 계산 (현재 값과 최고점 대비 하락률)
        drawdown = (pr_series - cumulative_max) / cumulative_max
        # MDD는 Drawdown 중 최소값 (가장 큰 하락)
        mdd = drawdown.min()
        return mdd, drawdown

def main():
    backtester = PortfolioBacktester()
    #오늘 날짜
    default_start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    # 1년 전 날짜
    default_end_date = datetime.now().strftime('%Y-%m-%d')
    # 백테스팅 기간 입력
    start_date =      input("백테스팅 시작일을 입력하세요 (YYYY-MM-DD)                 (미입력시 1년 전)↵: ") or default_start_date
    end_date =        input("백테스팅 종료일을 입력하세요 (YYYY-MM-DD)                 (미입력시   오늘)↵: ") or default_end_date

    # 주식 정보 입력
    while True:
        stock_input = input("주식 이름과 수량을 입력하세요 (예: 메리츠금융지주 20 KRW), 종료하려면 Enter↵: ")
        if stock_input == "":
            break
        try:
            stock_name, shares, currency = stock_input.rsplit(' ')
            shares = int(shares)
            # 한국 주식인지 미국 주식인지 확인
            if currency.lower() == 'usd':  # 미국 주식
                ticker = backtester.get_us_ticker(stock_name)
            else:  # 한국 주식
                ticker = backtester.get_korean_ticker(stock_name.upper())
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

def testCase1():
    backtester = PortfolioBacktester()
    #backtester.add_stock(backtester.get_korean_ticker('메리츠금융지주')+'.KS', 20)
    backtester.add_stock('coin', 20)
    backtester.run_backtest('2023-10-01', '2024-12-01')

def testCase2():
    backtester = PortfolioBacktester()
    #backtester.add_stock(backtester.get_korean_ticker('메리츠금융지주')+'.KS', 20)
    backtester.add_stock('cony', 20)
    backtester.run_backtest('2023-10-01', '2024-12-01')

if __name__ == "__main__":
    #main()
    testCase1()
    testCase2()