import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import calendar
import koreanize_matplotlib
import yfinance as yf
import logging
import requests
from bs4 import BeautifulSoup
import warnings


logging.basicConfig(
    level=logging.INFO,  # INFO 레벨 이상의 로그를 출력
    format='%(asctime)s - %(levelname)s - %(message)s',  # 로그 포맷 설정
    handlers=[logging.StreamHandler()]  # 콘솔에 로그 출력
)

class PortfolioAnalyzer:
    def __init__(self, exchange_rate_usd):
        """
        PortfolioAnalyzer 초기화

        Parameters:
        - exchange_rate_usd: USD to KRW 환율
        """
        self.portfolio = pd.DataFrame()
        self.dividend_schedule = pd.DataFrame()
        self.exchange_rate_usd = exchange_rate_usd
        logging.info(f"PortfolioAnalyzer initialized with exchange rate: {exchange_rate_usd}")

    def update_exchange_rate(self):
        """
        환율 정보를 자동으로 업데이트합니다.
        """
        try:
            url = "https://api.exchangerate-api.com/v4/latest/USD"
            response = requests.get(url)
            response.raise_for_status()  # HTTP 오류 확인
            data = response.json()
            self.exchange_rate_usd = data['rates']['KRW']
            logging.info(f"환율 정보 업데이트 완료: {self.exchange_rate_usd}원/USD")
        except Exception as e:
            logging.error(f"환율 정보를 가져오는 데 실패했습니다: {e}. 기본 환율을 사용합니다.")
            self.exchange_rate_usd = 1300  # 기본 환율

    def get_korean_ticker(self, stock_name):
        """
        한국 주식의 종목명을 입력하면 티커를 반환합니다.
        """
        url = "http://kind.krx.co.kr/corpgeneral/corpList.do?method=download"
        try:
            krx_list = pd.read_html(url, header=0, encoding='euc-kr')[0]  # 인코딩 지정
            krx_list['종목코드'] = krx_list['종목코드'].astype(str).str.zfill(6)
            match = krx_list[krx_list['회사명'] == stock_name]
            if not match.empty:
                return match['종목코드'].values[0] + ".KS"
            else:
                logging.warning(f"한국 주식 {stock_name}의 티커를 찾을 수 없습니다.")
                return None
        except Exception as e:
            logging.error(f"한국 주식 티커를 가져오는 데 실패했습니다: {e}")
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
                logging.warning(f"미국 주식 {stock_name}의 티커를 찾을 수 없습니다.")
                return None
        except Exception as e:
            logging.error(f"미국 주식 티커 검색 중 오류 발생: {e}")
            return None

    def add_stock(self, stock_name, shares, sector,
                  dividend_per_share=0, dividend_months=[], currency='KRW'):
        """
        포트폴리오에 주식 추가

        Parameters:
        - stock_name: 종목명 (한국 주식 또는 미국 주식)
        - shares: 보유 주식 수
        - sector: 섹터
        - dividend_per_share: 주당 배당금 (단일 값 또는 월별 리스트)
        - dividend_months: 배당금 지급 월
        - currency: 통화 ('KRW' 또는 'USD')
        """
        # 한국 주식인 경우 티커 찾기
        if currency == 'KRW':
            ticker = self.get_korean_ticker(stock_name)
            if not ticker:
                return
        # 미국 주식인 경우 티커 찾기
        elif currency == 'USD':
            ticker = self.get_us_ticker(stock_name)
            if not ticker:
                return
        else:
            logging.error(f"지원하지 않는 통화입니다: {currency}")
            return

        # 현재 주가 가져오기
        try:
            stock_data = yf.Ticker(ticker)
            current_price = stock_data.history(period="1d")['Close'].iloc[-1]
        except Exception as e:
            logging.error(f"{stock_name}의 주가 정보를 가져오는 데 실패했습니다: {e}")
            return

        # USD인 경우 원화로 환산
        if currency == 'USD':
            current_price = current_price * self.exchange_rate_usd
            if isinstance(dividend_per_share, list):
                dividend_per_share = [dps * self.exchange_rate_usd for dps in dividend_per_share]
            else:
                dividend_per_share = dividend_per_share * self.exchange_rate_usd

        stock_value = shares * current_price

        new_stock = pd.DataFrame({
            'ticker': [ticker],
            'stock_name': [stock_name],
            'shares': [shares],
            'current_price': [current_price],
            'value': [stock_value],
            'sector': [sector],
            'dividend_per_share': [dividend_per_share],
            'dividend_months': [dividend_months],
            'currency': [currency]
        })

        self.portfolio = pd.concat([self.portfolio, new_stock], ignore_index=True)
        self._update_dividend_schedule()
        logging.info(f"{stock_name} 주식이 포트폴리오에 추가되었습니다.")

    def _update_dividend_schedule(self):
        """배당금 지급 일정 업데이트"""
        current_year = datetime.now().year
        dividend_data = []

        for _, stock in self.portfolio.iterrows():
            if stock['dividend_months']:
                for i, month in enumerate(stock['dividend_months']):
                    last_day = calendar.monthrange(current_year, month)[1]
                    dividend_date = datetime(current_year, month, last_day)

                    # dividend_per_share가 리스트인 경우 월별 배당금을 다르게 설정
                    if isinstance(stock['dividend_per_share'], list):
                        quarterly_dividend = stock['dividend_per_share'][i] * stock['shares']
                    else:
                        quarterly_dividend = (stock['dividend_per_share'] * stock['shares']) / len(stock['dividend_months'])

                    dividend_data.append({
                        'ticker': stock['ticker'],
                        'date': dividend_date,
                        'amount': quarterly_dividend,
                        'currency': stock['currency']
                    })

        self.dividend_schedule = pd.DataFrame(dividend_data)
        logging.info("배당금 지급 일정이 업데이트되었습니다.")

    def get_dividend_calendar(self):
        """배당금 월별 막대 그래프"""
        if self.dividend_schedule.empty:
            logging.warning("배당금 지급 일정이 없습니다.")
            return

        # 월별 배당금 합계 계산
        monthly_dividends = self.dividend_schedule.groupby(
            self.dividend_schedule['date'].dt.month
        )['amount'].sum()

        # 모든 월에 대한 데이터 생성
        all_months = pd.Series(0, index=range(1, 13))
        monthly_dividends = monthly_dividends.combine_first(all_months)
        monthly_dividends = monthly_dividends.sort_index()

        # 월 이름으로 변환
        month_names = [calendar.month_abbr[m] for m in range(1, 13)]

        # 그래프 생성
        plt.figure(figsize=(12, 6))
        bars = plt.bar(month_names, monthly_dividends, color='skyblue')

        # 제목과 레이블 설정
        dividend_yield = self.calculate_dividend_yield()
        plt.title(f'월별 배당금 (시가 배당률 : {dividend_yield:.2f}%)',
                  fontsize=16, fontweight='light', pad=20) # title 변경
        plt.xlabel('월')
        plt.ylabel('배당금 (원)')

        # 막대 위에 값 표시
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:,.0f}원',
                        ha='center', va='bottom')

        # 연간 총 배당금 계산
        yearly_dividends = self.dividend_schedule.groupby('currency')['amount'].sum()
        total_dividends = yearly_dividends.sum()

        # 연간 예상 배당금 정보를 그래프 안에 텍스트로 추가
        dividend_text = "연간 예상 배당금:\n"
        for currency, amount in yearly_dividends.items():
            if currency == 'USD':
                usd_amount = amount / self.exchange_rate_usd
                dividend_text += f"{currency}: ${usd_amount:,.2f} (₩{amount:,.0f})\n"
            else:
                dividend_text += f"{currency}: ₩{amount:,.0f}\n"
        dividend_text += f"총액: ₩{total_dividends:,.0f}"

        plt.text(0.95, 0.95, dividend_text,
                transform=plt.gca().transAxes,
                fontsize=10,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig('dividend_calendar.png', dpi=300, bbox_inches='tight')
        plt.show()

    def calculate_dividend_yield(self):
        """배당률 계산 (총 배당금 / 포트폴리오 총액)"""
        total_dividend = self.dividend_schedule['amount'].sum()
        total_portfolio_value = self.portfolio['value'].sum()

        if total_portfolio_value == 0:
            logging.warning("포트폴리오 총액이 0입니다. 배당률을 계산할 수 없습니다.")
            return 0  # 또는 다른 적절한 값 반환

        dividend_yield = (total_dividend / total_portfolio_value) * 100
        logging.info(f"배당률 계산 완료: {dividend_yield:.2f}%")
        return dividend_yield

    def track_portfolio_performance(self):
        """
        포트폴리오 성능 추적 및 S&P 500, KOSPI와 비교 (환율 고려, 휴장일 제외)
        - 포트폴리오는 add_stock으로 추가된 종목과 수량을 사용
        - 날짜는 오늘부터 1년 전 데이터로 설정
        """
        # 날짜 설정 (오늘부터 1년 전)
        end_date = pd.Timestamp.today().strftime('%Y-%m-%d')
        start_date = (pd.Timestamp.today() - pd.DateOffset(years=1)).strftime('%Y-%m-%d')

        # 포트폴리오 구성 (add_stock으로 추가된 종목과 수량 사용)
        portfolio = {}
        for _, stock in self.portfolio.iterrows():
            ticker = stock['ticker']
            shares = stock['shares']
            portfolio[ticker] = shares

        # 오늘의 환율 적용
        exchange_rate = self.exchange_rate_usd

        # 포트폴리오 가격 데이터 가져오기
        portfolio_prices = pd.DataFrame()
        for ticker, shares in portfolio.items():
            try:
                data = yf.download(ticker, start=start_date, end=end_date)['Close']
                portfolio_prices[ticker] = data
            except Exception as e:
                logging.error(f"{ticker}의 데이터를 가져오는 데 실패했습니다: {e}")
                continue

        # 환율 적용 (미국 주식은 USD를 KRW로 변환)
        for ticker in portfolio:
            if not ticker.endswith('.KS'):  # 한국 주식이 아닌 경우 (미국 주식)
                portfolio_prices[ticker] *= exchange_rate

        # 포트폴리오 가격 데이터를 주식 수로 곱하여 포트폴리오 가치 계산
        portfolio_values = portfolio_prices.mul(portfolio)

        # 포트폴리오 총 가치 계산
        portfolio_values['Total'] = portfolio_values.sum(axis=1)

        # S&P 500과 KOSPI 지수 데이터 가져오기
        try:
            sp500 = yf.download('^GSPC', start=start_date, end=end_date)['Close'].squeeze()
            kospi = yf.download('^KS11', start=start_date, end=end_date)['Close'].squeeze()
        except Exception as e:
            logging.error(f"지수 데이터를 가져오는 데 실패했습니다: {e}")
            return

        # 모든 데이터를 하나의 DataFrame으로 합치기
        comparison = pd.DataFrame({
            'Portfolio': portfolio_values['Total'],
            'S&P 500': sp500,
            'KOSPI': kospi
        })

        # 한국과 미국의 휴장일을 고려하여 데이터 정렬
        # 두 시장 모두 열린 날짜만 선택
        comparison = comparison.dropna()  # 누락된 데이터 제거

        # 데이터 정규화 (첫 날 값을 1000으로 설정)
        normalized_comparison = comparison / comparison.iloc[0] * 1000

        # 그래프 그리기
        plt.figure(figsize=(14, 7))

        # 포트폴리오 라인
        plt.plot(normalized_comparison['Portfolio'], label='Portfolio', linewidth=3, color='skyblue', linestyle='--')

        # S&P 500과 KOSPI 라인
        plt.plot(normalized_comparison['S&P 500'], label='S&P 500', linewidth=2, color='darkgray', linestyle='--')
        plt.plot(normalized_comparison['KOSPI'], label='KOSPI', linewidth=2, color='lightgrey', linestyle='--')

        # 제목과 레이블 설정
        plt.title('Portfolio vs S&P 500 vs KOSPI (환율 고려, 휴장일 제외)',
                  fontsize=16, fontweight='light', pad=20)
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

        # 정규화된 데이터 출력
        print(normalized_comparison.tail())


    def analyze_portfolio(self):
        """포트폴리오 비중 분석 및 시각화"""
        # 전체 금액 계산
        total_value = self.portfolio['value'].sum()

        # 섹터별 비중 계산
        sector_weights = self.portfolio.groupby('sector')['value'].sum()
        sector_percentages = (sector_weights / total_value * 100).round(1)

        # 통화별 비중 계산
        currency_weights = self.portfolio.groupby('currency')['value'].sum()
        currency_percentages = (currency_weights / total_value * 100).round(1)

        # 섹터별 색상 매핑 생성
        unique_sectors = self.portfolio['sector'].unique()
        colors = plt.cm.Pastel2(np.linspace(0, 1, len(unique_sectors)))
        sector_colors = dict(zip(unique_sectors, colors))

        # 그래프 생성
        fig = plt.figure(figsize=(18, 8))

        # 서브플롯 위치 조정
        grid = plt.GridSpec(2, 3, height_ratios=[1, 4])

        # 총액 텍스트 표시 (상단 중앙)
        text_ax = plt.subplot(grid[0, :])
        text_ax.axis('off')
        text_ax.text(0.5, 0.5,
                    f'포트폴리오 총액: ₩{total_value:,.0f}',
                    ha='center', va='center',
                    fontsize=16, fontweight='light',
                    bbox=dict(facecolor='white', edgecolor='gray', linewidth = 5, pad=10, linestyle = '--'))

        # 섹터별 파이 차트
        ax1 = plt.subplot(grid[1, 0])
        wedges, texts, autotexts = ax1.pie(sector_percentages,
                                        labels=sector_percentages.index,
                                        colors=[sector_colors[sector] for sector in sector_percentages.index],
                                        autopct='%1.1f%%',
                                        startangle=90)
        ax1.set_title('섹터별 포트폴리오 비중', bbox=dict(facecolor='lightgray', edgecolor='w'))

        # 종목별 막대 그래프
        ax2 = plt.subplot(grid[1, 1])
        sorted_portfolio = self.portfolio.sort_values('value', ascending=False)
        bars = ax2.bar(sorted_portfolio['stock_name'],
                      sorted_portfolio['value'],
                      color=[sector_colors[sector] for sector in sorted_portfolio['sector']],)
        ax2.set_title('종목별 포트폴리오 금액', bbox=dict(facecolor='lightgray', edgecolor='w'))
        ax2.set_xlabel('종목')
        ax2.set_ylabel('금액 (원)')

        # 막대 위에 비중과 통화 표시
        for i, bar in enumerate(bars):
            value = bar.get_height()
            percentage = (value / total_value * 100).round(1)
            currency = self.portfolio.iloc[i]['currency']
            ax2.text(bar.get_x() + bar.get_width()/2., value,
                    f'{percentage}%\n({currency})',
                    ha='center', va='bottom', fontsize=8)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        ax2.ticklabel_format(axis='y', style='plain', useOffset=False) # y축 값 형식 변경
        y_max = ax2.get_ylim()[1]
        ax2.set_ylim(0, y_max * 1.10)

        # 통화별 파이 차트
        ax3 = plt.subplot(grid[1, 2])
        wedges, texts, autotexts = ax3.pie(currency_percentages,
                                        labels=currency_percentages.index,
                                        colors=['lightblue', 'lightgreen'],
                                        autopct='%1.1f%%',
                                        startangle=90)
        ax3.set_title('통화별 포트폴리오 비중', bbox=dict(facecolor='lightgray', edgecolor='w'))

        # 섹터별 범례 추가
        legend_elements = [plt.Rectangle((0,0), 1, 1, facecolor=color, label=sector)
                          for sector, color in sector_colors.items()]
        ax2.legend(handles=legend_elements, title='섹터',
                  loc='upper left', bbox_to_anchor=(1, 1))

        plt.tight_layout()
        plt.savefig('portfolio_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

# 사용 예시
if __name__ == "__main__":
    # 환율 설정 (예: 1달러 = 1300원)
    analyzer = PortfolioAnalyzer(exchange_rate_usd=1300)
    analyzer.update_exchange_rate()  # 환율 정보 업데이트
    # 주식 추가 (종목명으로 입력)
    analyzer.add_stock('메리츠금융지주', 132, '금융', dividend_per_share=2360, dividend_months=[4], currency='KRW')
    analyzer.add_stock('하나금융지주', 175, '금융', dividend_per_share=2400, dividend_months=[2, 5, 8, 11], currency='KRW')
    analyzer.add_stock('한미반도체', 25, '전자', dividend_per_share=420, dividend_months=[3], currency='KRW')
    analyzer.add_stock('HOOD', 23, '금융', dividend_per_share=0.25, dividend_months=[2, 5, 8, 11], currency='USD')
    analyzer.add_stock('CROX', 7, '소비재', dividend_per_share=0.25, dividend_months=[2, 5, 8, 11], currency='USD')

    # 포트폴리오 분석
    analyzer.analyze_portfolio()

    # 배당금 캘린더 표시
    analyzer.get_dividend_calendar()

    # 포트폴리오 성능 추적 및 대표 지수 비교
    analyzer.track_portfolio_performance()
