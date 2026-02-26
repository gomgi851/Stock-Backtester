import pandas as pd
import yfinance as yf
import requests
import matplotlib.pyplot as plt
import koreanize_matplotlib
from datetime import datetime
import numpy as np
import os
from matplotlib.path import Path
from matplotlib.patches import PathPatch

class PortfolioBacktester:
    def __init__(self):
        self.portfolio = pd.DataFrame(columns=['ticker', 'shares'])
        self.krx_list = self.load_krx_list()
        print('\033[32m')
        print("-----------------<  v2.8 복리 재투자 엔진 가동 준비 완료  >-----------------")
        print('\33[0m')

    def load_krx_list(self):
        file_path = 'krx_list.csv'
        if os.path.exists(file_path):
            krx_list = pd.read_csv(file_path, dtype={'종목코드': str})
        else:
            url = "http://kind.krx.co.kr/corpgeneral/corpList.do?method=download"
            try:
                krx_list = pd.read_html(url, header=0, encoding='euc-kr')[0]
                krx_list['종목코드'] = krx_list['종목코드'].astype(str).str.zfill(6)
                krx_list.to_csv(file_path, index=False)
            except Exception as e:
                print(f"KRX 리스트 확보 실패: {e}")
                return pd.DataFrame()
        return krx_list

    def get_historical_exchange_rates(self, start_date, end_date):
        try:
            # 과거 환율 시계열 데이터 확보
            ex_data = yf.download('USDKRW=X', start=start_date, end=end_date)['Close']
            if isinstance(ex_data, pd.DataFrame): ex_data = ex_data.iloc[:, 0]
            ex_data.index = ex_data.index.tz_localize(None)
            return ex_data
        except Exception as e:
            print(f"환율 로드 오류: {e}")
            return None

    def calculate_returns(self, data, quantity):
        """
        [v2.8 핵심 변경] 정석 TR(Total Return) 로직
        배당금을 주식 수 재투자로 가정하여 복리 수익률(cumprod)을 계산합니다.
        """
        # 1. 일일 TR 수익률: (오늘 종가 + 오늘 배당) / 어제 종가
        # 배당이 없는 날은 단순 주가 변동률과 같습니다.
        daily_tr_ratio = (data['Close'] + data['Dividends']) / data['Close'].shift(1)
        daily_tr_ratio.iloc[0] = 1.0  # 기준점
        
        # 2. 누적 곱(Cumulative Product)을 통한 TR 지수 생성
        tr_index = daily_tr_ratio.cumprod()
        
        # 3. PR(Price Return)과 TR(Total Return) 가치 환산
        data['PR'] = data['Close'] * quantity
        data['TR'] = (data['Close'].iloc[0] * tr_index) * quantity
        
        return data

    def calculate_mdd(self, series):
        cumulative_max = series.cummax()
        drawdown = (series - cumulative_max) / cumulative_max
        return drawdown.min(), drawdown

    def add_stock(self, ticker, shares):
        new_stock = pd.DataFrame({'ticker': [ticker], 'shares': [shares]})
        self.portfolio = pd.concat([self.portfolio, new_stock], ignore_index=True)

    def run_backtest(self, start_date, end_date):
        historical_ex = self.get_historical_exchange_rates(start_date, end_date)
        if historical_ex is None: return

        all_details = []

        # 1. 개별 종목 수집 및 환산
        for _, stock in self.portfolio.iterrows():
            ticker = stock['ticker']
            shares = stock['shares']
            try:
                stock_data = yf.Ticker(ticker).history(start=start_date, end=end_date)
                stock_data.index = stock_data.index.tz_localize(None)
                
                # 복리 재투자 로직 적용
                data = self.calculate_returns(stock_data, shares)

                pr_val, tr_val = data['PR'], data['TR']

                # 해외 주식인 경우 환율 적용 (시계열 매칭)
                if not ticker.endswith('.KS'):
                    combined = pd.DataFrame({'price': pr_val, 'tr': tr_val}).join(historical_ex.rename('ex_rate'), how='left')
                    combined['ex_rate'] = combined['ex_rate'].ffill().bfill()
                    pr_val = combined['price'] * combined['ex_rate']
                    tr_val = combined['tr'] * combined['ex_rate']

                temp_details = pd.DataFrame({
                    f'{ticker}_Price_KRW': pr_val, 
                    f'{ticker}_TR_KRW': tr_val
                })
                all_details.append(temp_details)
            except Exception as e:
                print(f"{ticker} 오류: {e}")

        # 2. 통합 데이터프레임 생성 및 정제
        detailed_df = pd.concat(all_details, axis=1)
        
        # 벤치마크 데이터 확보
        sp500 = yf.download('^GSPC', start=start_date, end=end_date)['Close'].squeeze()
        kospi = yf.download('^KS11', start=start_date, end=end_date)['Close'].squeeze()
        detailed_df['S&P 500'] = sp500.tz_localize(None) if hasattr(sp500, 'tz_localize') else sp500
        detailed_df['KOSPI'] = kospi.tz_localize(None) if hasattr(kospi, 'tz_localize') else kospi

        # 날짜 정렬 및 주말 제외
        detailed_df.index = detailed_df.index.tz_localize(None)
        detailed_df = detailed_df.resample('D').last().ffill()
        detailed_df = detailed_df[detailed_df.index.weekday < 5] 

        # 총합 계산
        price_cols = [col for col in detailed_df.columns if 'Price_KRW' in col]
        tr_cols = [col for col in detailed_df.columns if 'TR_KRW' in col]
        detailed_df['Total_PR'] = detailed_df[price_cols].sum(axis=1)
        detailed_df['Total_TR'] = detailed_df[tr_cols].sum(axis=1)

        # 3. CSV 저장 (v2.8 버전 명시)
        csv_name = f"portfolio_v2.8_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
        detailed_df.to_csv(csv_name, encoding='utf-8-sig')
        print(f"\n\033[34m[성공] 복리 재투자 반영 데이터가 '{csv_name}'로 저장되었습니다.\033[0m")

        # 4. 시각화
        viz_df = detailed_df[['Total_PR', 'Total_TR', 'S&P 500', 'KOSPI']].dropna()
        normalized = viz_df / viz_df.iloc[0] * 1000

        self.plot_performance(normalized)
        _, drawdown_series = self.calculate_mdd(normalized['Total_PR'])
        self.plot_mdd_gradient(drawdown_series)

    def plot_performance(self, df):
        plt.style.use('dark_background')
        plt.figure(figsize=(14, 7))
        
        # 1. 수익률 계산 (첫 행 대비 마지막 행의 변동률)
        # 1000으로 정규화되었으므로 (값 - 1000) / 10 을 하면 %가 나옵니다.
        final_ret = (df.iloc[-1] - 1000) / 10
        
        # 2. 범례에 수익률 포함하여 그리기
        plt.plot(df.index, df['Total_TR'], 
                 label=f"P/F (복리 TR) : {final_ret['Total_TR']:+.2f}%", 
                 color='dodgerblue', linewidth=2.5)
        
        plt.plot(df.index, df['Total_PR'], 
                 label=f"P/F (단순 PR) : {final_ret['Total_PR']:+.2f}%", 
                 color='cyan', linewidth=1.5, alpha=0.7)
        
        plt.plot(df.index, df['S&P 500'], 
                 label=f"S&P 500 : {final_ret['S&P 500']:+.2f}%", 
                 color='limegreen', alpha=0.4)
        
        plt.plot(df.index, df['KOSPI'], 
                 label=f"KOSPI : {final_ret['KOSPI']:+.2f}%", 
                 color='tomato', alpha=0.4)

        # 스타일링
        plt.title("v2.8 포트폴리오 성과 분석 (배당 복리 재투자 반영)", fontsize=16, pad=20)
        plt.ylabel("정규화 가격 (1000 기준)", fontsize=12)
        plt.legend(loc='upper left', fontsize=11, frameon=True, edgecolor='gray')
        plt.grid(axis='y', alpha=0.1)
        
        # 0% 기준선 표시 (1000점)
        plt.axhline(1000, color='white', linestyle='--', linewidth=0.8, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    def plot_mdd_gradient(self, drawdown):
        fig, ax = plt.subplots(figsize=(12, 5))
        num_points = len(drawdown)
        time_axis = np.arange(num_points)
        dd_pct = drawdown * 100

        ax.plot(time_axis, dd_pct, color='#FF9800', linewidth=1.5, alpha=0.9)

        path_data = np.concatenate([[[0, 0]], np.column_stack([time_axis, dd_pct]), [[time_axis[-1], 0]]])
        path = Path(path_data)
        patch = PathPatch(path, facecolor='none', edgecolor='none')
        ax.add_patch(patch)

        ax.imshow(time_axis.reshape(num_points, 1), cmap=plt.cm.YlOrBr_r, interpolation="bicubic",
                  origin='upper', extent=[0, num_points-1, dd_pct.min()-5, 0], aspect="auto",
                  alpha=0.6, clip_path=patch, clip_on=True)

        ax.axhline(0, color='#FF9800', linewidth=0.8, alpha=0.3)
        current_dd = dd_pct.iloc[-1]
        ax.set_title(f"Maximum Drawdown (Current: {current_dd:.2f}% / Max: {dd_pct.min():.2f}%)",
                     color='#FF9800', fontsize=13, fontweight='bold', pad=15)

        ax.set_facecolor('#101010')
        plt.tight_layout()
        plt.show()

def testCase(end_date=None):
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')

    bt = PortfolioBacktester()
    # 고배당 종목일수록 v2.7과 v2.8의 차이가 극명해집니다.
    bt.add_stock('O', 100) # 리얼티 인컴 100주
    bt.run_backtest('2023-01-01', end_date)

if __name__ == "__main__":
    testCase()