
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
                krx_list.to_csv(file_path, index=False, encoding='utf-8-sig')
            except Exception as e:
                print(f"KRX 리스트 확보 실패: {e}")
                return pd.DataFrame()
        return krx_list

    def get_historical_exchange_rates(self, start_date, end_date):
        try:
            ex_data = yf.download('USDKRW=X', start=start_date, end=end_date)['Close']
            if isinstance(ex_data, pd.DataFrame): ex_data = ex_data.iloc[:, 0]
            ex_data.index = ex_data.index.tz_localize(None)
            return ex_data
        except Exception as e:
            print(f"환율 로드 오류: {e}")
            return None

    def calculate_returns(self, data, quantity):
        """
        [v2.8 핵심] 배당금을 주식 수 재투자로 가정하여 복리 수익률(cumprod) 계산
        """
        # 일일 TR 수익률: (오늘 종가 + 오늘 배당) / 어제 종가
        daily_tr_ratio = (data['Close'] + data['Dividends']) / data['Close'].shift(1)
        daily_tr_ratio.iloc[0] = 1.0 
        
        # 누적 곱을 통한 TR 지수 생성
        tr_index = daily_tr_ratio.cumprod()
        
        data['PR'] = data['Close'] * quantity
        data['TR'] = (data['Close'].iloc[0] * tr_index) * quantity
        
        return data

    def calculate_mdd(self, series):
        cumulative_max = series.cummax()
        drawdown = (series - cumulative_max) / cumulative_max
        return drawdown.min(), drawdown

    def calculate_summary(self, df):
        days = (df.index[-1] - df.index[0]).days
        years = days / 365.25 if days > 0 else 1
        
        total_return_tr = (df['Total_TR'].iloc[-1] / df['Total_TR'].iloc[0]) - 1
        cagr = ((df['Total_TR'].iloc[-1] / df['Total_TR'].iloc[0]) ** (1/years) - 1) * 100
        mdd_val, _ = self.calculate_mdd(df['Total_TR'])

        # 디자인: 사용자님의 취향인 ║와 ═를 활용한 닫힌 박스
        # 팁: 아래 공백(Space) 개수는 터미널 폰트에 따라 1~2개 차이가 날 수 있습니다.
        print("\n  " + "╔" + "═" * 52 + "╗")
        print(f"  ║  {f'[ v2.8 Portfolio Backtest Report ]':^48}  ║")
        print("  " + "╠" + "═" * 52 + "╣")
        print(f"  ║  • 테스트 기간   : {df.index[0].date()} ~ {df.index[-1].date()}         ║")
        print(f"  ║  • 누적 수익률(TR): {total_return_tr*100:>12.2f}%                  ║")
        print(f"  ║  • 연평균 수익률(CAGR): {cagr:>10.2f}%                ║")
        # MDD 컬러 적용 (컬러 코드는 길이에 영향을 주지 않으므로 기존 칸수 유지)
        mdd_str = f"{mdd_val*100:>12.2f}%"
        print(f"  ║  • 최대 낙폭(MDD)  : \033[31m{mdd_str}\033[0m                 ║")
        print("  " + "╚" + "═" * 52 + "╝" + "\n")

    def add_stock(self, ticker, shares):
        new_stock = pd.DataFrame({'ticker': [ticker], 'shares': [shares]})
        self.portfolio = pd.concat([self.portfolio, new_stock], ignore_index=True)

    def run_backtest(self, start_date, end_date, total_investment=100_000_000): # 기본 1억 설정
        # 1. 환율 데이터 먼저 가져오기 (절대 날리면 안 됨!)
        historical_ex = self.get_historical_exchange_rates(start_date, end_date)
        if historical_ex is None: return

        # 2. 입력된 값(shares 컬럼에 저장된 값)을 '비중'으로 해석
        total_input_value = self.portfolio['shares'].sum() 
        all_details = []

        for _, stock in self.portfolio.iterrows():
            ticker = stock['ticker']
            # 입력값이 40, 40, 20이면 0.4, 0.4, 0.2로 변환
            weight = stock['shares'] / total_input_value
            allocated_krw = total_investment * weight
            
            try:
                stock_obj = yf.Ticker(ticker)
                stock_data = stock_obj.history(start=start_date, end=end_date)
                stock_data.index = stock_data.index.tz_localize(None)
                
                # [핵심] 첫날 가격(원화 환산 필요)으로 살 수 있는 주식 수 계산
                first_price = stock_data['Close'].iloc[0]
                
                # 미국 주식이라면 첫날 환율을 적용해 원화 기준 구매 가능 수량 산출
                if not ticker.endswith('.KS'):
                    first_ex_rate = historical_ex.loc[stock_data.index[0]]
                    # 원화예산 / (달러가격 * 환율) = 구매 수량
                    shares_to_buy = allocated_krw / (first_price * first_ex_rate)
                else:
                    # 한국 주식은 그냥 원화예산 / 원화가격
                    shares_to_buy = allocated_krw / first_price

                # 이후 수익률 계산(calculate_returns)은 역산된 shares_to_buy로 진행
                data = self.calculate_returns(stock_data, shares_to_buy)
                
                # (기존 환율 처리 로직 그대로 유지...)
                pr_val, tr_val = data['PR'], data['TR']
                if not ticker.endswith('.KS'):
                    combined = pd.DataFrame({'price': pr_val, 'tr': tr_val}).join(historical_ex.rename('ex_rate'), how='left')
                    combined['ex_rate'] = combined['ex_rate'].ffill().bfill()
                    pr_val = combined['price'] * combined['ex_rate']
                    tr_val = combined['tr'] * combined['ex_rate']

                all_details.append(pd.DataFrame({f'{ticker}_Price_KRW': pr_val, f'{ticker}_TR_KRW': tr_val}))
                
            except Exception as e:
                print(f"{ticker} 계산 오류: {e}")

        # ... 이하 데이터 결합 및 시각화 로직 동일

        detailed_df = pd.concat(all_details, axis=1)
        
        # 벤치마크 데이터
        sp500 = yf.download('^GSPC', start=start_date, end=end_date)['Close'].squeeze()
        kospi = yf.download('^KS11', start=start_date, end=end_date)['Close'].squeeze()
        detailed_df['S&P 500'] = sp500.tz_localize(None) if hasattr(sp500, 'tz_localize') else sp500
        detailed_df['KOSPI'] = kospi.tz_localize(None) if hasattr(kospi, 'tz_localize') else kospi

        detailed_df.index = detailed_df.index.tz_localize(None)
        detailed_df = detailed_df.resample('D').last().ffill()
        detailed_df = detailed_df[detailed_df.index.weekday < 5] 

        price_cols = [col for col in detailed_df.columns if 'Price_KRW' in col]
        tr_cols = [col for col in detailed_df.columns if 'TR_KRW' in col]
        detailed_df['Total_PR'] = detailed_df[price_cols].sum(axis=1)
        detailed_df['Total_TR'] = detailed_df[tr_cols].sum(axis=1)

        # CSV 저장
        csv_name = f"portfolio_v2.8_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
        detailed_df.to_csv(csv_name, encoding='utf-8-sig')
        print(f"\n\033[34m[성공] 데이터가 '{csv_name}'로 저장되었습니다.\033[0m")

        # 시각화 데이터 준비
        viz_df = detailed_df[['Total_PR', 'Total_TR', 'S&P 500', 'KOSPI']].dropna()
        normalized = viz_df / viz_df.iloc[0] * 1000

        # 성과 요약 출력
        self.calculate_summary(detailed_df)

        self.plot_performance(normalized)
        _, drawdown_series = self.calculate_mdd(normalized['Total_TR'])
        self.plot_mdd_gradient(drawdown_series)

    def plot_performance(self, df):
        plt.style.use('dark_background')
        plt.figure(figsize=(14, 7))
        
        final_ret = (df.iloc[-1] - 1000) / 10
        
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

        plt.title("v2.8 포트폴리오 성과 분석 (배당 복리 재투자 반영)", fontsize=16, pad=20)
        plt.ylabel("정규화 가격 (1000 기준)", fontsize=12)
        plt.legend(loc='upper left', fontsize=11, frameon=True, edgecolor='gray')
        plt.grid(axis='y', alpha=0.1)
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
    # 종목과 수량을 여기에 추가하세요
    bt.add_stock('O', 3)      # 리얼티 인컴
    bt.add_stock('AAPL', 3)   # 애플
    bt.add_stock('005930.KS', 11) # 삼성전자
    
    bt.run_backtest('2023-01-01', end_date)

if __name__ == "__main__":
    testCase()