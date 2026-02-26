import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

class StockAnalyzer:
    def __init__(self):
        # KRX 주식 리스트를 다운로드
        self.krx_list = pd.read_html("http://kind.krx.co.kr/corpgeneral/corpList.do?method=download", encoding = 'cp949')[0]
        self.krx_list['종목코드'] = self.krx_list['종목코드'].astype(str).str.zfill(6)

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

    def get_stock_data(self, ticker, start_date, end_date):
        """
        주식 티커와 기간을 입력하면 해당 주식의 데이터를 반환합니다.
        """
        try:
            stock = yf.Ticker(f"{ticker}.KS")
            return stock.history(start=start_date, end=end_date)
        except Exception as e:
            print(f"주식 데이터를 가져오는 데 실패했습니다: {e}")
            return None

    def calculate_returns(self, data, quantity):
        """
        주식 데이터와 수량을 입력하면 TR과 PR을 계산합니다.
        """
        data['PR'] = (data['Close'] - data['Close'].iloc[0]) / data['Close'].iloc[0] * 100
        data['TR'] = (data['Close'] + data['Dividends'].cumsum() - data['Close'].iloc[0]) / data['Close'].iloc[0] * 100
        data['PR'] *= quantity
        data['TR'] *= quantity
        print(data)
        return data

    def plot_returns(self, stock_name, quantity, start_date, end_date):
        """
        주식 이름과 수량을 입력하면 TR과 PR을 비교하는 그래프를 그립니다.
        """
        ticker = self.get_korean_ticker(stock_name)
        if ticker is None:
            return

        data = self.get_stock_data(ticker, start_date, end_date)
        if data is None:
            return

        data = self.calculate_returns(data, quantity)

        plt.figure(figsize=(10, 6))
        plt.plot(data.index, data['PR'], label='Price Return (PR)')
        plt.plot(data.index, data['TR'], label='Total Return (TR)')
        plt.title(f'{stock_name}의 PR vs TR 비교')
        plt.xlabel('Date')
        plt.ylabel('Return (%)')
        plt.legend()
        plt.grid(True)
        plt.show()

# 사용 예시
analyzer = StockAnalyzer()
analyzer.plot_returns('하나금융지주', 10, '2005-01-01', '2024-01-01')
