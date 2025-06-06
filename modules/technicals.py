import yfinance as yf
import pandas as pd
import ta

def fetch_stock_data(symbol, period='6mo', interval='1d'):
    """
    Fetch historical stock data using yfinance.
    """
    df = yf.download(symbol, period=period, interval=interval)
    return df


def calculate_sma(df, window=14):
    sma = df['Close'].rolling(window=window).mean()
    return sma.values.flatten()  # ensure 1D array

def calculate_rsi(df, window=14):
    rsi = ta.momentum.RSIIndicator(df['Close'].squeeze(), window=window).rsi()
    return rsi.values.flatten()  # ensure 1D array

def calculate_macd(df):
    close_prices = pd.Series(df['Close'].values.ravel())  # robust against (n,1)
    macd = ta.trend.MACD(close_prices)
    macd_line = macd.macd().values.flatten()  # ensure 1D array
    macd_signal = macd.macd_signal().values.flatten()  # ensure 1D array
    return macd_line, macd_signal

