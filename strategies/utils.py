import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm
import plotly.io as pio
from binance.client import Client

client = Client()

def fetch_klines_for_ticker(ticker, interval, start_time):
    """
    Fetch historical klines (CloseV data) from Binance for a specific ticker.
    """
    df = client.get_historical_klines(ticker, interval, start_time)
    columns = [
        'Open time', 'Open', 'High', 'Low', 'Close', 'Volume1',
        'Close time', 'Volume', 'Adj Close',
        'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'
    ]
    df = pd.DataFrame(df, columns=columns)

    # Drop unnecessary columns
    df = df.drop(['Close time', 'Volume1', 'Taker buy base asset volume', 
                  'Taker buy quote asset volume', 'Ignore'], axis='columns')

    # Convert columns to numeric
    df['Open'] = pd.to_numeric(df['Open'], errors='coerce')
    df['High'] = pd.to_numeric(df['High'], errors='coerce')
    df['Low'] = pd.to_numeric(df['Low'], errors='coerce')
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
    df['Adj Close'] = pd.to_numeric(df['Adj Close'], errors='coerce')

    # Convert 'Open time' to datetime and set it as the index
    df['Datetime'] = pd.to_datetime(df['Open time'], unit='ms')
    df = df.drop(['Open time'], axis='columns')
    df.set_index('Datetime', inplace=True)
    df.index = df.index.tz_localize('UTC')

    # Remove rows that do not have enough data for indicators
    df = df[51:]
    
    return df

def fetch_multiple_tickers(tickers, interval="1m", days_ago="5 day ago UTC"):
    """
    Fetch historical data for multiple tickers.
    
    Parameters:
        tickers (str): Comma-separated ticker symbols (e.g., "BTCUSDT,ETHUSDT").
        interval (str): Time interval for the kline data (e.g., "1m" for 1-minute).
        days_ago (str): The starting time for historical data (e.g., "5 day ago UTC").
        
    Returns:
        dict: A dictionary where keys are ticker symbols and values are DataFrames of the CloseV data.
    """
    tickers_list = [ticker.strip() for ticker in tickers.split(",")]
    ticker_data = {}
    
    for ticker in tqdm(tickers_list):
        df = fetch_klines_for_ticker(ticker, interval, days_ago)
        ticker_data[ticker] = df
        
    return ticker_data

def clean_df(df):
    df=df[df.High!=df.Low]
    return df

def is_bullish_rejection(open_price, close_price, high_price, low_price, tail_body_ratio, body_perc_limit = 0.5e-4):
    body = abs(close_price - open_price)
    lower_wick = min(open_price, close_price) - low_price
    upper_wick = high_price - max(close_price, open_price)
    
    currentbody = abs(open_price - close_price)
    c1 = currentbody > body_perc_limit*close_price
    return ( c1 and (lower_wick > body * tail_body_ratio) and (upper_wick < body / tail_body_ratio) )

def is_bearish_rejection(open_price, close_price, high_price, low_price, tail_body_ratio, body_perc_limit = 0.5e-4):
    body = abs(close_price - open_price)
    upper_wick = high_price - max(close_price, open_price)
    lower_wick = min(open_price, close_price) - low_price
    
    currentbody = abs(open_price - close_price)
    c1 = currentbody > body_perc_limit*close_price
    return ( c1 and (upper_wick > body * tail_body_ratio) and (lower_wick < body / tail_body_ratio) )

def detect_rejection_candles(df, tail_body_ratio=2, body_perc_limit = 0.5e-4):
    """
    Adds a 'candlesignal_rej' column to the DataFrame indicating if the current candle is a rejection candle.
    Bullish rejection candles are marked with 2, bearish rejection candles are marked with 1, and others with 0.

    Parameters:
    df (DataFrame): DataFrame containing the stock data with 'Open', 'High', 'Low', 'Close' columns.
    tail_body_ratio (float): The ratio of tail/wick to body to consider a candle as a rejection candle.

    Returns:
    DataFrame: The original DataFrame with an additional 'candlesignal_rej' column.
    """

    df['candlesignal_rej'] = 0  # Initialize the candlesignal_rej column with 0

    for i in range(len(df)):
        open_price = df.iloc[i]['Open']
        close_price = df.iloc[i]['Close']
        high_price = df.iloc[i]['High']
        low_price = df.iloc[i]['Low']

        if is_bullish_rejection(open_price, close_price, high_price, low_price, tail_body_ratio, body_perc_limit):
            df.at[df.index[i], 'candlesignal_rej'] = 2
        elif is_bearish_rejection(open_price, close_price, high_price, low_price, tail_body_ratio, body_perc_limit):
            df.at[df.index[i], 'candlesignal_rej'] = 1

    return df

def add_pointpos_column(df, signal_column):
    """
    Adds a 'pointpos' column to the DataFrame to indicate the position of support and resistance points.
    
    Parameters:
    df (DataFrame): DataFrame containing the stock data with the specified SR column, 'Low', and 'High' columns.
    sr_column (str): The name of the column to consider for the SR (support/resistance) points.
    
    Returns:
    DataFrame: The original DataFrame with an additional 'pointpos' column.
    """
    def pointpos(row):
        if row[signal_column] == 2:
            return row['Low'] - 1e-4
        elif row[signal_column] == 1:
            return row['High'] + 1e-4
        else:
            return np.nan

    df['pointpos'] = df.apply(lambda row: pointpos(row), axis=1)
    return df

def plot_candlestick_with_signals(df, start_index, num_rows):
    """
    Plots a candlestick chart with signal points.
    
    Parameters:
    df (DataFrame): DataFrame containing the stock data with 'Open', 'High', 'Low', 'Close', and 'pointpos' columns.
    start_index (int): The starting index for the subset of data to plot.
    num_rows (int): The number of rows of data to plot.
    
    Returns:
    None
    """
    df_subset = df[start_index:start_index + num_rows]
    
    fig = make_subplots(rows=1, cols=1)
    
    fig.add_trace(go.Candlestick(x=df_subset.index,
                                 open=df_subset['Open'],
                                 high=df_subset['High'],
                                 low=df_subset['Low'],
                                 close=df_subset['Close'],
                                 name='Candlesticks'),
                  row=1, col=1)
    
    fig.add_trace(go.Scatter(x=df_subset.index, y=df_subset['pointpos'], mode="markers",
                             marker=dict(size=10, color="MediumPurple", symbol='circle'),
                             name="Entry Points"),
                  row=1, col=1)
    
    fig.update_layout(
        width=1200, 
        height=800, 
        plot_bgcolor='black',
        paper_bgcolor='black',
        font=dict(color='white'),
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False),
        showlegend=True,
        legend=dict(
            x=0.01,
            y=0.99,
            traceorder="normal",
            font=dict(
                family="sans-serif",
                size=12,
                color="white"
            ),
            bgcolor="black",
            bordercolor="gray",
            borderwidth=2
        )
    )
    fig.show(renderer="browser")