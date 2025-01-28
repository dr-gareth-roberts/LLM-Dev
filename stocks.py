from datetime import datetime, timedelta
import logging

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import pandas as pd
import seaborn as sns
import yfinance as yf

# Configure logging
logging.basicConfig(level=logging.INFO)

# Function to retrieve stock data for NVIDIA
def get_nvidia_stock_data():
    """
    Retrieve stock data for NVIDIA from Yahoo Finance.

    Returns:
        pd.DataFrame: DataFrame containing NVIDIA stock data.
    """
    nvidia_data = yf.download('NVDA', start='2020-01-01', end='2025-01-28')
    logging.info('Checking for NaN values in NVIDIA data.')
    nvidia_data.dropna(inplace=True)
    return nvidia_data


def plot_stock_price(ticker="NVDA", days=7):
    """
    Plot the stock price with error bars for a given ticker.

    Args:
        ticker (str): The stock ticker symbol.
        days (int): Number of days to fetch data for.
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    stock = yf.Ticker(ticker)
    df = stock.history(start=start_date, end=end_date)
    logging.info('Checking for NaN values in stock data.')
    df.dropna(inplace=True)
    error = df['Close'].std()

    try:
        plt.figure(figsize=(12, 6))
        plt.errorbar(df.index, df['Close'], yerr=error, fmt='-o', label='Close Price with Error Bars')
        plt.title(f'{ticker} Stock Price with Error Bars')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        plt.show()
    except Exception as e:
        logging.error(f"Error during plotting: {str(e)}")


def plot_additional_visualizations(ticker="NVDA", days=30):
    """
    Plot additional visualizations for the specified ticker:
    1. Line Plot of Closing Prices
    2. Moving Average Plot
    3. Volume Plot
    4. Box Plot of Closing Prices
    5. Heatmap of Correlation

    Args:
        ticker (str): The ticker symbol (default: "NVDA")
        days (int): The number of days to fetch (default: 30)
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    stock = yf.Ticker(ticker)
    df = stock.history(start=start_date, end=end_date)
    logging.info('Checking for NaN values in stock data.')
    df.dropna(inplace=True)

    try:
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=df.reset_index(), x='Date', y='Close', label='Close Price')
        plt.title(f'{ticker} Closing Prices')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.xticks(rotation=45)
        plt.grid()
        plt.show()
    except Exception as e:
        logging.error(f"Error during plotting: {str(e)}")

    df['Moving Average'] = df['Close'].rolling(window=5).mean()

    try:
        plt.figure(figsize=(12, 6))
        data = df.reset_index()
        sns.lineplot(data=data, x='Date', y='Close', label='Close Price')
        sns.lineplot(data=data, x='Date', y='Moving Average', label='5-Day Moving Average', color='orange')
        plt.title(f'{ticker} Stock Price and Moving Average')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.xticks(rotation=45)
        plt.grid()
        plt.show()
    except Exception as e:
        logging.error(f"Error during plotting: {str(e)}")

    try:
        plt.figure(figsize=(12, 6))
        data = df.reset_index()
        sns.barplot(data=data, x=data.index, y='Volume', color='gray')
        plt.title(f'{ticker} Trading Volume')
        plt.xlabel('Date')
        plt.ylabel('Volume')
        plt.xticks(rotation=45)
        plt.grid()
        plt.show()
    except Exception as e:
        logging.error(f"Error during plotting: {str(e)}")

    try:
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=pd.DataFrame({'Close': df['Close'].values}), y='Close')
        plt.title(f'{ticker} Closing Price Distribution')
        plt.ylabel('Price')
        plt.grid()
        plt.show()
    except Exception as e:
        logging.error(f"Error during plotting: {str(e)}")

    try:
        plt.figure(figsize=(10, 8))
        correlation = df[['Open', 'High', 'Low', 'Close', 'Volume']].corr()
        sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title(f'{ticker} Correlation Heatmap')
        plt.show()
    except Exception as e:
        logging.error(f"Error during plotting: {str(e)}")


def plot_faang_distribution(start_date='2020-01-01', end_date=datetime.now()):
    """
    Plot the distribution curves of FAANG stocks closing prices.

    Args:
        start_date (str): The start date for data retrieval (default: '2020-01-01')
        end_date (datetime): The end date for data retrieval (default: datetime.now())
    """
    faang_stocks = {'AAPL': 'Apple', 'AMZN': 'Amazon', 'META': 'Meta Platforms', 'NFLX': 'Netflix', 'GOOGL': 'Google'}
    stock_data = {}

    for ticker in faang_stocks.keys():
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date)
        logging.info('Checking for NaN values in stock data.')
        df.dropna(inplace=True)

        if df.empty:
            logging.warning(f"No data available for {faang_stocks[ticker]} ({ticker}). Skipping.")
            continue

        stock_data[ticker] = df['Close']

    df = pd.DataFrame(stock_data)

    if df.isna().values.sum() > 0:
        logging.warning("Data contains NaN values. Please check the data source.")
        logging.info(df.isna().sum())
        return

    try:
        plt.figure(figsize=(12, 6))
        for ticker in faang_stocks.keys():
            data = pd.DataFrame({faang_stocks[ticker]: df[ticker].dropna().values})
            sns.kdeplot(data=data, x=faang_stocks[ticker], label=faang_stocks[ticker])
        plt.title('Distribution Curves of FAANG Stocks Closing Prices')
        plt.xlabel('Price')
        plt.ylabel('Density')
        plt.legend()
        plt.grid()
        plt.show()
    except Exception as e:
        logging.error(f"Error during plotting: {str(e)}")


def plot_candlestick_with_indicators(ticker="NVDA", days=60):
    """
    Plot a candlestick chart with indicators for a given stock ticker.

    Args:
        ticker (str): The stock ticker symbol (default: "NVDA")
        days (int): The number of days to fetch data for (default: 60)
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    stock = yf.Ticker(ticker)
    df = stock.history(start=start_date, end=end_date)
    logging.info('Checking for NaN values in stock data.')
    df.dropna(inplace=True)

    if len(df) == 0:
        logging.warning(f"No data available for {ticker} in the given date range.")
        return

    logging.info(f"Data frame shape: {df.shape}")
    logging.info(df.head())

    if df.isna().values.sum() > 0:
        logging.warning("Data contains NaN values. Please check the data source.")
        logging.info(df.isna().sum())
        return

    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()

    if df['MA20'].isna().values.sum() > 0 or df['MA50'].isna().values.sum() > 0:
        logging.warning("Moving averages contain NaN values. Adjusting data.")
        logging.info("MA20 NaN count:", df['MA20'].isna().sum())
        logging.info("MA50 NaN count:", df['MA50'].isna().sum())
        df.dropna(subset=['MA20', 'MA50'], inplace=True)

    if len(df) == 0:
        logging.warning("Data frame is empty after dropping NaN values. Cannot plot.")
        return

    add_plots = [
        mpf.make_addplot(df['MA20'], color='blue'),
        mpf.make_addplot(df['MA50'], color='red')
    ]

    try:
        mpf.plot(df, type='candle', volume=True, addplot=add_plots, title=f'{ticker} Candlestick Chart with MA', style='yahoo')
    except Exception as e:
        logging.error(f"Error during plotting: {str(e)}")


def plot_volatility_analysis(ticker="NVDA", days=180):
    """
    Plot the volatility analysis for a given stock ticker over a specified period of time.

    Args:
        ticker (str): The stock ticker symbol (default: "NVDA")
        days (int): The number of days to analyze volatility for (default: 180)
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    stock = yf.Ticker(ticker)
    df = stock.history(start=start_date, end=end_date)
    logging.info('Checking for NaN values in stock data.')
    df.dropna(inplace=True)

    df['Log Returns'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Volatility'] = df['Log Returns'].rolling(window=21).std() * np.sqrt(252)

    if df.isna().values.sum() > 0:
        logging.warning("Data contains NaN values. Please check the data source.")
        logging.info(df.isna().sum())
        return

    try:
        plt.figure(figsize=(12, 6))
        plt.plot(df.index, df['Volatility'], label='21-Day Volatility')
        plt.title(f'{ticker} Volatility Analysis')
        plt.xlabel('Date')
        plt.ylabel('Volatility')
        plt.grid()
        plt.show()
    except Exception as e:
        logging.error(f"Error during plotting: {str(e)}")


def plot_returns_distribution(ticker="NVDA", days=365):
    """
    Plot the distribution of returns for a given stock ticker over a specified period of time.

    Args:
        ticker (str): The stock ticker symbol (default: "NVDA")
        days (int): The number of days to plot the returns distribution for (default: 365)
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    stock = yf.Ticker(ticker)
    df = stock.history(start=start_date, end=end_date)
    logging.info('Checking for NaN values in stock data.')
    df.dropna(inplace=True)

    df['Returns'] = df['Close'].pct_change()

    if df.isna().values.sum() > 0:
        logging.warning("Data contains NaN values. Please check the data source.")
        logging.info(df.isna().sum())
        return

    try:
        plt.figure(figsize=(12, 6))
        data = pd.DataFrame({'Returns': df['Returns'].dropna().values})
        sns.histplot(data=data, x='Returns', bins=50, kde=True, color='purple')
        plt.title(f'{ticker} Returns Distribution')
        plt.xlabel('Daily Returns')
        plt.ylabel('Frequency')
        plt.grid()
        plt.show()
    except Exception as e:
        logging.error(f"Error during plotting: {str(e)}")


def plot_sector_performance(ticker="NVDA", sector_tickers=None, days=180):
    """
    Plot the comparative sector performance for a given stock ticker.

    Args:
        ticker (str): The stock ticker symbol (default: "NVDA")
        sector_tickers (list): A list of sector tickers (default: None)
        days (int): The number of days to analyze sector performance for (default: 180)
    """
    if sector_tickers is None:
        sector_tickers = ['AMD', 'INTC', 'QCOM']  # Example tickers from the same sector

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    try:
        plt.figure(figsize=(12, 6))
        for st in sector_tickers + [ticker]:
            stock = yf.Ticker(st)
            df = stock.history(start=start_date, end=end_date)
            logging.info('Checking for NaN values in stock data.')
            df.dropna(inplace=True)
            df['Normalized'] = df['Close'] / df['Close'].iloc[0]
            plt.plot(df.index, df['Normalized'], label=st)

        plt.title('Comparative Sector Performance')
        plt.xlabel('Date')
        plt.ylabel('Normalized Price')
        plt.legend()
        plt.grid()
        plt.show()
    except Exception as e:
        logging.error(f"Error during plotting: {str(e)}")


def plot_multi_timeframe_momentum(ticker="NVDA", timeframes=[30, 90, 180]):
    """
    Plot the multi-timeframe momentum analysis for a given stock ticker.

    Args:
        ticker (str): The stock ticker symbol (default: "NVDA")
        timeframes (list): A list of timeframes to analyze momentum for (default: [30, 90, 180])
    """
    end_date = datetime.now()

    try:
        plt.figure(figsize=(12, 6))
        for days in timeframes:
            start_date = end_date - timedelta(days=days)
            stock = yf.Ticker(ticker)
            df = stock.history(start=start_date, end=end_date)
            logging.info('Checking for NaN values in stock data.')
            df.dropna(inplace=True)
            df['Momentum'] = df['Close'] - df['Close'].shift(1)
            plt.plot(df.index, df['Momentum'], label=f'{days}-Day Momentum')

        plt.title(f'{ticker} Multi-Timeframe Momentum Analysis')
        plt.xlabel('Date')
        plt.ylabel('Momentum')
        plt.legend()
        plt.grid()
        plt.show()
    except Exception as e:
        logging.error(f"Error during plotting: {str(e)}")


def plot_bollinger_bands(ticker="NVDA", days=60):
    """
    Plot the Bollinger Bands for a given stock ticker.

    Args:
        ticker (str): The stock ticker symbol (default: "NVDA")
        days (int): The number of days to analyze Bollinger Bands for (default: 60)
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    stock = yf.Ticker(ticker)
    df = stock.history(start=start_date, end=end_date)
    logging.info('Checking for NaN values in stock data.')
    df.dropna(inplace=True)

    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['Upper Band'] = df['MA20'] + 2*df['Close'].rolling(window=20).std()
    df['Lower Band'] = df['MA20'] - 2*df['Close'].rolling(window=20).std()

    if df.isna().values.sum() > 0:
        logging.warning("Data contains NaN values. Please check the data source.")
        logging.info(df.isna().sum())
        return

    try:
        plt.figure(figsize=(12, 6))
        plt.plot(df.index, df['Close'], label='Close Price')
        plt.plot(df.index, df['MA20'], label='20-Day MA', color='green')
        plt.plot(df.index, df['Upper Band'], label='Upper Band', color='red')
        plt.plot(df.index, df['Lower Band'], label='Lower Band', color='blue')
        plt.fill_between(df.index, df['Lower Band'], df['Upper Band'], color='grey', alpha=0.1)
        plt.title(f'{ticker} Bollinger Bands')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid()
        plt.show()
    except Exception as e:
        logging.error(f"Error during plotting: {str(e)}")


def plot_rsi(ticker="NVDA", days=60):
    """
    Plot the Relative Strength Index (RSI) for a given stock ticker.

    Args:
        ticker (str): The stock ticker symbol (default: "NVDA")
        days (int): The number of days to analyze RSI for (default: 60)
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    stock = yf.Ticker(ticker)
    df = stock.history(start=start_date, end=end_date)
    logging.info('Checking for NaN values in stock data.')
    df.dropna(inplace=True)

    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    if df.isna().values.sum() > 0:
        logging.warning("Data contains NaN values. Please check the data source.")
        logging.info(df.isna().sum())
        return

    try:
        plt.figure(figsize=(12, 6))
        plt.plot(df.index, df['RSI'], label='RSI', color='purple')
        plt.axhline(70, linestyle='--', alpha=0.5, color='red')
        plt.axhline(30, linestyle='--', alpha=0.5, color='green')
        plt.title(f'{ticker} Relative Strength Index (RSI)')
        plt.xlabel('Date')
        plt.ylabel('RSI')
        plt.legend()
        plt.grid()
        plt.show()
    except Exception as e:
        logging.error(f"Error during plotting: {str(e)}")


def plot_macd(ticker="NVDA", days=60):
    """
    Plot the Moving Average Convergence Divergence (MACD) for a given stock ticker.

    Args:
        ticker (str): The stock ticker symbol (default: "NVDA")
        days (int): The number of days to analyze MACD for (default: 60)
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    stock = yf.Ticker(ticker)
    df = stock.history(start=start_date, end=end_date)
    logging.info('Checking for NaN values in stock data.')
    df.dropna(inplace=True)

    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    if df.isna().values.sum() > 0:
        logging.warning("Data contains NaN values. Please check the data source.")
        logging.info(df.isna().sum())
        return

    try:
        plt.figure(figsize=(12, 6))
        plt.plot(df.index, df['MACD'], label='MACD', color='blue')
        plt.plot(df.index, df['Signal'], label='Signal Line', color='red')
        plt.title(f'{ticker} MACD')
        plt.xlabel('Date')
        plt.ylabel('MACD')
        plt.legend()
        plt.grid()
        plt.show()
    except Exception as e:
        logging.error(f"Error during plotting: {str(e)}")


def plot_candlestick_patterns(ticker="NVDA", days=60):
    """
    Plot the candlestick patterns for a given stock ticker.

    Args:
        ticker (str): The stock ticker symbol (default: "NVDA")
        days (int): The number of days to analyze candlestick patterns for (default: 60)
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    stock = yf.Ticker(ticker)
    df = stock.history(start=start_date, end=end_date)
    logging.info('Checking for NaN values in stock data.')
    df.dropna(inplace=True)

    if len(df) == 0:
        logging.warning(f"No data available for {ticker} in the given date range.")
        return

    df['Hammer'] = ((df['Close'] > df['Open']) &
                    ((df['Close'] - df['Low']) > 2*(df['Open'] - df['Close'])) &
                    ((df['High'] - df['Close']) < 0.1*(df['Close'] - df['Open'])))

    try:
        plt.figure(figsize=(12, 6))
        mpf.plot(df, type='candle', volume=True, title=f'{ticker} Candlestick Patterns', style='yahoo')
        plt.title(f'{ticker} Candlestick Patterns')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.grid()
        plt.show()
    except Exception as e:
        logging.error(f"Error during plotting: {str(e)}")


def plot_combined_visualizations(ticker: str = 'NVDA', days: int = 60) -> None:
    """
    Plot a collection of technical indicators for a given stock ticker.

    Args:
        ticker (str): The stock ticker symbol (default: "NVDA")
        days (int): The number of days to analyze technical indicators for (default: 60)
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    stock = yf.Ticker(ticker)
    df = stock.history(start=start_date, end=end_date)
    logging.info('Checking for NaN values in stock data.')
    df.dropna(inplace=True)

    if df.empty:
        logging.warning(f"No data available for {ticker} in the given date range.")
        return

    df['ma_20'] = df['Close'].rolling(window=20).mean()
    df['upper_band'] = df['ma_20'] + 2 * df['Close'].rolling(window=20).std()
    df['lower_band'] = df['ma_20'] - 2 * df['Close'].rolling(window=20).std()
    df['ema_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['ema_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['macd'] = df['ema_12'] - df['ema_26']
    df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()

    # Calculate RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # Drop NaN values after calculating moving averages
    df.dropna(subset=['ma_20', 'upper_band', 'lower_band'], inplace=True)

    # Drop NaN values after calculating RSI
    df.dropna(subset=['rsi'], inplace=True)

    plt.style.use('default')
    fig, axes = plt.subplots(nrows=6, ncols=1, figsize=(14, 24), sharex=True)

    try:
        mpf.plot(df, type='candle', ax=axes[0], volume=False, style='yahoo')
        axes[0].plot(df.index, df['upper_band'], label='Upper Band', color='red', linestyle='--')
        axes[0].plot(df.index, df['lower_band'], label='Lower Band', color='blue', linestyle='--')
        axes[0].fill_between(df.index, df['lower_band'], df['upper_band'], color='grey', alpha=0.1)
        axes[0].set_title(f'{ticker} Candlestick Chart with Bollinger Bands')
        axes[0].set_ylabel('Price')
        axes[0].legend(loc='upper left')

        axes[1].plot(df.index, df['rsi'], label='RSI', color='purple')
        axes[1].axhline(70, linestyle='--', alpha=0.5, color='red', label='Overbought')
        axes[1].axhline(30, linestyle='--', alpha=0.5, color='green', label='Oversold')
        axes[1].set_title(f'{ticker} Relative Strength Index (RSI)')
        axes[1].set_ylabel('RSI')
        axes[1].legend(loc='upper left')

        axes[2].plot(df.index, df['macd'], label='MACD', color='blue')
        axes[2].plot(df.index, df['signal'], label='Signal Line', color='red')
        axes[2].set_title(f'{ticker} Moving Average Convergence Divergence (MACD)')
        axes[2].set_ylabel('MACD')
        axes[2].legend(loc='upper left')

        axes[3].bar(df.index, df['Volume'], color='grey')
        axes[3].set_title(f'{ticker} Trading Volume')
        axes[3].set_ylabel('Volume')

        axes[4].hist(df['Close'], bins=30, color='skyblue', edgecolor='black')
        axes[4].set_title(f'{ticker} Price Histogram')
        axes[4].set_xlabel('Price')
        axes[4].set_ylabel('Frequency')

        axes[5].plot(df.index, df['ma_20'], label='20-Day MA', color='orange')
        axes[5].plot(df.index, df['ema_12'], label='12-Day EMA', color='green')
        axes[5].plot(df.index, df['ema_26'], label='26-Day EMA', color='red')
        axes[5].set_title(f'{ticker} Moving Averages')
        axes[5].set_ylabel('Price')
        axes[5].set_xlabel('Date')
        axes[5].legend(loc='upper left')

        plt.tight_layout()
        plt.show()
    except Exception as e:
        logging.error(f"Error during plotting: {str(e)}")


if __name__ == "__main__":
    nvidia_stock_data = get_nvidia_stock_data()
    logging.info(nvidia_stock_data)
    plot_stock_price(ticker="NVDA", days=7)
    plot_additional_visualizations(ticker="NVDA", days=30)
    plot_faang_distribution(start_date='2020-01-01', end_date=datetime.now())
    plot_candlestick_with_indicators(ticker="NVDA", days=60)
    plot_volatility_analysis(ticker="NVDA", days=180)
    plot_returns_distribution(ticker="NVDA", days=365)
    plot_sector_performance(ticker="NVDA", sector_tickers=['AMD', 'INTC', 'QCOM'], days=180)
    plot_multi_timeframe_momentum(ticker="NVDA", timeframes=[30, 90, 180])
    plot_bollinger_bands(ticker="NVDA", days=60)
    plot_rsi(ticker="NVDA", days=60)
    plot_macd(ticker="NVDA", days=60)
    plot_candlestick_patterns(ticker="NVDA", days=60)
    plot_combined_visualizations(ticker="NVDA", days=60)