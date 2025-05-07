"""
Data collection script for Stock Market Analysis Project.
This script fetches stock data from Yahoo Finance API and stores it in the SQLite database.
"""

import os
import sqlite3
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import time

# Define the database path
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'stock_data.db')

def get_stock_symbols():
    """Retrieve stock symbols from the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, symbol FROM stocks")
    stocks = cursor.fetchall()
    conn.close()
    return stocks

def fetch_stock_data(symbol, start_date, end_date):
    """Fetch historical stock data from Yahoo Finance."""
    try:
        # Add a small delay to avoid rate limiting
        time.sleep(1)
        
        # Fetch historical data
        stock_data = yf.download(
            symbol, 
            start=start_date, 
            end=end_date, 
            progress=False
        )
        
        # Handle multi-index columns if present
        if isinstance(stock_data.columns, pd.MultiIndex):
            # Flatten the multi-index columns
            stock_data.columns = [col[0] for col in stock_data.columns]
        
        # Reset index to make date a column
        stock_data = stock_data.reset_index()
        
        # Convert date to string format for SQLite
        stock_data['Date'] = stock_data['Date'].dt.strftime('%Y-%m-%d')
        
        return stock_data
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return None

def calculate_technical_indicators(df):
    """Calculate technical indicators for the stock data."""
    # Make a copy to avoid SettingWithCopyWarning
    df_indicators = df.copy()
    
    # Calculate Simple Moving Averages
    df_indicators['SMA_20'] = df['Close'].rolling(window=20).mean()
    df_indicators['SMA_50'] = df['Close'].rolling(window=50).mean()
    df_indicators['SMA_200'] = df['Close'].rolling(window=200).mean()
    
    # Calculate RSI (Relative Strength Index)
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    
    rs = avg_gain / avg_loss
    df_indicators['RSI_14'] = 100 - (100 / (1 + rs))
    
    # Calculate MACD (Moving Average Convergence Divergence)
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df_indicators['MACD'] = ema_12 - ema_26
    df_indicators['MACD_Signal'] = df_indicators['MACD'].ewm(span=9, adjust=False).mean()
    
    # Calculate Bollinger Bands
    sma_20 = df['Close'].rolling(window=20).mean()
    std_20 = df['Close'].rolling(window=20).std()
    df_indicators['Bollinger_Upper'] = sma_20 + (std_20 * 2)
    df_indicators['Bollinger_Middle'] = sma_20
    df_indicators['Bollinger_Lower'] = sma_20 - (std_20 * 2)
    
    return df_indicators

def store_stock_data(stock_id, data):
    """Store stock data and technical indicators in the database."""
    if data is None or data.empty:
        return
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Calculate technical indicators
    data_with_indicators = calculate_technical_indicators(data)
    
    # Store stock prices
    for _, row in data.iterrows():
        # Check if 'Adj Close' column exists, otherwise use 'Close'
        adj_close = row['Close']
        if 'Adj Close' in row:
            adj_close = row['Adj Close']
        
        cursor.execute('''
        INSERT OR REPLACE INTO stock_prices 
        (stock_id, date, open, high, low, close, adj_close, volume)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            int(stock_id),
            str(row['Date']),
            float(row['Open']),
            float(row['High']),
            float(row['Low']),
            float(row['Close']),
            float(adj_close),
            int(row['Volume'])
        ))
    
    # Store technical indicators
    for _, row in data_with_indicators.iterrows():
        if pd.notna(row['SMA_20']):  # Only insert if we have calculated indicators
            cursor.execute('''
            INSERT OR REPLACE INTO technical_indicators
            (stock_id, date, sma_20, sma_50, sma_200, rsi_14, macd, macd_signal, 
            bollinger_upper, bollinger_middle, bollinger_lower)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                int(stock_id),
                str(row['Date']),
                float(row['SMA_20']) if pd.notna(row['SMA_20']) else None,
                float(row['SMA_50']) if pd.notna(row['SMA_50']) else None,
                float(row['SMA_200']) if pd.notna(row['SMA_200']) else None,
                float(row['RSI_14']) if pd.notna(row['RSI_14']) else None,
                float(row['MACD']) if pd.notna(row['MACD']) else None,
                float(row['MACD_Signal']) if pd.notna(row['MACD_Signal']) else None,
                float(row['Bollinger_Upper']) if pd.notna(row['Bollinger_Upper']) else None,
                float(row['Bollinger_Middle']) if pd.notna(row['Bollinger_Middle']) else None,
                float(row['Bollinger_Lower']) if pd.notna(row['Bollinger_Lower']) else None
            ))
    
    conn.commit()
    conn.close()

def main():
    """Main function to collect and store stock data."""
    # Define date range (5 years of historical data)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5*365)
    
    # Format dates for Yahoo Finance
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')
    
    print(f"Collecting stock data from {start_date_str} to {end_date_str}")
    
    # Get stock symbols from database
    stocks = get_stock_symbols()
    
    # Fetch and store data for each stock
    for stock_id, symbol in stocks:
        print(f"Processing {symbol}...")
        stock_data = fetch_stock_data(symbol, start_date_str, end_date_str)
        if stock_data is not None:
            print(f"Data columns for {symbol}: {stock_data.columns.tolist()}")
            store_stock_data(stock_id, stock_data)
            print(f"Successfully stored data for {symbol}")
        else:
            print(f"Failed to collect data for {symbol}")
    
    print("Data collection completed!")

if __name__ == "__main__":
    main()
