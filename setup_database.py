"""
Database setup script for Stock Market Analysis Project.
This script creates a SQLite database and tables for storing stock data.
"""

import sqlite3
import os

# Define the database path
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'stock_data.db')

def create_database():
    """Create the SQLite database and necessary tables."""
    # Ensure the data directory exists
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    
    # Connect to the database (will create it if it doesn't exist)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create stocks table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS stocks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT NOT NULL,
        company_name TEXT NOT NULL,
        sector TEXT,
        industry TEXT,
        UNIQUE(symbol)
    )
    ''')
    
    # Create stock_prices table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS stock_prices (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        stock_id INTEGER NOT NULL,
        date TEXT NOT NULL,
        open REAL,
        high REAL,
        low REAL,
        close REAL,
        adj_close REAL,
        volume INTEGER,
        FOREIGN KEY (stock_id) REFERENCES stocks (id),
        UNIQUE(stock_id, date)
    )
    ''')
    
    # Create financial_metrics table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS financial_metrics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        stock_id INTEGER NOT NULL,
        date TEXT NOT NULL,
        pe_ratio REAL,
        eps REAL,
        market_cap REAL,
        dividend_yield REAL,
        beta REAL,
        FOREIGN KEY (stock_id) REFERENCES stocks (id),
        UNIQUE(stock_id, date)
    )
    ''')
    
    # Create technical_indicators table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS technical_indicators (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        stock_id INTEGER NOT NULL,
        date TEXT NOT NULL,
        sma_20 REAL,
        sma_50 REAL,
        sma_200 REAL,
        rsi_14 REAL,
        macd REAL,
        macd_signal REAL,
        bollinger_upper REAL,
        bollinger_middle REAL,
        bollinger_lower REAL,
        FOREIGN KEY (stock_id) REFERENCES stocks (id),
        UNIQUE(stock_id, date)
    )
    ''')
    
    # Create predictions table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        stock_id INTEGER NOT NULL,
        date TEXT NOT NULL,
        model_name TEXT NOT NULL,
        predicted_price REAL,
        confidence_interval_lower REAL,
        confidence_interval_upper REAL,
        FOREIGN KEY (stock_id) REFERENCES stocks (id),
        UNIQUE(stock_id, date, model_name)
    )
    ''')
    
    # Insert some initial stock symbols
    stocks = [
        ('NFLX', 'Netflix Inc.', 'Communication Services', 'Entertainment'),
        ('AAPL', 'Apple Inc.', 'Technology', 'Consumer Electronics'),
        ('MSFT', 'Microsoft Corporation', 'Technology', 'Software—Infrastructure'),
        ('AMZN', 'Amazon.com Inc.', 'Consumer Cyclical', 'Internet Retail'),
        ('GOOGL', 'Alphabet Inc.', 'Communication Services', 'Internet Content & Information'),
        ('META', 'Meta Platforms Inc.', 'Communication Services', 'Internet Content & Information'),
        ('TSLA', 'Tesla Inc.', 'Consumer Cyclical', 'Auto Manufacturers'),
        ('JPM', 'JPMorgan Chase & Co.', 'Financial Services', 'Banks—Diversified'),
        ('V', 'Visa Inc.', 'Financial Services', 'Credit Services'),
        ('WMT', 'Walmart Inc.', 'Consumer Defensive', 'Discount Stores')
    ]
    
    cursor.executemany('''
    INSERT OR IGNORE INTO stocks (symbol, company_name, sector, industry)
    VALUES (?, ?, ?, ?)
    ''', stocks)
    
    # Commit changes and close connection
    conn.commit()
    conn.close()
    
    print(f"Database created successfully at {DB_PATH}")
    return DB_PATH

if __name__ == "__main__":
    create_database()
