"""
Time Series Forecasting Models for Stock Market Analysis Project.
This script builds and evaluates various time series forecasting models for stock price prediction.
"""

import os
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Create models directory if it doesn't exist
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

# Create visualizations directory if it doesn't exist
VISUALIZATIONS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'visualizations')
os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)

# Define the database path
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'stock_data.db')

# Function to connect to the database
def get_connection():
    return sqlite3.connect(DB_PATH)

# Function to get stock data with technical indicators
def get_stock_data_with_indicators(symbol):
    conn = get_connection()
    
    query = """
    SELECT s.symbol, p.date, p.open, p.high, p.low, p.close, p.adj_close, p.volume,
           t.sma_20, t.sma_50, t.sma_200, t.rsi_14, t.macd, t.macd_signal,
           t.bollinger_upper, t.bollinger_middle, t.bollinger_lower
    FROM stocks s
    JOIN stock_prices p ON s.id = p.stock_id
    JOIN technical_indicators t ON s.id = t.stock_id AND p.date = t.date
    WHERE s.symbol = ?
    ORDER BY p.date
    """
    
    df = pd.read_sql_query(query, conn, params=(symbol,))
    conn.close()
    
    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    return df

# Function to prepare data for time series forecasting
def prepare_time_series_data(df, target_col='close', sequence_length=60):
    # Create a copy of the dataframe
    data = df.copy()
    
    # Add additional features
    data['day_of_week'] = data['date'].dt.dayofweek
    data['month'] = data['date'].dt.month
    data['year'] = data['date'].dt.year
    
    # Calculate returns
    data['return_1d'] = data[target_col].pct_change(1)
    data['return_5d'] = data[target_col].pct_change(5)
    data['return_10d'] = data[target_col].pct_change(10)
    data['return_20d'] = data[target_col].pct_change(20)
    
    # Calculate volatility
    data['volatility_5d'] = data['return_1d'].rolling(window=5).std()
    data['volatility_10d'] = data['return_1d'].rolling(window=10).std()
    data['volatility_20d'] = data['return_1d'].rolling(window=20).std()
    
    # Calculate price momentum
    data['momentum_5d'] = data[target_col] / data[target_col].shift(5) - 1
    data['momentum_10d'] = data[target_col] / data[target_col].shift(10) - 1
    data['momentum_20d'] = data[target_col] / data[target_col].shift(20) - 1
    
    # Calculate distance from moving averages
    data['dist_sma_20'] = data[target_col] / data['sma_20'] - 1
    data['dist_sma_50'] = data[target_col] / data['sma_50'] - 1
    data['dist_sma_200'] = data[target_col] / data['sma_200'] - 1
    
    # Drop rows with NaN values
    data = data.dropna()
    
    # Select features for the model
    features = ['open', 'high', 'low', 'close', 'volume', 
                'sma_20', 'sma_50', 'sma_200', 'rsi_14', 'macd', 'macd_signal',
                'bollinger_upper', 'bollinger_middle', 'bollinger_lower',
                'day_of_week', 'month', 'year',
                'return_1d', 'return_5d', 'return_10d', 'return_20d',
                'volatility_5d', 'volatility_10d', 'volatility_20d',
                'momentum_5d', 'momentum_10d', 'momentum_20d',
                'dist_sma_20', 'dist_sma_50', 'dist_sma_200']
    
    # Create a new dataframe with selected features
    model_data = data[['date'] + features].copy()
    
    # Create target variable (next day's closing price)
    model_data['target'] = model_data[target_col].shift(-1)
    
    # Drop rows with NaN values in target
    model_data = model_data.dropna()
    
    # Split data into training and testing sets (80% train, 20% test)
    train_size = int(len(model_data) * 0.8)
    train_data = model_data[:train_size]
    test_data = model_data[train_size:]
    
    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    # Fit scaler on training data only
    train_scaled = scaler.fit_transform(train_data[features])
    test_scaled = scaler.transform(test_data[features])
    
    # Create a separate scaler for the target variable
    target_scaler = MinMaxScaler(feature_range=(0, 1))
    train_target_scaled = target_scaler.fit_transform(train_data[['target']])
    test_target_scaled = target_scaler.transform(test_data[['target']])
    
    # Create sequences for LSTM
    X_train, y_train = create_sequences(train_scaled, train_target_scaled, sequence_length)
    X_test, y_test = create_sequences(test_scaled, test_target_scaled, sequence_length)
    
    return (X_train, y_train, X_test, y_test, 
            train_data, test_data, 
            scaler, target_scaler,
            features)

# Function to create sequences for LSTM
def create_sequences(X, y, sequence_length):
    X_seq, y_seq = [], []
    
    for i in range(len(X) - sequence_length):
        X_seq.append(X[i:i+sequence_length])
        y_seq.append(y[i+sequence_length])
    
    return np.array(X_seq), np.array(y_seq)

# Function to build and train LSTM model
def build_lstm_model(X_train, y_train, X_test, y_test, symbol):
    # Define LSTM model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    
    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=1
    )
    
    # Plot training history
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'LSTM Model Training and Validation Loss - {symbol}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(VISUALIZATIONS_DIR, f'{symbol}_lstm_training_history.png'))
    plt.close()
    
    # Save the model
    model.save(os.path.join(MODELS_DIR, f'{symbol}_lstm_model.h5'))
    
    return model

# Function to build and train traditional ML models
def build_traditional_models(train_data, test_data, features, symbol):
    # Prepare data
    X_train = train_data[features]
    y_train = train_data['target']
    X_test = test_data[features]
    y_test = test_data['target']
    
    # Initialize models
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    
    results = {}
    
    # Train and evaluate each model
    for name, model in models.items():
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        # Calculate metrics
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        train_mae = mean_absolute_error(y_train, train_pred)
        test_mae = mean_absolute_error(y_test, test_pred)
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        
        # Store results
        results[name] = {
            'model': model,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'test_predictions': test_pred
        }
        
        # Save the model
        with open(os.path.join(MODELS_DIR, f'{symbol}_{name.replace(" ", "_").lower()}_model.pkl'), 'wb') as f:
            pickle.dump(model, f)
    
    return results

# Function to evaluate and visualize model predictions
def evaluate_models(results, test_data, symbol):
    # Create a figure for model comparison
    plt.figure(figsize=(16, 8))
    
    # Plot actual prices
    plt.plot(test_data['date'], test_data['close'], label='Actual', color='black', linewidth=2)
    
    # Plot predictions for each model
    colors = ['blue', 'green', 'red']
    for i, (name, result) in enumerate(results.items()):
        plt.plot(test_data['date'], result['test_predictions'], label=f'{name} (RMSE: {result["test_rmse"]:.2f})', 
                 color=colors[i], alpha=0.7)
    
    plt.title(f'Stock Price Predictions for {symbol}', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Price (USD)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATIONS_DIR, f'{symbol}_model_predictions.png'))
    plt.close()
    
    # Create a table of model performance metrics
    metrics_df = pd.DataFrame({
        'Model': list(results.keys()),
        'Train RMSE': [result['train_rmse'] for result in results.values()],
        'Test RMSE': [result['test_rmse'] for result in results.values()],
        'Train MAE': [result['train_mae'] for result in results.values()],
        'Test MAE': [result['test_mae'] for result in results.values()],
        'Train R²': [result['train_r2'] for result in results.values()],
        'Test R²': [result['test_r2'] for result in results.values()]
    })
    
    # Sort by Test RMSE (lower is better)
    metrics_df = metrics_df.sort_values('Test RMSE')
    
    return metrics_df

# Function to evaluate LSTM model
def evaluate_lstm_model(model, X_test, y_test, test_data, target_scaler, symbol):
    # Make predictions
    y_pred_scaled = model.predict(X_test)
    
    # Inverse transform the predictions
    y_pred = target_scaler.inverse_transform(y_pred_scaled)
    y_true = target_scaler.inverse_transform(y_test)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    
    # Plot predictions
    plt.figure(figsize=(16, 8))
    
    # Adjust the test_data index to match the predictions
    test_dates = test_data['date'].iloc[len(test_data) - len(y_pred):]
    
    plt.plot(test_dates, y_true, label='Actual', color='black', linewidth=2)
    plt.plot(test_dates, y_pred, label=f'LSTM Predictions (RMSE: {rmse:.2f})', color='blue', alpha=0.7)
    
    plt.title(f'LSTM Stock Price Predictions for {symbol}', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Price (USD)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATIONS_DIR, f'{symbol}_lstm_predictions.png'))
    plt.close()
    
    return {
        'rmse': rmse,
        'mae': mae,
        'predictions': y_pred.flatten()
    }

# Function to perform risk analysis
def perform_risk_analysis(df, symbol):
    # Calculate daily returns
    returns = df['close'].pct_change().dropna()
    
    # Calculate volatility (annualized)
    volatility = returns.std() * np.sqrt(252)
    
    # Calculate Value at Risk (VaR) at 95% confidence level
    var_95 = np.percentile(returns, 5) * 100
    
    # Calculate Conditional Value at Risk (CVaR) / Expected Shortfall
    cvar_95 = returns[returns <= np.percentile(returns, 5)].mean() * 100
    
    # Calculate Maximum Drawdown
    cumulative_returns = (1 + returns).cumprod()
    running_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns / running_max - 1) * 100
    max_drawdown = drawdown.min()
    
    # Plot the distribution of returns
    plt.figure(figsize=(14, 8))
    
    # Histogram of returns
    plt.hist(returns * 100, bins=50, alpha=0.7, color='blue')
    
    # Add vertical lines for VaR and CVaR
    plt.axvline(var_95, color='red', linestyle='--', linewidth=2, label=f'VaR 95%: {var_95:.2f}%')
    plt.axvline(cvar_95, color='black', linestyle='--', linewidth=2, label=f'CVaR 95%: {cvar_95:.2f}%')
    
    plt.title(f'Distribution of Daily Returns for {symbol}', fontsize=16)
    plt.xlabel('Daily Return (%)', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATIONS_DIR, f'{symbol}_return_distribution.png'))
    plt.close()
    
    # Plot drawdown over time
    plt.figure(figsize=(14, 8))
    plt.plot(drawdown.index, drawdown, color='red', alpha=0.7)
    plt.fill_between(drawdown.index, drawdown, 0, color='red', alpha=0.3)
    plt.axhline(max_drawdown, color='black', linestyle='--', linewidth=2, label=f'Max Drawdown: {max_drawdown:.2f}%')
    
    plt.title(f'Drawdown Over Time for {symbol}', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Drawdown (%)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATIONS_DIR, f'{symbol}_drawdown.png'))
    plt.close()
    
    # Return risk metrics
    risk_metrics = {
        'Annualized Volatility (%)': volatility * 100,
        'Value at Risk 95% (%)': var_95,
        'Conditional VaR 95% (%)': cvar_95,
        'Maximum Drawdown (%)': max_drawdown
    }
    
    return risk_metrics

def main():
    print("Starting Predictive Modeling and Risk Analysis...")
    
    # Define the stocks to analyze
    symbols = ['NFLX', 'AAPL', 'MSFT', 'AMZN', 'GOOGL']
    
    # Store model performance metrics
    all_metrics = []
    all_risk_metrics = []
    
    # Process each stock
    for symbol in symbols:
        print(f"\nProcessing {symbol}...")
        
        # Get stock data with technical indicators
        df = get_stock_data_with_indicators(symbol)
        print(f"Data loaded: {len(df)} rows")
        
        # Prepare data for time series forecasting
        print("Preparing time series data...")
        X_train, y_train, X_test, y_test, train_data, test_data, scaler, target_scaler, features = prepare_time_series_data(df)
        
        # Build and train traditional ML models
        print("Building traditional ML models...")
        model_results = build_traditional_models(train_data, test_data, features, symbol)
        
        # Evaluate traditional models
        print("Evaluating traditional models...")
        metrics_df = evaluate_models(model_results, test_data, symbol)
        metrics_df['Symbol'] = symbol
        all_metrics.append(metrics_df)
        
        # Build and train LSTM model
        print("Building and training LSTM model...")
        lstm_model = build_lstm_model(X_train, y_train, X_test, y_test, symbol)
        
        # Evaluate LSTM model
        print("Evaluating LSTM model...")
        lstm_results = evaluate_lstm_model(lstm_model, X_test, y_test, test_data, target_scaler, symbol)
        
        # Add LSTM results to metrics
        lstm_metrics = pd.DataFrame({
            'Model': ['LSTM'],
            'Symbol': [symbol],
            'Train RMSE': [np.nan],  # We don't calculate this for LSTM in this implementation
            'Test RMSE': [lstm_results['rmse']],
            'Train MAE': [np.nan],
            'Test MAE': [lstm_results['mae']],
            'Train R²': [np.nan],
            'Test R²': [np.nan]
        })
        all_metrics.append(lstm_metrics)
        
        # Perform risk analysis
        print("Performing risk analysis...")
        risk_metrics = perform_risk_analysis(df, symbol)
        risk_df = pd.DataFrame({
            'Symbol': [symbol],
            'Annualized Volatility (%)': [risk_metrics['Annualized Volatility (%)']],
            'Value at Risk 95% (%)': [risk_metrics['Value at Risk 95% (%)']],
            'Conditional VaR 95% (%)': [risk_metrics['Conditional VaR 95% (%)']],
            'Maximum Drawdown (%)': [risk_metrics['Maximum Drawdown (%)']]
        })
        all_risk_metrics.append(risk_df)
    
    # Combine all metrics
    combined_metrics = pd.concat(all_metrics)
    combined_risk_metrics = pd.concat(all_risk_metrics)
    
    # Save metrics to CSV
    combined_metrics.to_csv(os.path.join(MODELS_DIR, 'model_performance_metrics.csv'), index=False)
    combined_risk_metrics.to_csv(os.path.join(MODELS_DIR, 'risk_metrics.csv'), index=False)
    
    # Create summary report
    summary_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'docs', 'modeling_summary.md')
    
    with open(summary_path, 'w') as f:
        f.write("# Stock Market Analysis - Predictive Modeling Summary\n\n")
        
        f.write("## Model Performance Metrics\n\n")
        f.write("### Traditional Machine Learning Models\n\n")
        f.write(combined_metrics[combined_metrics['Model'] != 'LSTM'].to_markdown(index=False))
        f.write("\n\n")
        
        f.write("### LSTM Models\n\n")
        f.write(combined_metrics[combined_metrics['Model'] == 'LSTM'].to_markdown(index=False))
        f.write("\n\n")
        
        f.write("## Risk Analysis\n\n")
        f.write(combined_risk_metrics.to_markdown(index=False))
        f.write("\n\n")
        
        f.write("## Key Findings\n\n")
        
        # Identify best performing model overall
        best_model_idx = combined_metrics[combined_metrics['Model'] != 'LSTM']['Test RMSE'].idxmin()
        best_model = combined_metrics.iloc[best_model_idx]
        
        f.write(f"1. **Best Traditional Model**: {best_model['Model']} for {best_model['Symbol']} with Test RMSE of {best_model['Test RMSE']:.2f}\n")
        
        # Compare LSTM performance
        lstm_metrics = combined_metrics[combined_metrics['Model'] == 'LSTM']
        best_lstm_idx = lstm_metrics['Test RMSE'].idxmin()
        best_lstm = lstm_metrics.iloc[best_lstm_idx]
        
        f.write(f"2. **Best LSTM Model**: LSTM for {best_lstm['Symbol']} with Test RMSE of {best_lstm['Test RMSE']:.2f}\n")
        
        # Identify stock with highest risk
        highest_risk_idx = combined_risk_metrics['Annualized Volatility (%)'].idxmax()
        highest_risk = combined_risk_metrics.iloc[highest_risk_idx]
        
        f.write(f"3. **Highest Risk Stock**: {highest_risk['Symbol']} with Annualized Volatility of {highest_risk['Annualized Volatility (%)']:.2f}% and Maximum Drawdown of {highest_risk['Maximum Drawdown (%)']:.2f}%\n")
        
        # Identify stock with lowest risk
        lowest_risk_idx = combined_risk_metrics['Annualized Volatility (%)'].idxmin()
        lowest_risk = combined_risk_metrics.iloc[lowest_risk_idx]
        
        f.write(f"4. **Lowest Risk Stock**: {lowest_risk['Symbol']} with Annualized Volatility of {lowest_risk['Annualized Volatility (%)']:.2f}% and Maximum Drawdown of {lowest_risk['Maximum Drawdown (%)']:.2f}%\n")
        
        f.write("\n## Next Steps\n\n")
        f.write("1. **Model Ensemble**: Combine predictions from multiple models to improve accuracy\n")
        f.write("2. **Hyperparameter Tuning**: Fine-tune model parameters for better performance\n")
        f.write("3. **Feature Importance Analysis**: Identify the most influential features for stock price prediction\n")
        f.write("4. **Portfolio Optimization**: Use risk metrics to create an optimal portfolio allocation\n")
        f.write("5. **Interactive Dashboard**: Develop a Power BI dashboard to visualize predictions and risk metrics\n")
    
    print(f"\nModeling summary saved to {summary_path}")
    print(f"Model performance metrics saved to {os.path.join(MODELS_DIR, 'model_performance_metrics.csv')}")
    print(f"Risk metrics saved to {os.path.join(MODELS_DIR, 'risk_metrics.csv')}")
    print("\nPredictive Modeling and Risk Analysis completed successfully!")

if __name__ == "__main__":
    main()
