# Stock Market Analysis and Prediction Project - User Guide

## Introduction

This user guide provides instructions for using the Stock Market Analysis and Prediction Project. This project is designed to analyze historical stock data, perform exploratory data analysis, build predictive models, and conduct risk assessment for investment decision-making.

## Installation

### Prerequisites
- Python 3.8 or higher
- Pip package manager
- Git (optional)

### Setup Instructions

1. **Clone or download the repository**
   ```
   git clone <repository-url>
   ```
   or download and extract the ZIP file.

2. **Navigate to the project directory**
   ```
   cd stock_analysis_project
   ```

3. **Install required dependencies**
   ```
   pip install -r requirements.txt
   ```

## Project Components

### 1. Data Collection

The data collection module retrieves historical stock data from Yahoo Finance API and stores it in a SQLite database.

**To collect data:**
```
python scripts/collect_data.py
```

This will:
- Download 5 years of historical data for selected stocks
- Calculate technical indicators
- Store all data in the SQLite database

**Configuration options:**
- Edit the `symbols` list in `collect_data.py` to add or remove stocks
- Modify the date range by changing the `start_date` and `end_date` variables

### 2. Exploratory Data Analysis

The EDA module analyzes the collected stock data to identify patterns, trends, and relationships.

**To run the analysis:**
```
python scripts/exploratory_data_analysis.py
```

This will:
- Generate visualizations in the `visualizations/` directory
- Create a summary report in `docs/eda_summary.md`

**Key visualizations:**
- Price trends for all stocks
- Technical indicators (SMA, RSI, MACD, Bollinger Bands)
- Correlation heatmaps
- Sector performance comparisons
- Volume and volatility analysis
- Seasonal patterns

### 3. Predictive Modeling

The predictive modeling module builds and evaluates various machine learning models for stock price prediction.

**To train and evaluate models:**
```
python scripts/build_predictive_models.py
```

This will:
- Train multiple models (Linear Regression, Random Forest, Gradient Boosting, LSTM)
- Evaluate model performance
- Generate prediction visualizations
- Save trained models to the `models/` directory
- Create a summary report in `docs/modeling_summary.md`

**Model selection:**
- Edit the `symbols` list in `build_predictive_models.py` to focus on specific stocks
- Modify model parameters in the respective model building functions

### 4. Risk Analysis

The risk analysis module is integrated into the predictive modeling script and evaluates investment risk.

**Risk metrics generated:**
- Annualized volatility
- Value at Risk (VaR)
- Conditional VaR (CVaR)
- Maximum drawdown

**Visualizations:**
- Return distributions
- Drawdown over time

## Interpreting Results

### Exploratory Analysis Results

1. **Price Trend Analysis**
   - Look for long-term trends and patterns in the `all_stocks_closing_prices.png` and `normalized_stock_performance.png`
   - Compare relative performance across stocks in the normalized charts

2. **Technical Indicators**
   - Moving averages (SMA) indicate trend direction and potential support/resistance levels
   - RSI values above 70 suggest overbought conditions, below 30 suggest oversold
   - MACD crossovers can signal potential trend changes
   - Bollinger Bands help identify volatility and potential breakout points

3. **Correlation Analysis**
   - High correlation (close to 1.0) indicates stocks that move together
   - Low or negative correlation suggests diversification opportunities
   - Use `tech_stocks_correlation.png` to identify relationships

4. **Risk-Return Profiles**
   - Higher returns typically come with higher risk (volatility)
   - Identify optimal stocks in the `tech_stocks_risk_return.png` (high return, low risk)

### Predictive Model Results

1. **Model Performance Metrics**
   - Lower RMSE and MAE values indicate better prediction accuracy
   - Higher R² values indicate better fit to the data
   - Compare metrics across models in `model_performance_metrics.csv`

2. **Prediction Visualizations**
   - Compare actual vs. predicted prices in the `*_model_predictions.png` files
   - Assess how well each model captures trends and turning points

3. **Risk Analysis**
   - Lower volatility indicates more stable stocks
   - Less negative VaR and CVaR values indicate lower downside risk
   - Smaller maximum drawdown values indicate less severe price declines
   - Review metrics in `risk_metrics.csv`

## Customization

### Adding New Stocks

1. Edit the `symbols` list in `collect_data.py`
2. Run the data collection script
3. Run the EDA and modeling scripts to analyze the new stocks

### Modifying Time Periods

1. Change the `start_date` and `end_date` variables in `collect_data.py`
2. Run the data collection script with the new date range
3. Run the analysis scripts to process the updated data

### Adjusting Model Parameters

1. Locate the model building functions in `build_predictive_models.py`
2. Modify parameters such as:
   - `n_estimators` for Random Forest and Gradient Boosting
   - `units` and `dropout` for LSTM
   - `sequence_length` for time series preparation

## Troubleshooting

### Common Issues

1. **Missing dependencies**
   - Ensure all required packages are installed: `pip install -r requirements.txt`
   - For TensorFlow issues, try: `pip install tensorflow`

2. **Database errors**
   - Check if the database file exists in the `data/` directory
   - Run `setup_database.py` to recreate the database schema

3. **API rate limiting**
   - Yahoo Finance may limit requests; add delays between API calls
   - Consider using a different data source if persistent issues occur

4. **Memory errors during model training**
   - Reduce the number of stocks or date range
   - Decrease batch size for LSTM training
   - Use a machine with more RAM

## Conclusion

This Stock Market Analysis and Prediction Project provides a comprehensive toolkit for analyzing stock market data and making informed investment decisions. By following this user guide, you can leverage the power of data analysis and machine learning to gain insights into stock market behavior and potential future movements.

For technical details about the implementation, please refer to the Technical Documentation in the `docs/` directory.
