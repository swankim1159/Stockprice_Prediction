Stock Price Prediction Model

A machine learning model for predicting stock prices, focusing on Samsung Electronics (005930.KS) and Samsung Electronics Preferred (005935.KS). 
This project automates data collection, ensures statistical rigor, and provides practical insights for investors.

*Warning; This project is NOT related to any stock recommendation or subsidiaries. 

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

PHASE 1

🎯 Key Features

Automated Data Collection

Fetches data for Samsung Electronics (005930.KS) and Samsung Electronics Preferred (005935.KS) using the Yahoo Finance API.

Converts daily data to weekly data to reduce noise.

Utilizes 3 years of historical data for robust pattern learning.


Statistical Rigor

Performs stationarity testing using the Augmented Dickey-Fuller (ADF) test.

Automatically selects optimal ARIMA parameters based on the Akaike Information Criterion (AIC).

Provides prediction results with confidence intervals.


Practical Insights

Analyzes the correlation between trading volume and stock price.

Forecasts stock prices and expected returns for the next 12 weeks.

Offers actionable interpretations from an investment perspective.


💪 Practical Implementation Tips

Prerequisites

Install the required libraries using the following command:

"pip install pandas numpy matplotlib seaborn statsmodels scikit-learn yfinance beautifulsoup4 requests"


Usage

Run the code to automatically handle data collection, processing, and prediction.

Visualize results through intuitive graphs.

Review detailed investment insights printed in the output.

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

PHASE 2

🎯 Major Improvements

Financial Metrics Integration (ARIMAX Model)

Valuation Metrics: PER, PBR, EPS

Profitability Metrics: ROE, Operating Margin

Stability Metric: Debt-to-Equity Ratio

Scale/Growth Metrics: Market Capitalization, Revenue Growth Rate


Expanded Technical Indicators

Moving Averages (5, 20, 60 weeks)

Bollinger Bands, RSI, MACD

Volume Indicators: Volume Ratio, Volume-Price Trend

Price Volatility Indicators


Model Performance Enhancements

Utilizes exogenous variables in an ARIMAX model

Automated optimization of (p,d,q) parameters

Feature normalization using StandardScaler

Measures directional prediction accuracy


Comprehensive Analysis System

Simultaneous analysis of Samsung Electronics (005930.KS) and Samsung Electronics Preferred (005935.KS)

Performance comparison and future predictions

Visualization: Displays comprehensive analysis results in 4 charts


📈 Expected Outcomes
Compared to Phase 1, prediction accuracy is expected to improve by 15-25%. Specifically:

Short-Term Predictions (1-4 weeks): Directional accuracy of 65-75%

Mid-Term Predictions (1-3 months): Improved trend identification accuracy

Volatility Predictions: Enhanced risk management capabilities

