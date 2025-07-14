Stock Price Prediction Model

A machine learning model for predicting stock prices, focusing on Samsung Electronics (005930.KS) and Samsung Electronics Preferred (005935.KS). 
This project automates data collection, ensures statistical rigor, and provides practical insights for investors.

*Warning; This project is NOT related to any stock recommendation or subsidiaries. 

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
