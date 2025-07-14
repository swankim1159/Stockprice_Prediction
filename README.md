diff --git a/README.md b/README.md
index 13dfa5a00062788f8d5c44f0d732dc12752b5e2f..3ac1e08065e9d6e8f31134480f678a9fce9861e0 100644
--- a/README.md
+++ b/README.md
@@ -1,10 +1,18 @@
 # Stockprice_Prediction
 
 This repository contains a multi phase experiment for analysing Samsung Electronics stock prices using ARIMA models. The project is divided into a number of steps:
 
 1. **Phase 1** – Basic ARIMA model (`Phaseone_ARIMA`).
 2. **Phase 2** – Integration of additional indicators such as PER, EPS and market cap (`Phasetwo_integrate_index`).
 3. **Phase 3** – Real time alert system (`Phasethree_realtime_alertsystem`).
 4. **Phase 4** – Simple web dashboard (`Phasefour_webdashboard_dev`).
 
 This code is for research purposes only and does not constitute financial advice.
+
+## Setup
+Install the required Python packages using pip:
+
+```bash
+pip install pandas numpy matplotlib seaborn statsmodels scikit-learn requests beautifulsoup4 yfinance flask
+```
+
