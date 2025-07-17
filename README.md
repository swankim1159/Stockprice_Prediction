
 # Stockprice_Prediction
 
 This repository contains a multi phase experiment for analysing Samsung Electronics stock prices using ARIMA models. The project is divided into a number of steps:
 
 1. **Phase 1** – Basic ARIMA model (`Phaseone_ARIMA.py`).
 2. **Phase 2** – Integration of additional indicators such as PER, EPS and market cap (`Phasetwo_integrate_index.py`).
 3. **Phase 3** – Real time alert system (`Phasethree_notification_system.py`).
 4. **Phase 4** – Simple web dashboard (`Phasefour_webdashboard_dev`).
 
 This code is for research purposes only and does not constitute financial advice.

## Setup
Install the required Python packages using pip:

```bash
pip install pandas numpy matplotlib seaborn statsmodels scikit-learn requests flask twilio
```

After running the Phase 1 script, a `forecast.csv` file will be generated which is used by the simple dashboard in Phase 4.

# Caution
Because many finance APIs are blocked, Phase 1 loads historical Samsung prices from a public CSV hosted on GitHub (Emma243220056/005930.KS_daily_data.csv).

Samsung Electronics is listed on the Korean exchange with ticker **005930** and the preferred shares use ticker **005935**.
Fundamental indicators in Phase 2 are provided from a small static sample for demonstration purposes.
