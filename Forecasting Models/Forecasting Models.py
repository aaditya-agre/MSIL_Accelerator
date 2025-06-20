import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings

warnings.filterwarnings("ignore")
sns.set(style="whitegrid")


def perform_time_series_eda(df, datetime_col, target_col, freq='D'):
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    df = df.sort_values(by=datetime_col)
    df.set_index(datetime_col, inplace=True)
    df = df.asfreq(freq)
    
    print("\n📅 Time Range:", df.index.min(), "→", df.index.max())
    print("📉 Missing values in target:", df[target_col].isnull().sum())

    plt.figure(figsize=(12, 4))
    plt.plot(df[target_col], label='Time Series')
    plt.title("Time Series Line Plot")
    plt.xlabel("Time")
    plt.ylabel(target_col)
    plt.legend()
    plt.show()

    window = 7
    plt.figure(figsize=(12, 4))
    plt.plot(df[target_col], label='Original')
    plt.plot(df[target_col].rolling(window).mean(), label='Rolling Mean')
    plt.plot(df[target_col].rolling(window).std(), label='Rolling Std')
    plt.legend()
    plt.title("Rolling Stats")
    plt.show()

    try:
        decomposition = seasonal_decompose(df[target_col].dropna(), model='additive', period=window)
        decomposition.plot()
        plt.suptitle("Seasonal Decomposition", fontsize=16)
        plt.show()
    except Exception as e:
        print("⚠ Decomposition failed:", e)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    plot_acf(df[target_col].dropna(), ax=axes[0])
    plot_pacf(df[target_col].dropna(), ax=axes[1])
    plt.tight_layout()
    plt.show()

    print("\n🧪 Augmented Dickey-Fuller (ADF) Test:")
    result = adfuller(df[target_col].dropna())
    print(f"ADF Statistic : {result[0]:.4f}")
    print(f"p-value       : {result[1]:.4f}")
    for key, val in result[4].items():
        print(f"   {key}: {val:.4f}")
    if result[1] < 0.05:
        print("✅ Stationary")
    else:
        print("❌ Non-Stationary")
    
    return df


def forecast_arima(df, target_col, steps=30):
    model = ARIMA(df[target_col].dropna(), order=(5,1,0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=steps)
    forecast.index = pd.date_range(start=df.index[-1] + pd.Timedelta(1, unit='D'), periods=steps)
    
    plt.figure(figsize=(12, 4))
    plt.plot(df[target_col], label='History')
    plt.plot(forecast, label='ARIMA Forecast')
    plt.title("ARIMA Forecast")
    plt.legend()
    plt.show()
    return forecast


def forecast_prophet(df, datetime_col, target_col, steps=30):
    prophet_df = df.reset_index()[[datetime_col, target_col]].rename(columns={datetime_col: 'ds', target_col: 'y'})
    model = Prophet()
    model.fit(prophet_df)

    future = model.make_future_dataframe(periods=steps)
    forecast = model.predict(future)

    model.plot(forecast)
    plt.title("Prophet Forecast")
    plt.show()
    return forecast[['ds', 'yhat']].tail(steps)


def forecast_linear_regression(df, target_col, steps=30):
    df_lr = df.copy().dropna()
    df_lr['time'] = np.arange(len(df_lr))

    X = df_lr[['time']]
    y = df_lr[target_col]

    model = LinearRegression()
    model.fit(X, y)

    future_time = np.arange(len(df_lr), len(df_lr) + steps).reshape(-1, 1)
    future_dates = pd.date_range(start=df_lr.index[-1] + pd.Timedelta(1, unit='D'), periods=steps)
    preds = model.predict(future_time)

    plt.figure(figsize=(12, 4))
    plt.plot(df_lr.index, df_lr[target_col], label='History')
    plt.plot(future_dates, preds, label='Linear Forecast')
    plt.title("Linear Regression Forecast")
    plt.legend()
    plt.show()

    return pd.DataFrame({'date': future_dates, 'forecast': preds})


### MAIN FLOW

df = pd.read_excel("/content/sample_timeseries.xlsx")  # change path if needed
datetime_col = 'date'
target_col = 'sales'

df = perform_time_series_eda(df, datetime_col, target_col)

# Choose model to forecast
forecast_arima(df, target_col)
forecast_prophet(df, datetime_col, target_col)
forecast_linear_regression(df, target_col)