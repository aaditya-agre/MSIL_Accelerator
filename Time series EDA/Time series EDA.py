import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

df = pd.read_excel("/content/sample_timeseries.xlsx")  # Replace with your file
datetime_col = 'date'  # Replace with your actual datetime column
target_col = 'sales'   # Replace with your actual target column



def perform_time_series_eda(df, datetime_col, target_col, freq='D'):
    # Step 1: Parse and set datetime index
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    df = df.sort_values(by=datetime_col)
    df.set_index(datetime_col, inplace=True)
    
    # Step 2: Reindex to ensure time continuity
    df = df.asfreq(freq)
    
    print("\n📅 Time Range:", df.index.min(), "→", df.index.max())
    print("📉 Missing values in target:", df[target_col].isnull().sum())
    
    # Step 3: Line Plot
    plt.figure(figsize=(12, 4))
    plt.plot(df[target_col], label='Time Series')
    plt.title("Time Series Line Plot")
    plt.xlabel("Time")
    plt.ylabel(target_col)
    plt.legend()
    plt.show()
    
    # Step 4: Rolling Statistics
    window = 7
    plt.figure(figsize=(12, 4))
    plt.plot(df[target_col], label='Original')
    plt.plot(df[target_col].rolling(window).mean(), label=f'{window}-Day Rolling Mean')
    plt.plot(df[target_col].rolling(window).std(), label=f'{window}-Day Rolling Std')
    plt.title("Rolling Mean & Std Dev")
    plt.legend()
    plt.show()
    
    # Step 5: Seasonal Decomposition
    try:
        decomposition = seasonal_decompose(df[target_col].dropna(), model='additive', period=window)
        decomposition.plot()
        plt.suptitle("Seasonal Decomposition", fontsize=16)
        plt.show()
    except Exception as e:
        print("⚠ Decomposition failed:", e)
    
    # Step 6: ACF and PACF
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    plot_acf(df[target_col].dropna(), ax=axes[0])
    plot_pacf(df[target_col].dropna(), ax=axes[1])
    axes[0].set_title("Autocorrelation (ACF)")
    axes[1].set_title("Partial Autocorrelation (PACF)")
    plt.tight_layout()
    plt.show()
    
    # Step 7: Augmented Dickey-Fuller Test (Stationarity)
    print("\n🧪 Augmented Dickey-Fuller (ADF) Test:")
    result = adfuller(df[target_col].dropna())
    print(f"ADF Statistic : {result[0]:.4f}")
    print(f"p-value       : {result[1]:.4f}")
    print("Critical Values:")
    for key, value in result[4].items():
        print(f"   {key}: {value:.4f}")
    if result[1] < 0.05:
        print("✅ Likely stationary (p < 0.05)")
    else:
        print("❌ Likely non-stationary (p ≥ 0.05)")
    
    return df

df = perform_time_series_eda(df, datetime_col, target_col, freq='D')  # Daily frequency
