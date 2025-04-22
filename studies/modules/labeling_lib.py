import os
import random
import numpy as np
import pandas as pd
from numba import njit
from sklearn.cluster import KMeans
from scipy.signal import savgol_filter
from scipy.interpolate import UnivariateSpline
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

# Obtener precios
def get_prices(hyper_params) -> pd.DataFrame:
    history_file = os.path.join(hyper_params["history_path"], f"{hyper_params['symbol']}_{hyper_params['timeframe']}.csv")
    p = pd.read_csv(history_file, sep=r"\s+")
    pFixed = pd.DataFrame(columns=['time', 'close'])
    pFixed['time'] = p['<DATE>'] + ' ' + p['<TIME>']
    pFixed['time'] = pd.to_datetime(pFixed['time'], format='mixed')
    pFixed['close'] = p['<CLOSE>']
    pFixed.set_index('time', inplace=True)
    return pFixed.dropna()

# Ingeniería de características
@njit
def compute_features(close, periods_main, periods_meta, stats_main, stats_meta):
    n = len(close)
    total_features = (len(periods_main) * len(stats_main)) + (len(periods_meta) * len(stats_meta))
    features = np.full((n, total_features), np.nan)

    def std_manual(x):
        m = np.mean(x)
        return np.sqrt(np.sum((x - m) ** 2) / (x.size - 1))

    def skew_manual(x):
        m = np.mean(x)
        s = std_manual(x)
        return np.mean(((x - m) / s) ** 3) if s != 0 else 0.0

    def kurt_manual(x):
        m = np.mean(x)
        s = std_manual(x)
        return np.mean(((x - m) / s) ** 4) - 3 if s != 0 else 0.0
    
    def zscore_manual(x):
        m = np.mean(x)
        s = std_manual(x)
        return (x[0] - m) / s if s != 0 else 0.0
    
    def entropy_manual(x):
        bins = 10
        minv = np.min(x)
        maxv = np.max(x)
        width = (maxv - minv) / bins
        if width == 0:
            return 0.0
        hist = np.zeros(bins)
        for val in x:
            idx = int((val - minv) / width)
            if idx == bins:  # caso borde
                idx -= 1
            hist[idx] += 1
        total = x.size
        entropy = 0.0
        for i in range(bins):
            p = hist[i] / total
            if p > 0:
                entropy -= p * np.log(p)
        return entropy

    def slope_manual(x):
        n = x.size
        x_idx = np.ascontiguousarray(np.arange(n))
        x_mean = np.mean(x_idx)
        y_mean = np.mean(x)
        numerator = np.sum((x_idx - x_mean) * (x - y_mean))
        denominator = np.sum((x_idx - x_mean) ** 2)
        return numerator / denominator if denominator != 0 else 0.0

    def momentum_manual(x):
        return (x[0] / x[-1]) - 1.0 if x[-1] != 0 else 0.0
    
    def roc_manual(x):
        n = len(x)
        if n < 2:
            return 0.0
        return ((x[0] - x[-1]) / x[-1]) * 100 if x[-1] != 0 else 0.0
    
    def fractal_dimension_manual(x):
        # Método de box-counting simplificado
        x = np.ascontiguousarray(x)  # Asegurar array contiguo
        eps = np.std(x) / 4  # epsilon como fracción de std
        if eps == 0:  # Evitar división por cero
            return 1.0
        count = np.sum(np.abs(np.diff(x)) > eps)
        if count == 0:
            return 1.0
        return 1.0 + np.log(count) / np.log(len(x))
    
    def hurst_manual(x):
        # Exponente de Hurst simplificado
        n = len(x)
        if n < 4:
            return 0.5
        lags = min(n-1, 20)  # usar máximo 20 lags
        tau = np.ascontiguousarray(np.arange(1, lags+1))
        rs = np.zeros(lags)
        
        for lag in range(1, lags+1):
            roll_mean = np.mean(x[:n-lag+1])
            roll_std = std_manual(x[:n-lag+1])
            if roll_std == 0:
                continue
            rs[lag-1] = np.max(x[:n-lag+1]) / roll_std
        
        valid_rs = rs[rs != 0]
        if len(valid_rs) < 2:
            return 0.5
        
        return np.mean(np.log(valid_rs)) / np.log(n) if n > 1 else 0.5

    # Procesar períodos normales
    col = 0
    for win in periods_main:
        for s in stats_main:
            for i in range(win, n):
                window = close[i - win:i][::-1]
                if s == "std":
                    features[i, col] = std_manual(window)
                elif s == "skew":
                    features[i, col] = skew_manual(window)
                elif s == "kurt":
                    features[i, col] = kurt_manual(window)
                elif s == "zscore":
                    features[i, col] = zscore_manual(window)
                elif s == "mean":
                    features[i, col] = np.mean(window)
                elif s == "range":
                    features[i, col] = np.max(window) - np.min(window)
                elif s == "median":
                    features[i, col] = np.median(window)
                elif s == "mad":
                    features[i, col] = np.mean(np.abs(window - np.mean(window)))
                elif s == "var":
                    features[i, col] = np.var(window)
                elif s == "entropy":
                    features[i, col] = entropy_manual(window)
                elif s == "slope":
                    features[i, col] = slope_manual(window)
                elif s == "momentum":
                    features[i, col] = momentum_manual(window)
                elif s == "roc":
                    features[i, col] = roc_manual(window)
                elif s == "fractal":
                    features[i, col] = fractal_dimension_manual(window)
                elif s == "hurst":
                    features[i, col] = hurst_manual(window)
            col += 1  # Incrementar col después de procesar todas las filas para esta estadística y ventana

    # Procesar períodos meta
    for win in periods_meta:
        for s in stats_meta:
            for i in range(win, n):
                window = close[i - win:i][::-1]
                if s == "std":
                    features[i, col] = std_manual(window)
                elif s == "skew":
                    features[i, col] = skew_manual(window)
                elif s == "kurt":
                    features[i, col] = kurt_manual(window)
                elif s == "zscore":
                    features[i, col] = zscore_manual(window)
                elif s == "mean":
                    features[i, col] = np.mean(window)
                elif s == "range":
                    features[i, col] = np.max(window) - np.min(window)
                elif s == "median":
                    features[i, col] = np.median(window)
                elif s == "mad":
                    features[i, col] = np.mean(np.abs(window - np.mean(window)))
                elif s == "var":
                    features[i, col] = np.var(window)
                elif s == "entropy":
                    features[i, col] = entropy_manual(window)
                elif s == "slope":
                    features[i, col] = slope_manual(window)
                elif s == "momentum":
                    features[i, col] = momentum_manual(window)
                elif s == "roc":
                    features[i, col] = roc_manual(window)
                elif s == "fractal":
                    features[i, col] = fractal_dimension_manual(window)
                elif s == "hurst":
                    features[i, col] = hurst_manual(window)
            col += 1  # Incrementar col después de procesar todas las filas para esta estadística meta

    return features

# Ingeniería de características
def get_clustering_features(data: pd.DataFrame, hp):
    close = data['close'].values
    index = data.index
    periods_main = hp["periods_main"]
    periods_meta = hp["periods_meta"]
    stats_main = hp["stats_main"]
    stats_meta = hp["stats_meta"]
    if len(stats_main) == 0:
        raise ValueError("La lista de estadísticas MAIN está vacía.")
    if len(stats_meta) == 0:
        raise ValueError("La lista de estadísticas META está vacía.")
    # Asegurar que los arrays sean contiguos
    close = np.ascontiguousarray(close)
    periods_main = np.ascontiguousarray(periods_main)
    periods_meta = np.ascontiguousarray(periods_meta)
    feats = compute_features(close, periods_main, periods_meta, stats_main, stats_meta)
    if np.isnan(feats).all():
        return pd.DataFrame(index=index)
    # Nombres de columnas
    colnames = []
    for p in periods_main:
        for s in stats_main:
            colnames.extend([f"{p}_{s}_feature"])
    for p in periods_meta:
        for s in stats_meta:
            colnames.extend([f"{p}_{s}_meta_feature"])
    df = pd.DataFrame(feats, columns=colnames, index=index)
    df["close"] = data["close"]
    return df.dropna()

# TREND OR NEUTRAL BASED LABELING
@njit
def calculate_labels(close_data, markup, min_val, max_val):
    labels = []
    for i in range(len(close_data) - max_val):
        rand = random.randint(min_val, max_val)
        curr_pr = close_data[i]
        future_pr = close_data[i + rand]

        if (future_pr + markup) < curr_pr:
            labels.append(1.0)
        elif (future_pr - markup) > curr_pr:
            labels.append(0.0)
        else:
            labels.append(2.0)
    return labels

def get_labels(dataset, markup, min = 1, max = 15) -> pd.DataFrame:
    """
    Generates labels for a financial dataset based on price movements.

    This function calculates labels indicating buy, sell, or hold signals 
    based on future price movements relative to a given markup percentage.

    Args:
        dataset (pd.DataFrame): DataFrame containing financial data with a 'close' column.
        markup (float): The percentage markup used to determine buy and sell signals.
        min (int, optional): Minimum number of consecutive days the markup must hold. Defaults to 1.
        max (int, optional): Maximum number of consecutive days the markup is considered. Defaults to 15.

    Returns:
        pd.DataFrame: The original DataFrame with a new 'labels' column and filtered rows:
                       - 'labels' column: 
                            - 0: Hold (price change doesn't meet criteria)
                            - 1: Buy (future price increases by at least 'markup' within 'max' days) 
                       - Rows where 'labels' is 2 (sell signal) are removed.
                       - Rows with missing values (NaN) are removed. 
    """

    # Extract closing prices from the dataset
    close_data = dataset['close'].values

    # Calculate buy/hold labels based on future price movements
    labels = calculate_labels(close_data, markup, min, max)

    # Trim the dataset to match the length of calculated labels
    dataset = dataset.iloc[:len(labels)].copy() 

    # Add the calculated labels as a new column
    dataset['labels'] = labels

    # Remove rows with NaN values (potentially introduced in 'calculate_labels')
    dataset = dataset.dropna()

    # Remove rows where the label is 2 (sell signal). 
    dataset = dataset.drop(dataset[dataset.labels == 2.0].index)

    return dataset

@njit
def calculate_labels_trend(normalized_trend, threshold):
    labels = np.empty(len(normalized_trend), dtype=np.float64)
    for i in range(len(normalized_trend)):
        if normalized_trend[i] > threshold:
            labels[i] = 0.0  # Buy (Up trend)
        elif normalized_trend[i] < -threshold:
            labels[i] = 1.0  # Sell (Down trend)
        else:
            labels[i] = 2.0  # No signal
    return labels

def get_labels_trend(dataset, rolling=200, polyorder=3, threshold=0.5, vol_window=50) -> pd.DataFrame:
    smoothed_prices = savgol_filter(dataset['close'].values, window_length=rolling, polyorder=polyorder)
    trend = np.gradient(smoothed_prices)
    vol = dataset['close'].rolling(vol_window).std().values
    normalized_trend = np.where(vol != 0, trend / vol, np.nan)  # Set NaN where vol is 0
    labels = calculate_labels_trend(normalized_trend, threshold)
    dataset = dataset.iloc[:len(labels)].copy()
    dataset['labels'] = labels
    dataset = dataset.dropna()  # Remove rows with NaN
    return dataset

def plot_trading_signals(
    dataset: pd.DataFrame,
    rolling: int = 200,
    polyorder: int = 3,
    threshold: float = 0.5,
    vol_window: int = 50,
    figsize: tuple = (14, 7)
) -> None:
    """
    Visualizes price data with calculated indicators and trading signals in one integrated plot.
    
    Args:
        dataset: DataFrame with 'close' prices and datetime index
        rolling: Window size for Savitzky-Golay filter. Default 200
        polyorder: Polynomial order for smoothing. Default 3
        threshold: Signal generation threshold. Default 0.5
        vol_window: Volatility calculation window. Default 50
        figsize: Figure dimensions. Default (14,7)
    """
    # Copy and clean data
    df = dataset[['close']].copy().dropna()
    close_prices = df['close'].values
    
    # 1. Smooth prices using Savitzky-Golay filter
    smoothed = savgol_filter(close_prices, window_length=rolling, polyorder=polyorder)
    
    # 2. Calculate trend gradient
    trend = np.gradient(smoothed)
    
    # 3. Compute volatility (rolling std)
    vol = df['close'].rolling(vol_window).std().values
    
    # 4. Normalize trend by volatility
    normalized_trend = np.zeros_like(trend)
    valid_mask = vol != 0  # Filter zero-volatility periods
    normalized_trend[valid_mask] = trend[valid_mask] / vol[valid_mask]
    
    # 5. Generate trading signals
    labels = np.full(len(normalized_trend), 2.0, dtype=np.float64)  # Default 2.0 (no signal)
    labels[normalized_trend > threshold] = 0.0  # Buy signals
    labels[normalized_trend < -threshold] = 1.0  # Sell signals
    
    # 6. Calculate threshold bands
    upper_band = smoothed + threshold * vol
    lower_band = smoothed - threshold * vol
    
    # Trim arrays to valid data (remove NaN/zero-vol periods)
    valid_idx = np.where(valid_mask)[0]
    df = df.iloc[valid_idx]
    smoothed = smoothed[valid_idx]
    upper_band = upper_band[valid_idx]
    lower_band = lower_band[valid_idx]
    labels = labels[valid_idx]
    
    # Create plot
    plt.figure(figsize=figsize)
    
    # Plot raw prices with semi-transparent line
    plt.plot(df.index, df['close'], 
             label='Actual Price', 
             color='#1f77b4', 
             alpha=0.4, 
             lw=1.2)
    
    # Plot smoothed trend line
    plt.plot(df.index, smoothed,
             label=f'Savitzky-Golay ({rolling},{polyorder})', 
             color='#ff7f0e', 
             lw=2)
    
    # Fill between threshold bands
    plt.fill_between(df.index, upper_band, lower_band,
                     color='#e0e0e0', 
                     alpha=0.3, 
                     label=f'Threshold ±{threshold}σ')
    
    # Plot buy/sell signals with distinct markers
    buy_dates = df.index[labels == 0.0]
    sell_dates = df.index[labels == 1.0]
    
    plt.scatter(buy_dates, df.loc[buy_dates, 'close'],
                color='#2ca02c', 
                marker='^', 
                s=80,
                edgecolor='black',
                label=f'Buy Signal (>+{threshold}σ)',
                zorder=5)
    
    plt.scatter(sell_dates, df.loc[sell_dates, 'close'],
                color='#d62728', 
                marker='v', 
                s=80,
                edgecolor='black',
                label=f'Sell Signal (<-{threshold}σ)',
                zorder=5)
    
    # Configure plot aesthetics
    plt.title(f'Trading Signals Generation\n(Smoothing: {rolling} periods, Threshold: {threshold}σ)')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.grid(alpha=0.2, linestyle='--')
    plt.legend(loc='upper left', framealpha=0.9)
    plt.tight_layout()
    plt.show()

@njit
def calculate_labels_trend_with_profit(close, normalized_trend, threshold, markup, min_l, max_l):
    labels = np.empty(len(normalized_trend) - max_l, dtype=np.float64)
    for i in range(len(normalized_trend) - max_l):
        if normalized_trend[i] > threshold:
            # Проверяем условие для Buy
            rand = random.randint(min_l, max_l)
            future_pr = close[i + rand]
            if future_pr >= close[i] + markup:
                labels[i] = 0.0  # Buy (Profit reached)
            else:
                labels[i] = 2.0  # No profit
        elif normalized_trend[i] < -threshold:
            # Проверяем условие для Sell
            rand = random.randint(min_l, max_l)
            future_pr = close[i + rand]
            if future_pr <= close[i] - markup:
                labels[i] = 1.0  # Sell (Profit reached)
            else:
                labels[i] = 2.0  # No profit
        else:
            labels[i] = 2.0  # No signal
    return labels

def get_labels_trend_with_profit(dataset, rolling=200, polyorder=3, threshold=0.5, 
                    vol_window=50, markup=0.5, min_l=1, max_l=15) -> pd.DataFrame:
    # Smoothing and trend calculation
    smoothed_prices = savgol_filter(dataset['close'].values, window_length=rolling, polyorder=polyorder)
    trend = np.gradient(smoothed_prices)
    
    # Normalizing the trend by volatility
    vol = dataset['close'].rolling(vol_window).std().values
    normalized_trend = np.where(vol != 0, trend / vol, np.nan)
    
    # Removing NaN and synchronizing data
    valid_mask = ~np.isnan(normalized_trend)
    normalized_trend_clean = normalized_trend[valid_mask]
    close_clean = dataset['close'].values[valid_mask]
    dataset_clean = dataset[valid_mask].copy()
    
    # Generating labels
    labels = calculate_labels_trend_with_profit(close_clean, normalized_trend_clean, threshold, markup, min_l, max_l)
    
    # Trimming the dataset and adding labels
    dataset_clean = dataset_clean.iloc[:len(labels)].copy()
    dataset_clean['labels'] = labels
    
    # Filtering the results
    dataset_clean = dataset_clean.dropna()    
    return dataset_clean

@njit
def calculate_labels_trend_different_filters(close, normalized_trend, threshold, markup, min_l, max_l):
    labels = np.empty(len(normalized_trend) - max_l, dtype=np.float64)
    for i in range(len(normalized_trend) - max_l):
        if normalized_trend[i] > threshold:
            # Проверяем условие для Buy
            rand = random.randint(min_l, max_l)
            future_pr = close[i + rand]
            if future_pr >= close[i] + markup:
                labels[i] = 0.0  # Buy (Profit reached)
            else:
                labels[i] = 2.0  # No profit
        elif normalized_trend[i] < -threshold:
            # Проверяем условие для Sell
            rand = random.randint(min_l, max_l)
            future_pr = close[i + rand]
            if future_pr <= close[i] - markup:
                labels[i] = 1.0  # Sell (Profit reached)
            else:
                labels[i] = 2.0  # No profit
        else:
            labels[i] = 2.0  # No signal
    return labels

def get_labels_trend_with_profit_different_filters(dataset, method='savgol', rolling=200, polyorder=3, threshold=0.5, 
                    vol_window=50, markup=0.5, min_l=1, max_l=15) -> pd.DataFrame:
    # Smoothing and trend calculation
    close_prices = dataset['close'].values
    if method == 'savgol':
        smoothed_prices = savgol_filter(close_prices, window_length=rolling, polyorder=polyorder)
    elif method == 'spline':
        x = np.arange(len(close_prices))
        spline = UnivariateSpline(x, close_prices, k=polyorder, s=rolling)
        smoothed_prices = spline(x)
    elif method == 'sma':
        smoothed_series = pd.Series(close_prices).rolling(window=rolling).mean()
        smoothed_prices = smoothed_series.values
    elif method == 'ema':
        smoothed_series = pd.Series(close_prices).ewm(span=rolling, adjust=False).mean()
        smoothed_prices = smoothed_series.values
    else:
        raise ValueError(f"Unknown smoothing method: {method}")
    
    trend = np.gradient(smoothed_prices)
    
    # Normalizing the trend by volatility
    vol = dataset['close'].rolling(vol_window).std().values
    normalized_trend = np.where(vol != 0, trend / vol, np.nan)
    
    # Removing NaN and synchronizing data
    valid_mask = ~np.isnan(normalized_trend)
    normalized_trend_clean = normalized_trend[valid_mask]
    close_clean = dataset['close'].values[valid_mask]
    dataset_clean = dataset[valid_mask].copy()
    
    # Generating labels
    labels = calculate_labels_trend_different_filters(close_clean, normalized_trend_clean, threshold, markup, min_l, max_l)
    
    # Trimming the dataset and adding labels
    dataset_clean = dataset_clean.iloc[:len(labels)].copy()
    dataset_clean['labels'] = labels
    
    # Filtering the results
    dataset_clean = dataset_clean.dropna()    
    return dataset_clean

@njit
def calculate_labels_trend_multi(close, normalized_trends, threshold, markup, min_l, max_l):
    num_periods = normalized_trends.shape[0]  # Number of periods
    labels = np.empty(len(close) - max_l, dtype=np.float64)
    for i in range(len(close) - max_l):
        # Select a random number of bars forward once for all periods
        rand = np.random.randint(min_l, max_l + 1)
        buy_signals = 0
        sell_signals = 0
        # Check conditions for each period
        for j in range(num_periods):
            if normalized_trends[j, i] > threshold:
                if close[i + rand] >= close[i] + markup:
                    buy_signals += 1
            elif normalized_trends[j, i] < -threshold:
                if close[i + rand] <= close[i] - markup:
                    sell_signals += 1
        # Combine signals
        if buy_signals > 0 and sell_signals == 0:
            labels[i] = 0.0  # Buy
        elif sell_signals > 0 and buy_signals == 0:
            labels[i] = 1.0  # Sell
        else:
            labels[i] = 2.0  # No signal or conflict
    return labels

def get_labels_trend_with_profit_multi(dataset, method='savgol', rolling_periods=[10, 20, 30], polyorder=3, threshold=0.5, 
                                       vol_window=50, markup=0.5, min_l=1, max_l=15) -> pd.DataFrame:
    """
    Generates labels for trading signals (Buy/Sell) based on the normalized trend,
    calculated for multiple smoothing periods.

    Args:
        dataset (pd.DataFrame): DataFrame with data, containing the 'close' column.
        method (str): Smoothing method ('savgol', 'spline', 'sma', 'ema').
        rolling_periods (list): List of smoothing window sizes. Default is [200].
        polyorder (int): Polynomial order for 'savgol' and 'spline' methods.
        threshold (float): Threshold for the normalized trend.
        vol_window (int): Window for volatility calculation.
        markup (float): Minimum profit to confirm the signal.
        min_l (int): Minimum number of bars forward.
        max_l (int): Maximum number of bars forward.

    Returns:
        pd.DataFrame: DataFrame with added 'labels' column:
                      - 0.0: Buy
                      - 1.0: Sell
                      - 2.0: No signal
    """
    close_prices = dataset['close'].values
    normalized_trends = []

    # Calculate normalized trend for each period
    for rolling in rolling_periods:
        if method == 'savgol':
            smoothed_prices = savgol_filter(close_prices, window_length=rolling, polyorder=polyorder)
        elif method == 'spline':
            x = np.arange(len(close_prices))
            spline = UnivariateSpline(x, close_prices, k=polyorder, s=rolling)
            smoothed_prices = spline(x)
        elif method == 'sma':
            smoothed_series = pd.Series(close_prices).rolling(window=rolling).mean()
            smoothed_prices = smoothed_series.values
        elif method == 'ema':
            smoothed_series = pd.Series(close_prices).ewm(span=rolling, adjust=False).mean()
            smoothed_prices = smoothed_series.values
        else:
            raise ValueError(f"Unknown smoothing method: {method}")
        
        trend = np.gradient(smoothed_prices)
        vol = pd.Series(close_prices).rolling(vol_window).std().values
        normalized_trend = np.where(vol != 0, trend / vol, np.nan)
        normalized_trends.append(normalized_trend)

    # Transform list into 2D array
    normalized_trends_array = np.vstack(normalized_trends)

    # Remove rows with NaN
    valid_mask = ~np.isnan(normalized_trends_array).any(axis=0)
    normalized_trends_clean = normalized_trends_array[:, valid_mask]
    close_clean = close_prices[valid_mask]
    dataset_clean = dataset[valid_mask].copy()

    # Generate labels
    labels = calculate_labels_trend_multi(close_clean, normalized_trends_clean, threshold, markup, min_l, max_l)

    # Trim data and add labels
    dataset_clean = dataset_clean.iloc[:len(labels)].copy()
    dataset_clean['labels'] = labels

    # Remove remaining NaN
    dataset_clean = dataset_clean.dropna()
    return dataset_clean

@njit
def calculate_labels_clusters(close_data, clusters, markup):
    labels = []
    current_cluster = clusters[0]
    last_price = close_data[0]
    for i in range(1, len(close_data)):
        next_cluster = clusters[i]
        if next_cluster != current_cluster and (abs(close_data[i] - last_price) > markup):
            if close_data[i] > last_price:
                labels.append(0.0)
            else:
                labels.append(1.0)
            current_cluster = next_cluster
            last_price = close_data[i]
        else:
            labels.append(2.0)

    if len(labels) < len(close_data):
        labels.append(2.0)
    return labels

def get_labels_clusters(dataset, markup, num_clusters=20) -> pd.DataFrame:
    kmeans = KMeans(n_clusters=num_clusters)
    dataset['cluster'] = kmeans.fit_predict(dataset[['close']])

    close_data = dataset['close'].values
    clusters = dataset['cluster'].values

    labels = calculate_labels_clusters(close_data, clusters, markup)

    dataset['labels'] = labels
    dataset = dataset.drop(dataset[dataset.labels == 2.0].index)
    dataset = dataset.drop(columns=['cluster'])
    return dataset

def sliding_window_clustering(dataset, n_clusters, window_size):
    meta_X = dataset.loc[:, dataset.columns.str.contains('meta_feature')]
    
    # Сначала создаем глобальные эталонные центроиды
    global_kmeans = KMeans(n_clusters=n_clusters).fit(meta_X)
    global_centroids = global_kmeans.cluster_centers_
    
    clusters = np.zeros(len(dataset))
    
    # Применяем кластеризацию в скользящем окне
    for i in range(window_size, len(dataset) - window_size + 1, window_size):
        window_data = meta_X.iloc[i:i+window_size]
        
        # Обучаем KMeans на текущем окне
        local_kmeans = KMeans(n_clusters=n_clusters).fit(window_data)
        local_centroids = local_kmeans.cluster_centers_
        
        # Сопоставляем локальные центроиды с глобальными
        # для обеспечения согласованности меток кластеров
        centroid_mapping = {}
        for local_idx in range(n_clusters):
            # Находим ближайший глобальный центроид к данному локальному
            distances = np.linalg.norm(local_centroids[local_idx] - global_centroids, axis=1)
            global_idx = np.argmin(distances)
            centroid_mapping[local_idx] = global_idx + 1  # +1 для начала нумерации с 1
        
        # Получаем метки для текущего окна
        local_labels = local_kmeans.predict(window_data)
        
        # Преобразуем локальные метки в согласованные глобальные метки
        for j in range(window_size):
            if i+j < len(clusters):  # Проверка на выход за границы
                clusters[i+j] = centroid_mapping[local_labels[j]]
    
    dataset = dataset.assign(clusters=clusters)
    return dataset

@njit
def calculate_signals(prices, window_sizes, threshold_pct):
    max_window = max(window_sizes)
    signals = []
    for i in range(max_window, len(prices)):
        long_signals = 0
        short_signals = 0
        for window_size in window_sizes:
            window = prices[i-window_size:i]
            resistance = max(window)
            support = min(window)
            current_price = prices[i]
            if current_price > resistance * (1 + threshold_pct):
                long_signals += 1
            elif current_price < support * (1 - threshold_pct):
                short_signals += 1
        if long_signals > short_signals:
            signals.append(0.0) 
        elif short_signals > long_signals:
            signals.append(1.0)
        else:
            signals.append(2.0)
    return signals

def get_labels_multi_window(dataset, window_sizes=[20, 50, 100], threshold_pct=0.02) -> pd.DataFrame:
    prices = dataset['close'].values
    signals = calculate_signals(prices, window_sizes, threshold_pct)
    signals = [2.0] * max(window_sizes) + signals
    dataset['labels'] = signals
    dataset = dataset.drop(dataset[dataset.labels == 2.0].index)
    return dataset

@njit
def calculate_labels_validated_levels(prices, window_size, threshold_pct, min_touches):
    resistance_touches = {}
    support_touches = {}
    labels = []
    for i in range(window_size, len(prices)):
        window = prices[i-window_size:i]
        current_price = prices[i]

        potential_resistance = np.max(window)
        potential_support = np.min(window)

        for level in resistance_touches:
            if abs(current_price - level) <= level * threshold_pct:
                resistance_touches[level] += 1

        for level in support_touches:
            if abs(current_price - level) <= level * threshold_pct:
                support_touches[level] += 1

        if potential_resistance not in resistance_touches:
            resistance_touches[potential_resistance] = 1
        if potential_support not in support_touches:
            support_touches[potential_support] = 1

        valid_resistance = [level for level, touches in resistance_touches.items() if touches >= min_touches]
        valid_support = [level for level, touches in support_touches.items() if touches >= min_touches]

        if valid_resistance and current_price > min(valid_resistance) * (1 + threshold_pct):
            labels.append(0.0)
        elif valid_support and current_price < max(valid_support) * (1 - threshold_pct):
            labels.append(1.0) 
        else:
            labels.append(2.0)

    return labels

def get_labels_validated_levels(dataset, window_size=20, threshold_pct=0.02, min_touches=2) -> pd.DataFrame:
    prices = dataset['close'].values
    
    labels = calculate_labels_validated_levels(prices, window_size, threshold_pct, min_touches)
    
    labels = [2.0] * window_size + labels
    dataset['labels'] = labels
    dataset = dataset.drop(dataset[dataset.labels == 2.0].index)
    return dataset

@njit
def calculate_labels_zigzag(peaks, troughs, len_close):
    """
    Generates labels based on the occurrence of peaks and troughs in the data.

    Args:
        peaks (np.array): Indices of the peaks in the data.
        troughs (np.array): Indices of the troughs in the data.
        len_close (int): The length of the close prices.

    Returns:
        np.array: An array of labels.
    """
    labels = np.empty(len_close, dtype=np.float64)
    labels.fill(2.0)  # Initialize all labels to 2.0 (no signal)
    
    last_peak_type = None  # None for the start, 'up' for peak, 'down' for trough
    
    for i in range(len_close):
        if i in peaks:
            labels[i] = 1.0  # Sell signal at peaks
            last_peak_type = 'up'
        elif i in troughs:
            labels[i] = 0.0  # Buy signal at troughs
            last_peak_type = 'down'
        else:
            # If we already have a peak established, use it for labeling
            if last_peak_type == 'up':
                labels[i] = 1.0
            elif last_peak_type == 'down':
                labels[i] = 0.0 
    return labels

def get_labels_filter_ZZ(dataset, peak_prominence=0.1) -> pd.DataFrame:
    """
    Generates labels for a financial dataset based on zigzag peaks and troughs.

    This function identifies peaks and troughs in the closing prices using the 'find_peaks' 
    function from scipy.signal. It generates buy signals at troughs and sell signals at peaks.

    Args:
        dataset (pd.DataFrame): DataFrame containing financial data with a 'close' column.
        peak_prominence (float, optional): Minimum prominence of peaks and troughs, 
                                           used to filter out insignificant fluctuations. 
                                           Defaults to 0.1.

    Returns:
        pd.DataFrame: The original DataFrame with a new 'labels' column and filtered rows:
                       - 'labels' column: 
                            - 0: Buy
                            - 1: Sell
                       - Rows where 'labels' is 2 (no signal) are removed.
                       - Rows with missing values (NaN) are removed.
    """

    # Find peaks and troughs in the closing prices
    peaks, _ = find_peaks(dataset['close'], prominence=peak_prominence)
    troughs, _ = find_peaks(-dataset['close'], prominence=peak_prominence)
    
    # Calculate buy/sell labels using the new zigzag-based labeling function
    labels = calculate_labels_zigzag(peaks, troughs, len(dataset)) 

    # Add the calculated labels as a new 'labels' column to the DataFrame
    dataset['labels'] = labels

    # Remove rows where the 'labels' column has a value of 2.0 (no signal)
    dataset = dataset.drop(dataset[dataset.labels == 2.0].index)
    
    # Return the modified DataFrame 
    return dataset

# MEAN REVERSION WITH RESTRICTIONS BASED LABELING
@njit
def calculate_labels_mean_reversion(close, lvl, markup, min_l, max_l, q):
    labels = np.empty(len(close) - max_l, dtype=np.float64)
    for i in range(len(close) - max_l):
        rand = random.randint(min_l, max_l)
        curr_pr = close[i]
        curr_lvl = lvl[i]
        future_pr = close[i + rand]

        if curr_lvl > q[1] and (future_pr + markup) < curr_pr:
            labels[i] = 1.0
        elif curr_lvl < q[0] and (future_pr - markup) > curr_pr:
            labels[i] = 0.0
        else:
            labels[i] = 2.0
    return labels

def get_labels_mean_reversion(dataset, markup, min_l=1, max_l=15, rolling=0.5, quantiles=[.45, .55], method='spline', decay_factor=0.95, shift=0) -> pd.DataFrame:
    """
    Generates labels for a financial dataset based on mean reversion principles.

    This function calculates trading signals (buy/sell) based on the deviation of
    the price from a chosen moving average or smoothing method. It identifies
    potential buy opportunities when the price deviates significantly below its 
    smoothed trend, anticipating a reversion to the mean.

    Args:
        dataset (pd.DataFrame): DataFrame containing financial data with a 'close' column.
        markup (float): The percentage markup used to determine buy signals.
        min_l (int, optional): Minimum number of consecutive days the markup must hold. Defaults to 1.
        max_l (int, optional): Maximum number of consecutive days the markup is considered. Defaults to 15.
        rolling (float, optional): Rolling window size for smoothing/averaging. 
                                     If method='spline', this controls the spline smoothing factor.
                                     Defaults to 0.5.
        quantiles (list, optional): Quantiles to define the "reversion zone". Defaults to [.45, .55].
        method (str, optional): Method for calculating the price deviation:
                                 - 'mean': Deviation from the rolling mean.
                                 - 'spline': Deviation from a smoothed spline.
                                 - 'savgol': Deviation from a Savitzky-Golay filter.
                                 Defaults to 'spline'.
        shift (int, optional): Shift the smoothed price data forward (positive) or backward (negative).
                                 Useful for creating a lag/lead effect. Defaults to 0.

    Returns:
        pd.DataFrame: The original DataFrame with a new 'labels' column and filtered rows:
                       - 'labels' column: 
                            - 0: Buy
                            - 1: Sell
                       - Rows where 'labels' is 2 (no signal) are removed.
                       - Rows with missing values (NaN) are removed.
                       - The temporary 'lvl' column is removed. 
    """

    # Calculate the price deviation ('lvl') based on the chosen method
    if method == 'mean':
        diff = (dataset['close'] - dataset['close'].rolling(rolling).mean())
        weighted_diff = diff * np.exp(np.arange(len(diff)) * decay_factor / len(diff)) 
        dataset['lvl'] = weighted_diff # Add the weighted difference as 'lvl'
    elif method == 'spline':
        x = np.array(range(dataset.shape[0]))
        y = dataset['close'].values
        spl = UnivariateSpline(x, y, k=3, s=rolling) 
        yHat = spl(np.linspace(min(x), max(x), num=x.shape[0]))
        yHat_shifted = np.roll(yHat, shift=shift) # Apply the shift
        diff = dataset['close'] - yHat_shifted
        weighted_diff = diff * np.exp(np.arange(len(diff)) * decay_factor / len(diff)) 
        dataset['lvl'] = weighted_diff # Add the weighted difference as 'lvl'
        dataset = dataset.dropna()  # Remove NaN values potentially introduced by spline/shift
    elif method == 'savgol':
        smoothed_prices = savgol_filter(dataset['close'].values, window_length=int(rolling), polyorder=3)
        diff = dataset['close'] - smoothed_prices
        weighted_diff = diff * np.exp(np.arange(len(diff)) * decay_factor / len(diff)) 
        dataset['lvl'] = weighted_diff # Add the weighted difference as 'lvl'

    dataset = dataset.dropna()  # Remove NaN values before proceeding
    q = dataset['lvl'].quantile(quantiles).to_list()  # Calculate quantiles for the 'reversion zone'

    # Prepare data for label calculation
    close = dataset['close'].values
    lvl = dataset['lvl'].values
    
    # Calculate buy/sell labels 
    labels = calculate_labels_mean_reversion(close, lvl, markup, min_l, max_l, q) 

    # Process the dataset and labels
    dataset = dataset.iloc[:len(labels)].copy()
    dataset['labels'] = labels
    dataset = dataset.dropna()
    dataset = dataset.drop(dataset[dataset.labels == 2.0].index)  # Remove sell signals (if any)
    return dataset.drop(columns=['lvl'])  # Remove the temporary 'lvl' column 

@njit
def calculate_labels_mean_reversion_multi(close_data, lvl_data, q, markup, min_l, max_l, windows):
    labels = []
    for i in range(len(close_data) - max_l):
        rand = random.randint(min_l, max_l)
        curr_pr = close_data[i]
        future_pr = close_data[i + rand]

        buy_condition = True
        sell_condition = True
        qq = 0
        for _ in windows:  # Loop over each window, variable unused
            curr_lvl = lvl_data[i, qq]            
            if not (curr_lvl >= q[qq, 1]):  # Access q as 2D array
                sell_condition = False
            if not (curr_lvl <= q[qq, 0]):
                buy_condition = False
            qq += 1
    
        if sell_condition and (future_pr + markup) < curr_pr:
            labels.append(1.0)
        elif buy_condition and (future_pr - markup) > curr_pr:
            labels.append(0.0)
        else:
            labels.append(2.0)
    return labels

def get_labels_mean_reversion_multi(dataset, markup, min_l=1, max_l=15, windows=[0.2, 0.3, 0.5], quantiles=[.45, .55]) -> pd.DataFrame:
    q = np.empty((len(windows), 2))  # Initialize as 2D NumPy array
    lvl_data = np.empty((dataset.shape[0], len(windows)))

    for i, rolling in enumerate(windows):
        x = np.arange(dataset.shape[0])
        y = dataset['close'].values
        spl = UnivariateSpline(x, y, k=3, s=rolling)
        yHat = spl(np.linspace(x.min(), x.max(), x.shape[0]))
        lvl_data[:, i] = dataset['close'] - yHat
        # Store quantiles directly into the NumPy array
        quantile_values = np.quantile(lvl_data[:, i], quantiles)
        q[i, 0] = quantile_values[0]
        q[i, 1] = quantile_values[1]

    dataset = dataset.dropna()
    close_data = dataset['close'].values

    # Convert windows to a tuple for Numba compatibility (optional)
    labels = calculate_labels_mean_reversion_multi(close_data, lvl_data, q, markup, min_l, max_l, tuple(windows))

    dataset = dataset.iloc[:len(labels)].copy()
    dataset['labels'] = labels
    dataset = dataset.dropna()
    dataset = dataset[dataset.labels != 2.0]
    
    return dataset

@njit
def calculate_labels_mean_reversion_v(close_data, lvl_data, volatility_group, quantile_groups_low, quantile_groups_high, markup, min_l, max_l):
    labels = []
    for i in range(len(close_data) - max_l):
        rand = random.randint(min_l, max_l)
        curr_pr = close_data[i]
        curr_lvl = lvl_data[i]
        curr_vol_group = volatility_group[i]
        future_pr = close_data[i + rand]

        # Access quantiles directly from arrays
        low_q = quantile_groups_low[int(curr_vol_group)]
        high_q = quantile_groups_high[int(curr_vol_group)]

        if curr_lvl > high_q and (future_pr + markup) < curr_pr:
            labels.append(1.0)
        elif curr_lvl < low_q and (future_pr - markup) > curr_pr:
            labels.append(0.0)
        else:
            labels.append(2.0)
    return labels

def get_labels_mean_reversion_v(dataset, markup, min_l=1, max_l=15, rolling=0.5, quantiles=[.45, .55], method='spline', shift=1, volatility_window=20) -> pd.DataFrame:
    """
    Generates trading labels based on mean reversion principles, incorporating
    volatility-based adjustments to identify buy opportunities.

    This function calculates trading signals (buy/sell), taking into account the 
    volatility of the asset. It groups the data into volatility bands and calculates 
    quantiles for each band. This allows for more dynamic "reversion zones" that 
    adjust to changing market conditions.

    Args:
        dataset (pd.DataFrame): DataFrame containing financial data with a 'close' column.
        markup (float): The percentage markup used to determine buy signals.
        min_l (int, optional): Minimum number of consecutive days the markup must hold. Defaults to 1.
        max_l (int, optional): Maximum number of consecutive days the markup is considered. Defaults to 15.
        rolling (float, optional): Rolling window size or spline smoothing factor (see 'method'). 
                                     Defaults to 0.5.
        quantiles (list, optional): Quantiles to define the "reversion zone". Defaults to [.45, .55].
        method (str, optional): Method for calculating the price deviation:
                                 - 'mean': Deviation from the rolling mean.
                                 - 'spline': Deviation from a smoothed spline.
                                 - 'savgol': Deviation from a Savitzky-Golay filter.
                                 Defaults to 'spline'.
        shift (int, optional): Shift the smoothed price data (lag/lead effect). Defaults to 1.
        volatility_window (int, optional): Window size for calculating volatility. Defaults to 20.

    Returns:
        pd.DataFrame: The original DataFrame with a new 'labels' column and filtered rows:
                       - 'labels' column: 
                            - 0: Buy
                            - 1: Sell
                       - Rows where 'labels' is 2 (no signal) are removed.
                       - Rows with missing values (NaN) are removed.
                       - Temporary 'lvl', 'volatility', 'volatility_group' columns are removed.
    """

    # Calculate Volatility
    dataset['volatility'] = dataset['close'].pct_change().rolling(window=volatility_window).std()
    
    # Divide into 20 groups by volatility 
    dataset['volatility_group'] = pd.qcut(dataset['volatility'], q=20, labels=False)
    
    # Calculate price deviation ('lvl') based on the chosen method
    if method == 'mean':
        dataset['lvl'] = (dataset['close'] - dataset['close'].rolling(rolling).mean())
    elif method == 'spline':
        x = np.array(range(dataset.shape[0]))
        y = dataset['close'].values
        spl = UnivariateSpline(x, y, k=3, s=rolling)
        yHat = spl(np.linspace(min(x), max(x), num=x.shape[0]))
        yHat_shifted = np.roll(yHat, shift=shift) # Apply the shift 
        dataset['lvl'] = dataset['close'] - yHat_shifted
        dataset = dataset.dropna() 
    elif method == 'savgol':
        smoothed_prices = savgol_filter(dataset['close'].values, window_length=rolling, polyorder=5)
        dataset['lvl'] = dataset['close'] - smoothed_prices

    dataset = dataset.dropna()
    
    # Calculate quantiles for each volatility group
    quantile_groups = {}
    quantile_groups_low = []
    quantile_groups_high = []
    for group in range(20):
        group_data = dataset[dataset['volatility_group'] == group]['lvl']
        quantiles_values = group_data.quantile(quantiles).to_list()
        quantile_groups[group] = quantiles_values
        quantile_groups_low.append(quantiles_values[0])
        quantile_groups_high.append(quantiles_values[1])

    # Prepare data for label calculation (potentially using Numba)
    close_data = dataset['close'].values
    lvl_data = dataset['lvl'].values
    volatility_group = dataset['volatility_group'].values
    
    # Convert quantile groups to numpy arrays
    quantile_groups_low = np.array(quantile_groups_low)
    quantile_groups_high = np.array(quantile_groups_high)

    # Calculate buy/sell labels 
    labels = calculate_labels_mean_reversion_v(close_data, lvl_data, volatility_group, quantile_groups_low, quantile_groups_high, markup, min_l, max_l)
    
    # Process dataset and labels
    dataset = dataset.iloc[:len(labels)].copy()
    dataset['labels'] = labels
    dataset = dataset.dropna()
    dataset = dataset.drop(dataset[dataset.labels == 2.0].index) # Remove sell signals
    
    # Remove temporary columns and return
    return dataset.drop(columns=['lvl', 'volatility', 'volatility_group'])

# FILTERING BASED LABELING W/O RESTRICTIONS
@njit
def calculate_labels_filter(close, lvl, q):
    labels = np.empty(len(close), dtype=np.float64)
    for i in range(len(close)):
        curr_lvl = lvl[i]

        if curr_lvl > q[1]:
            labels[i] = 1.0
        elif curr_lvl < q[0]:
            labels[i] = 0.0
        else:
            labels[i] = 2.0
    return labels

def get_labels_filter(dataset, rolling=200, quantiles=[.45, .55], polyorder=3, decay_factor=0.95) -> pd.DataFrame:
    """
    Generates labels for a financial dataset based on price deviation from a Savitzky-Golay filter,
    with exponential weighting applied to prioritize recent data. Optionally incorporates a 
    cyclical component to the price deviation.

    Args:
        dataset (pd.DataFrame): DataFrame containing financial data with a 'close' column.
        rolling (int, optional): Window size for the Savitzky-Golay filter. Defaults to 200.
        quantiles (list, optional): Quantiles to define the "reversion zone". Defaults to [.45, .55].
        polyorder (int, optional): Polynomial order for the Savitzky-Golay filter. Defaults to 3.
        decay_factor (float, optional): Exponential decay factor for weighting past data. 
                                        Lower values prioritize recent data more. Defaults to 0.95.
        cycle_period (int, optional): Period of the cycle in number of data points. If None, 
                                     no cycle is applied. Defaults to None.
        cycle_amplitude (float, optional): Amplitude of the cycle. If None, no cycle is applied. 
                                          Defaults to None.

    Returns:
        pd.DataFrame: The original DataFrame with a new 'labels' column and filtered rows:
                       - 'labels' column: 
                            - 0: Buy
                            - 1: Sell
                       - Rows where 'labels' is 2 (no signal) are removed.
                       - Rows with missing values (NaN) are removed.
                       - The temporary 'lvl' column is removed. 
    """

    # Calculate smoothed prices using the Savitzky-Golay filter
    smoothed_prices = savgol_filter(dataset['close'].values, window_length=rolling, polyorder=polyorder)
    
    # Calculate the difference between the actual closing prices and the smoothed prices
    diff = dataset['close'] - smoothed_prices
    
    # Apply exponential weighting to the 'diff' values
    weighted_diff = diff * np.exp(np.arange(len(diff)) * decay_factor / len(diff)) 
    
    dataset['lvl'] = weighted_diff # Add the weighted difference as 'lvl'

    # Remove any rows with NaN values 
    dataset = dataset.dropna()
    
    # Calculate the quantiles of the 'lvl' column (price deviation)
    q = dataset['lvl'].quantile(quantiles).to_list() 

    # Extract the closing prices and the calculated 'lvl' values as NumPy arrays
    close = dataset['close'].values
    lvl = dataset['lvl'].values
    
    # Calculate buy/sell labels using the 'calculate_labels_filter' function 
    labels = calculate_labels_filter(close, lvl, q) 

    # Trim the dataset to match the length of the calculated labels
    dataset = dataset.iloc[:len(labels)].copy()
    
    # Add the calculated labels as a new 'labels' column to the DataFrame
    dataset['labels'] = labels
    
    # Remove any rows with NaN values
    dataset = dataset.dropna()
    
    # Return the modified DataFrame with the 'lvl' column removed
    return dataset.drop(columns=['lvl']) 

@njit
def calc_labels_multiple_filters(close, lvls, qs):
    labels = np.empty(len(close), dtype=np.float64)
    for i in range(len(close)):
        label_found = False
        
        for j in range(len(lvls)):
            curr_lvl = lvls[j][i]
            curr_q_low = qs[j][0][i]
            curr_q_high = qs[j][1][i]
            
            if curr_lvl > curr_q_high:
                labels[i] = 1.0
                label_found = True
                break
            elif curr_lvl < curr_q_low:
                labels[i] = 0.0
                label_found = True
                break
                
        if not label_found:
            labels[i] = 2.0
            
    return labels

def get_labels_multiple_filters(dataset, rolling_periods=[200, 400, 600], quantiles=[.45, .55], window=100, polyorder=3) -> pd.DataFrame:
    """
    Generates trading signals (buy/sell) based on price deviation from multiple 
    smoothed price trends calculated using a Savitzky-Golay filter with different
    rolling periods and rolling quantiles. 

    This function applies a Savitzky-Golay filter to the closing prices for each 
    specified 'rolling_period'. It then calculates the price deviation from these
    smoothed trends and determines dynamic "reversion zones" using rolling quantiles.
    Buy signals are generated when the price is within these reversion zones 
    across multiple timeframes.

    Args:
        dataset (pd.DataFrame): DataFrame containing financial data with a 'close' column.
        rolling_periods (list, optional): List of rolling window sizes for the Savitzky-Golay filter. 
                                           Defaults to [200, 400, 600].
        quantiles (list, optional): Quantiles to define the "reversion zone". Defaults to [.05, .95].
        window (int, optional): Window size for calculating rolling quantiles. Defaults to 100.
        polyorder (int, optional): Polynomial order for the Savitzky-Golay filter. Defaults to 3.

    Returns:
        pd.DataFrame: The original DataFrame with a new 'labels' column and filtered rows:
                       - 'labels' column: 
                            - 0: Buy
                            - 1: Sell
                       - Rows where 'labels' is 2 (no signal) are removed.
                       - Rows with missing values (NaN) are removed. 
    """
    
    # Create a copy of the dataset to avoid modifying the original
    dataset = dataset.copy()
    
    # Lists to store price deviation levels and quantiles for each rolling period
    all_levels = []
    all_quantiles = []
    
    # Calculate smoothed price trends and rolling quantiles for each rolling period
    for rolling in rolling_periods:
        # Calculate smoothed prices using the Savitzky-Golay filter
        smoothed_prices = savgol_filter(dataset['close'].values, 
                                      window_length=rolling, 
                                      polyorder=polyorder)
        # Calculate the price deviation from the smoothed prices
        diff = dataset['close'] - smoothed_prices
        
        # Create a temporary DataFrame to calculate rolling quantiles
        temp_df = pd.DataFrame({'diff': diff})
        
        # Calculate rolling quantiles for the price deviation
        q_low = temp_df['diff'].rolling(window=window).quantile(quantiles[0])
        q_high = temp_df['diff'].rolling(window=window).quantile(quantiles[1])
        
        # Store the price deviation and quantiles for the current rolling period
        all_levels.append(diff)
        all_quantiles.append([q_low.values, q_high.values])
    
    # Convert lists to NumPy arrays for faster calculations (potentially using Numba)
    lvls_array = np.array(all_levels)
    qs_array = np.array(all_quantiles)
    
    # Calculate buy/sell labels using the 'calc_labels_multiple_filters' function 
    labels = calc_labels_multiple_filters(dataset['close'].values, lvls_array, qs_array)
    
    # Add the calculated labels to the DataFrame
    dataset['labels'] = labels
    
    # Remove rows with NaN values and sell signals (labels == 2.0)
    dataset = dataset.dropna()
    dataset = dataset.drop(dataset[dataset.labels == 2.0].index)
    
    # Return the DataFrame with the new 'labels' column
    return dataset

@njit
def calc_labels_bidirectional(close, lvl1, lvl2, q1, q2):
    labels = np.empty(len(close), dtype=np.float64)
    for i in range(len(close)):
        curr_lvl1 = lvl1[i]
        curr_lvl2 = lvl2[i]

        if curr_lvl1 > q1[1]:
            labels[i] = 1.0
        elif curr_lvl2 < q2[0]:
            labels[i] = 0.0
        else:
            labels[i] = 2.0
    return labels

def get_labels_filter_bidirectional(dataset, rolling1=200, rolling2=200, quantiles=[.45, .55], polyorder=3) -> pd.DataFrame:
    """
    Generates trading labels based on price deviation from two Savitzky-Golay filters applied
    in opposite directions (forward and reversed) to the closing price data.

    This function calculates trading signals (buy/sell) based on the price's 
    position relative to smoothed price trends generated by two Savitzky-Golay filters 
    with potentially different window sizes (`rolling1`, `rolling2`). 

    Args:
        dataset (pd.DataFrame): DataFrame containing financial data with a 'close' column.
        rolling1 (int, optional): Window size for the first Savitzky-Golay filter. Defaults to 200.
        rolling2 (int, optional): Window size for the second Savitzky-Golay filter. Defaults to 200.
        quantiles (list, optional): Quantiles to define the "reversion zones". Defaults to [.45, .55].
        polyorder (int, optional): Polynomial order for both Savitzky-Golay filters. Defaults to 3.

    Returns:
        pd.DataFrame: The original DataFrame with a new 'labels' column and filtered rows:
                       - 'labels' column: 
                            - 0: Buy
                            - 1: Sell
                       - Rows where 'labels' is 2 (no signal) are removed.
                       - Rows with missing values (NaN) are removed.
                       - Temporary 'lvl1' and 'lvl2' columns are removed.
    """

    # Apply the first Savitzky-Golay filter (forward direction)
    smoothed_prices = savgol_filter(dataset['close'].values, window_length=rolling1, polyorder=polyorder)
    
    # Apply the second Savitzky-Golay filter (could be in reverse direction if rolling2 is negative)
    smoothed_prices2 = savgol_filter(dataset['close'].values, window_length=rolling2, polyorder=polyorder)

    # Calculate price deviations from both smoothed price series
    diff1 = dataset['close'] - smoothed_prices
    diff2 = dataset['close'] - smoothed_prices2

    # Add price deviations as new columns to the DataFrame
    dataset['lvl1'] = diff1
    dataset['lvl2'] = diff2
    
    # Remove rows with NaN values 
    dataset = dataset.dropna()

    # Calculate quantiles for the "reversion zones" for both price deviation series
    q1 = dataset['lvl1'].quantile(quantiles).to_list()
    q2 = dataset['lvl2'].quantile(quantiles).to_list()

    # Extract relevant data for label calculation
    close = dataset['close'].values
    lvl1 = dataset['lvl1'].values
    lvl2 = dataset['lvl2'].values
    
    # Calculate buy/sell labels using the 'calc_labels_bidirectional' function
    labels = calc_labels_bidirectional(close, lvl1, lvl2, q1, q2)

    # Process the dataset and labels
    dataset = dataset.iloc[:len(labels)].copy()
    dataset['labels'] = labels
    dataset = dataset.dropna()
    dataset = dataset.drop(dataset[dataset.labels == 2.0].index) # Remove bad signals (if any)
    
    # Return the DataFrame with temporary columns removed
    return dataset.drop(columns=['lvl1', 'lvl2']) 

# ONE DIRECTION LABELING
@njit
def calculate_labels_one_direction(close_data, markup, min_val, max_val, direction):
    labels = []
    for i in range(len(close_data) - max_val):
        rand = random.randint(min_val, max_val)
        curr_pr = close_data[i]
        future_pr = close_data[i + rand]
        
        if direction == "buy":
            # Label 1 si el precio sube MÁS que el markup (trade positivo)
            if (future_pr - markup) > curr_pr:
                labels.append(1.0)
            else:
                labels.append(0.0)
        elif direction == "sell":
            # Label 1 si el precio baja MÁS que el markup (trade positivo)
            if (future_pr + markup) < curr_pr:
                labels.append(1.0)
            else:
                labels.append(0.0)
    return labels

def get_labels_one_direction(dataset, markup, min = 1, max = 15, direction = 'buy') -> pd.DataFrame:
    close_data = dataset['close'].values
    labels = calculate_labels_one_direction(close_data, markup, min, max, direction)
    dataset = dataset.iloc[:len(labels)].copy()
    dataset['labels'] = labels
    dataset = dataset.dropna()
    return dataset

@njit
def calculate_labels_filter_one_direction(close, lvl, q, direction):
    labels = np.empty(len(close), dtype=np.float64)
    for i in range(len(close)):
        curr_lvl = lvl[i]

        if direction == "sell":
          if curr_lvl > q[1]:
              labels[i] = 1.0
          else:
              labels[i] = 0.0
        if direction == "buy":
          if curr_lvl < q[0]:
              labels[i] = 1.0
          else:
              labels[i] = 0.0
    return labels

def get_labels_filter_one_direction(dataset, rolling=200, quantiles=[.45, .55], polyorder=3, direction='buy') -> pd.DataFrame:
    """
    Generates labels for a financial dataset based on price deviation from a Savitzky-Golay filter.

    This function applies a Savitzky-Golay filter to the closing prices to generate a smoothed
    price trend. It then calculates trading signals (buy/sell) based on the deviation of the 
    actual price from this smoothed trend. Buy signals are generated when the price is 
    significantly below the smoothed trend, anticipating a potential price reversal. 

    Args:
        dataset (pd.DataFrame): DataFrame containing financial data with a 'close' column.
        rolling (int, optional): Window size for the Savitzky-Golay filter. Defaults to 200.
        quantiles (list, optional): Quantiles to define the "reversion zone". Defaults to [.45, .55].
        polyorder (int, optional): Polynomial order for the Savitzky-Golay filter. Defaults to 3.
        direction (str, optional): The direction to generate signals for. Either 'buy' or 'sell'. Defaults to 'buy'.

    Returns:
        pd.DataFrame: The original DataFrame with a new 'labels' column and filtered rows:
                       - 'labels' column: 
                            - 0: No trade
                            - 1: Trade (buy or sell depending on direction)
                       - Rows with missing values (NaN) are removed.
                       - The temporary 'lvl' column is removed. 
    """

    # Calculate smoothed prices using the Savitzky-Golay filter
    smoothed_prices = savgol_filter(dataset['close'].values, window_length=rolling, polyorder=polyorder)
    
    # Calculate the difference between the actual closing prices and the smoothed prices
    diff = dataset['close'] - smoothed_prices
    dataset['lvl'] = diff  # Add the difference as a new column 'lvl' to the DataFrame
    
    # Remove any rows with NaN values 
    dataset = dataset.dropna()
    
    # Calculate the quantiles of the 'lvl' column (price deviation)
    q = dataset['lvl'].quantile(quantiles).to_list() 

    # Extract the closing prices and the calculated 'lvl' values as NumPy arrays
    close = dataset['close'].values
    lvl = dataset['lvl'].values
    
    # Calculate buy/sell labels using the 'calculate_labels_filter' function 
    labels = calculate_labels_filter_one_direction(close, lvl, q, direction) 

    # Trim the dataset to match the length of the calculated labels
    dataset = dataset.iloc[:len(labels)].copy()
    
    # Add the calculated labels as a new 'labels' column to the DataFrame
    dataset['labels'] = labels
    
    # Remove any rows with NaN values
    dataset = dataset.dropna()
    
    # Return the modified DataFrame with the 'lvl' column removed
    return dataset.drop(columns=['lvl']) 

@njit
def calculate_labels_trend_one_direction(normalized_trend, threshold, direction):
    labels = np.empty(len(normalized_trend), dtype=np.float64)
    for i in range(len(normalized_trend)):
        if direction == 'buy':
            if normalized_trend[i] > threshold:
                labels[i] = 1.0  # Buy (Up trend)
            else:
                labels[i] = 0.0
        if direction == 'sell':
            if normalized_trend[i] < -threshold:
                labels[i] = 1.0  # Sell (Down trend)
            else:
                labels[i] = 0.0  # No signal
    return labels

def get_labels_trend_one_direction(dataset, rolling=50, polyorder=3, threshold=0.001, vol_window=50, direction='buy') -> pd.DataFrame:
    smoothed_prices = savgol_filter(dataset['close'].values, window_length=rolling, polyorder=polyorder)
    trend = np.gradient(smoothed_prices)
    vol = dataset['close'].rolling(vol_window).std().values
    normalized_trend = np.where(vol != 0, trend / vol, np.nan)  # Set NaN where vol is 0
    labels = calculate_labels_trend_one_direction(normalized_trend, threshold, direction)
    dataset = dataset.iloc[:len(labels)].copy()
    dataset['labels'] = labels
    dataset = dataset.dropna()  # Remove rows with NaN
    return dataset
