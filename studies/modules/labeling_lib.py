import os
import random
import logging
import numpy as np
import pandas as pd
from numba import njit
from hdbscan import HDBSCAN
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import empirical_covariance
from hmmlearn import hmm, vhmm
from scipy.optimize import linear_sum_assignment
from scipy.signal import savgol_filter
from scipy.interpolate import UnivariateSpline
from scipy.signal import find_peaks
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
logging.getLogger('hmmlearn').setLevel(logging.ERROR)

# Obtener precios
def get_prices(symbol, timeframe, history_path) -> pd.DataFrame:
    history_file = os.path.join(history_path, f"{symbol}_{timeframe}.csv")
    p = pd.read_csv(history_file, sep=r"\s+")
    # Crear DataFrame con todas las columnas necesarias
    pFixed = pd.DataFrame(columns=['time', 'open', 'high', 'low', 'close', 'volume'])
    pFixed['time'] = p['<DATE>'] + ' ' + p['<TIME>']
    pFixed['time'] = pd.to_datetime(pFixed['time'], format='mixed')
    pFixed['open'] = p['<OPEN>']
    pFixed['high'] = p['<HIGH>']
    pFixed['low'] = p['<LOW>']
    pFixed['close'] = p['<CLOSE>']
    pFixed['volume'] = p['<TICKVOL>']
    pFixed.set_index('time', inplace=True)
    pFixed = pFixed.drop_duplicates().sort_index()
    return pFixed.dropna()

@njit(cache=True, nogil=True, fastmath=True)
def std_manual(x):
    m = mean_manual(x)
    return np.sqrt(np.sum((x - m) ** 2) / (x.size - 1)) if x.size > 1 else 0.0

@njit(cache=True, nogil=True, fastmath=True)
def skew_manual(x):
    s = std_manual(x)
    if s == 0:
        return 0.0
    m = mean_manual(x)
    return mean_manual(((x - m) / s) ** 3)

@njit(cache=True, nogil=True, fastmath=True)
def kurt_manual(x):
    s = std_manual(x)
    if s == 0:
        return 0.0
    m = mean_manual(x)
    return mean_manual(((x - m) / s) ** 4) - 3.0

@njit(cache=True, nogil=True, fastmath=True)
def zscore_manual(x):
    s = std_manual(x)
    if s == 0:
        return 0.0
    m = mean_manual(x)
    return (x[-1] - m) / s

@njit(cache=True, nogil=True, fastmath=True)
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

@njit(cache=True, nogil=True, fastmath=True)
def mean_manual(x):
    if x.size == 0:
        return 0.0
    sum_val = 0.0
    for i in range(x.size):
        sum_val += x[i]
    return sum_val / x.size

@njit(cache=True, nogil=True, fastmath=True)
def slope_manual(x):
    n = x.size
    if n <= 1:
        return 0.0
    
    # Crear el vector de índices x
    x_idx = np.arange(n)
    
    # Calcular medias usando la función existente
    x_mean = mean_manual(x_idx)
    y_mean = mean_manual(x)
    
    # Calcular covarianza
    cov = 0.0
    for i in range(n):
        cov += (x_idx[i] - x_mean) * (x[i] - y_mean)
    cov /= n
    
    # Calcular varianza de x usando std_manual
    x_std = std_manual(x_idx)
    var_x = x_std * x_std * (n - 1) / n  # Convertir de varianza muestral a poblacional
    
    return cov / var_x if var_x != 0 else 0.0

@njit(cache=True, nogil=True, fastmath=True)
def momentum_roc(x):
    if len(x) < 2: return 0.0
    ratio = x[0]/x[-1]
    return ratio - 1.0

@njit(cache=True, nogil=True, fastmath=True)
def fractal_dimension_manual(x):
    x = np.ascontiguousarray(x)
    eps = std_manual(x) / 4
    if eps == 0:
        return 1.0
    count = np.sum(np.abs(np.diff(x)) > eps)
    if count == 0:
        return 1.0
    return 1.0 + np.log(count) / np.log(len(x))

@njit(cache=True, nogil=True, fastmath=True)
def hurst_manual(x):
    n = x.size
    if n < 2:
        return 0.5
    
    # Calcular rangos reescalados
    valid_rs = np.zeros(n-1)
    valid_count = 0
    
    for i in range(1, n):
        # Calcular media y desviación estándar para cada subserie
        subseries = x[:i+1]
        m = mean_manual(subseries)
        s = std_manual(subseries)
        if s == 0:
            continue
            
        # Calcular rango reescalado
        max_val = subseries[0]
        min_val = subseries[0]
        for j in range(1, subseries.size):
            if subseries[j] > max_val:
                max_val = subseries[j]
            if subseries[j] < min_val:
                min_val = subseries[j]
        r = max_val - min_val
        rs = r / s
        if rs > 0:  # Solo guardar valores positivos
            valid_rs[valid_count] = rs
            valid_count += 1
    
    # Verificar si tenemos suficientes valores válidos
    if valid_count == 0:
        return 0.5
    
    # Usar solo los valores válidos para el cálculo
    log_rs = np.log(valid_rs[:valid_count])
    mean_log_rs = mean_manual(log_rs)
    log_n = np.log(n)
    
    # Evitar división por valores cercanos a cero
    if abs(log_n) < 1e-10:
        return 0.5
        
    return mean_log_rs / log_n

@njit(cache=True, nogil=True, fastmath=True)
def autocorr1_manual(x):
    n = x.size
    if n < 2:
        return 0.0
    xm = mean_manual(x)
    num = 0.0
    den = 0.0
    for i in range(n - 1):
        a = x[i]   - xm
        b = x[i+1] - xm
        num += a * b
        den += a * a
    return num / den if den != 0 else 0.0

@njit(cache=True, nogil=True, fastmath=True)
def max_dd_manual(x):
    peak = x[0]
    max_dd = 0.0
    for i in range(x.size):
        if x[i] > peak:
            peak = x[i]
        dd = (peak - x[i]) / peak
        if dd > max_dd:
            max_dd = dd
    return max_dd

@njit(cache=True, nogil=True, fastmath=True)
def sharpe_manual(x):
    if x.size < 2:
        return 0.0
    ret_sum = 0.0
    ret_sq  = 0.0
    for i in range(1, x.size):
        r = (x[i]/x[i-1] - 1)
        ret_sum += r
        ret_sq  += r*r
    n = x.size - 1
    mean = ret_sum / n
    std = std_manual(x)
    return mean / std if std != 0 else 0.0

@njit(cache=True, nogil=True, fastmath=True)
def fisher_transform(x):
    return 0.5 * np.log((1 + x) / (1 - x))

@njit(cache=True, nogil=True, fastmath=True)
def chande_momentum(x):
    returns = np.diff(x)
    up = np.sum(returns[returns > 0])
    down = np.abs(np.sum(returns[returns < 0]))
    return (up - down) / (up + down) if (up + down) != 0 else 0.0

@njit(cache=True, nogil=True, fastmath=True)
def approximate_entropy(x):
    n = len(x)
    m = 2
    if n <= m + 1:
        return 0.0
    sd = std_manual(x)
    r = 0.2 * sd
    count = 0
    for i in range(n - 1):
        for j in range(n - 1):
            if i != j and abs(x[i] - x[j]) <= r and abs(x[i+1] - x[j+1]) <= r:
                count += 1
    phi1 = np.log(count / (n - 1)) if count > 0 else 0.0
    count = 0
    for i in range(n - 2):
        for j in range(n - 2):
            if i != j and abs(x[i] - x[j]) <= r and abs(x[i+1] - x[j+1]) <= r and abs(x[i+2] - x[j+2]) <= r:
                count += 1
    phi2 = np.log(count / (n - 2)) if count > 0 else 0.0
    return phi1 - phi2

@njit(cache=True, nogil=True, fastmath=True)
def efficiency_ratio(x):
    direction = x[-1] - x[0]
    volatility = np.sum(np.abs(np.diff(x)))
    return direction/volatility if volatility != 0 else 0.0

@njit(cache=True, nogil=True, fastmath=True)
def corr_manual(a, b):
    if a.size != b.size or a.size < 2:
        return 0.0
    
    ma = mean_manual(a)
    mb = mean_manual(b)
    
    # Calcular covarianza
    cov = 0.0
    for i in range(a.size):
        cov += (a[i] - ma) * (b[i] - mb)
    
    # Calcular desviaciones estándar
    sa = std_manual(a)
    sb = std_manual(b)
    
    if sa == 0 or sb == 0:
        return 0.0
    
    return cov / (a.size * sa * sb)

@njit(cache=True, nogil=True, fastmath=True)
def correlation_skew_manual(x):
    lag = min(5, x.size // 2)
    if x.size < lag + 1:
        return 0.0
    corr_pos = corr_manual(x[:-lag], x[lag:])
    corr_neg = corr_manual(-x[:-lag], x[lag:])
    return corr_pos - corr_neg

@njit(cache=True, nogil=True, fastmath=True)
def median_manual(a):
    n = a.size
    if n == 0:
        return np.nan
    
    # Método más eficiente para ordenar en Numba
    b = a.copy()
    b.sort()  # Usar el método sort() del array directamente
    
    mid = n // 2
    if n % 2:
        return b[mid]
    else:
        return 0.5 * (b[mid-1] + b[mid])

@njit(cache=True, nogil=True, fastmath=True)
def jump_volatility_manual(x):
    if x.size < 2:
        return 0.0
    log_ret = np.log(x[:-1]/x[1:])
    med     = median_manual(log_ret)
    abs_dev = np.abs(log_ret - med)
    mad     = median_manual(abs_dev)
    thresh  = 3.0 * mad
    if mad == 0.0:
        return 0.0
    jumps = 0
    for i in range(log_ret.size):
        if abs_dev[i] > thresh:
            jumps += 1
    return jumps / log_ret.size

@njit(cache=True, nogil=True, fastmath=True)
def volatility_skew(x):
    n = len(x)
    if n < 2:
        return 0.0
    up_vol = std_manual(np.maximum(x[1:] - x[:-1], 0))
    down_vol = std_manual(np.maximum(x[:-1] - x[1:], 0))
    return (up_vol - down_vol)/(up_vol + down_vol) if (up_vol + down_vol) != 0 else 0.0

# Ingeniería de características
@njit(cache=True, nogil=True, fastmath=True)
def compute_features(close, periods_main, periods_meta, stats_main, stats_meta):
    n = len(close)
    # Calcular total de features considerando si hay meta o no
    total_features = len(periods_main) * len(stats_main)
    if periods_meta is not None and stats_meta is not None:
        total_features += len(periods_meta) * len(stats_meta)
    features = np.full((n, total_features), np.nan)

    # Procesar períodos main
    col = 0
    for win in periods_main:
        for s in stats_main:
            for i in range(win, n):
                window = np.ascontiguousarray(close[i - win:i][::-1])
                try:
                    if s == "std":
                        features[i, col] = std_manual(window)
                    elif s == "skew":
                        features[i, col] = skew_manual(window)
                    elif s == "kurt":
                        features[i, col] = kurt_manual(window)
                    elif s == "zscore":
                        features[i, col] = zscore_manual(window)
                    elif s == "range":
                        features[i, col] = np.max(window) - np.min(window)
                    elif s == "mad":
                        m = mean_manual(window)
                        features[i, col] = mean_manual(np.abs(window - m))
                    elif s == "entropy":
                        features[i, col] = entropy_manual(window)
                    elif s == "slope":
                        features[i, col] = slope_manual(window)
                    elif s == "momentum":
                        features[i, col] = momentum_roc(window)
                    elif s == "fractal":
                        features[i, col] = fractal_dimension_manual(window)
                    elif s == "hurst":
                        features[i, col] = hurst_manual(window)
                    elif s == "autocorr":
                        features[i, col] = autocorr1_manual(window)
                    elif s == "max_dd":
                        features[i, col] = max_dd_manual(window)
                    elif s == "sharpe":
                        features[i, col] = sharpe_manual(window)
                    elif s == "fisher":
                        features[i, col] = fisher_transform(momentum_roc(window))
                    elif s == "chande":
                        features[i, col] = chande_momentum(window)
                    elif s == "var":
                        std = std_manual(window)
                        features[i, col] = std * std * (window.size - 1) / window.size
                    elif s == "approx_entropy":
                        features[i, col] = approximate_entropy(window)
                    elif s == "eff_ratio":
                        features[i, col] = efficiency_ratio(window)
                    elif s == "corr_skew":
                        features[i, col] = correlation_skew_manual(window)
                    elif s == "jump_vol":
                        features[i, col] = jump_volatility_manual(window)
                    elif s == "vol_skew":
                        features[i, col] = volatility_skew(window)
                except:
                    return np.full((n, total_features), np.nan)
            col += 1

    # Procesar períodos meta solo si existen
    if periods_meta is not None and stats_meta is not None:
        for win in periods_meta:
            for s in stats_meta:
                for i in range(win, n):
                    window = np.ascontiguousarray(close[i - win:i][::-1])
                    try:
                        if s == "std":
                            features[i, col] = std_manual(window)
                        elif s == "skew":
                            features[i, col] = skew_manual(window)
                        elif s == "kurt":
                            features[i, col] = kurt_manual(window)
                        elif s == "zscore":
                            features[i, col] = zscore_manual(window)
                        elif s == "range":
                            features[i, col] = np.max(window) - np.min(window)
                        elif s == "mad":
                            m = mean_manual(window)
                            features[i, col] = mean_manual(np.abs(window - m))
                        elif s == "entropy":
                            features[i, col] = entropy_manual(window)
                        elif s == "slope":
                            features[i, col] = slope_manual(window)
                        elif s == "momentum":
                            features[i, col] = momentum_roc(window)
                        elif s == "fractal":
                            features[i, col] = fractal_dimension_manual(window)
                        elif s == "hurst":
                            features[i, col] = hurst_manual(window)
                        elif s == "autocorr":
                            features[i, col] = autocorr1_manual(window)
                        elif s == "max_dd":
                            features[i, col] = max_dd_manual(window)
                        elif s == "sharpe":
                            features[i, col] = sharpe_manual(window)
                        elif s == "fisher":
                            features[i, col] = fisher_transform(momentum_roc(window))
                        elif s == "chande":
                            features[i, col] = chande_momentum(window)
                        elif s == "var":
                            std = std_manual(window)
                            features[i, col] = std * std * (window.size - 1) / window.size
                        elif s == "approx_entropy":
                            features[i, col] = approximate_entropy(window)
                        elif s == "eff_ratio":
                            features[i, col] = efficiency_ratio(window)
                        elif s == "corr_skew":
                            features[i, col] = correlation_skew_manual(window)
                        elif s == "jump_vol":
                            features[i, col] = jump_volatility_manual(window)
                        elif s == "vol_skew":
                            features[i, col] = volatility_skew(window)
                    except:
                        return np.full((n, total_features), np.nan)
                col += 1

    return features

def get_features(data: pd.DataFrame, hp):
    close = data['close'].values
    index = data.index
    periods_main = hp["periods_main"]
    stats_main = hp["stats_main"]
    
    # Obtener períodos y estadísticas meta solo si existen
    periods_meta = hp.get("periods_meta")
    stats_meta = hp.get("stats_meta")
    
    if len(stats_main) == 0:
        raise ValueError("La lista de estadísticas MAIN está vacía.")
    
    # Asegurar que los arrays sean contiguos
    close = np.ascontiguousarray(close)
    
    feats = compute_features(close, periods_main, periods_meta, stats_main, stats_meta)
    if np.isnan(feats).all():
        return pd.DataFrame(index=index)
    
    # Nombres de columnas
    colnames = []
    for p in periods_main:
        for s in stats_main:
            colnames.extend([f"{p}_{s}_feature"])
    
    # Agregar nombres de columnas meta solo si existen
    if periods_meta is not None and stats_meta is not None:
        for p in periods_meta:
            for s in stats_meta:
                colnames.extend([f"{p}_{s}_meta_feature"])
    # ───────────- NUEVO BLOQUE FINAL -───────────
    df = pd.DataFrame(feats, columns=colnames, index=index)

    # 1) añadir OHLCV **antes** de limpiar para que sufran el mismo filtrado
    ohlcv = data.loc[index, ["open", "high", "low", "close", "volume"]]
    df = pd.concat([df, ohlcv], axis=1)

    # 2) limpiar Inf/NaN SOLO en columnas de features
    feat_cols = df.filter(like='_feature').columns
    df[feat_cols] = df[feat_cols].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=feat_cols)

    # 3) verificación — el close debe coincidir al 100 %
    if not np.allclose(df["close"].values,
                    ohlcv.loc[df.index, "close"].values):
        raise RuntimeError("❌ 'close' desalineado tras generar features")

    return df

# TREND OR NEUTRAL BASED LABELING
@njit(cache=True, nogil=True, fastmath=True)
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

def get_labels(dataset, markup, min=1, max=15, atr_period=14) -> pd.DataFrame:
    """Label trades for both directions using the ATR based methodology.

    This function is kept for backwards compatibility and internally delegates
    to :func:`get_labels_one_direction` with ``direction='both'``.
    """

    labeled = get_labels_one_direction(
        dataset,
        markup=markup,
        min_val=min,
        max_val=max,
        direction='both',
        atr_period=atr_period,
    )
    labeled = labeled.rename(columns={'labels_main': 'labels'})
    return labeled

@njit(cache=True, nogil=True, fastmath=True)
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

@njit(cache=True, nogil=True, fastmath=True)
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

@njit(cache=True, nogil=True, fastmath=True)
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

@njit(cache=True, nogil=True, fastmath=True)
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

@njit(cache=True, nogil=True, fastmath=True)
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
    kmeans = KMeans(n_clusters=num_clusters, n_init='auto')
    dataset['cluster'] = kmeans.fit_predict(dataset[['close']])

    close_data = dataset['close'].values
    clusters = dataset['cluster'].values

    labels = calculate_labels_clusters(close_data, clusters, markup)

    dataset['labels'] = labels
    dataset = dataset.drop(dataset[dataset.labels == 2.0].index)
    dataset = dataset.drop(columns=['cluster'])
    return dataset

@njit(cache=True, nogil=True, fastmath=True)
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

@njit(cache=True, nogil=True, fastmath=True)
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

@njit(cache=True, nogil=True, fastmath=True)
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
@njit(cache=True, nogil=True, fastmath=True)
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

@njit(cache=True, nogil=True, fastmath=True)
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

@njit(cache=True, nogil=True, fastmath=True)
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
        shift (int, optional): Shift the smoothed price data forward (positive) or backward (negative).
                                 Useful for creating a lag/lead effect. Defaults to 1.
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
@njit(cache=True, nogil=True, fastmath=True)
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

@njit(cache=True, nogil=True, fastmath=True)
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
        quantiles (list, optional): Quantiles to define the "reversion zone". Defaults to [.45, .55].
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

@njit(cache=True, nogil=True, fastmath=True)
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

@njit(cache=True, nogil=True, fastmath=True)
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
        direction (str, optional):
            Direction to generate signals for. ``'buy'`` or ``'sell'`` will
            return labels for that specific side. ``'both'`` combines the two
            and returns ``0`` for buy, ``1`` for sell and drops neutral periods.

    Returns:
        pd.DataFrame: The original DataFrame with a new ``labels_main`` column
        and filtered rows:
            - ``labels_main`` column:
                - ``0``: No trade
                - ``1``: Trade (buy or sell depending on ``direction``)
            - Rows with missing values (``NaN``) are removed.
            - The temporary ``lvl`` column is removed.
    """

    # Calculate smoothed prices using the Savitzky-Golay filter
    smoothed_prices = savgol_filter(dataset['close'].values, window_length=rolling, polyorder=polyorder)
    
    # Calculate the difference between the actual closing prices and the smoothed prices
    diff = dataset['close'] - smoothed_prices
    
    # Add the difference as a new column 'lvl' to the DataFrame
    dataset['lvl'] = diff
    
    # Remove any rows with NaN values 
    dataset = dataset.dropna()
    
    # Calculate the quantiles of the 'lvl' column (price deviation)
    q = dataset['lvl'].quantile(quantiles).to_list() 

    # Extract the closing prices and the calculated 'lvl' values as NumPy arrays
    close = dataset['close'].values
    lvl = dataset['lvl'].values
    
    if direction in {'buy', 'sell'}:
        labels = calculate_labels_filter_one_direction(close, lvl, q, direction)

        dataset = dataset.iloc[:len(labels)].copy()
        dataset['labels_main'] = labels
        dataset = dataset.dropna()
        return dataset.drop(columns=['lvl'])

    if direction == 'both':
        labels_buy = calculate_labels_filter_one_direction(close, lvl, q, 'buy')
        labels_sell = calculate_labels_filter_one_direction(close, lvl, q, 'sell')

        n = min(len(labels_buy), len(labels_sell))
        labels = np.full(n, 2.0, dtype=np.float64)

        buy_sig = labels_buy[:n] == 1.0
        sell_sig = labels_sell[:n] == 1.0

        labels[buy_sig & ~sell_sig] = 0.0
        labels[sell_sig & ~buy_sig] = 1.0

        dataset = dataset.iloc[:n].copy()
        dataset['labels_main'] = labels
        dataset = dataset.dropna()
        dataset = dataset.drop(dataset[dataset.labels_main == 2.0].index)
        return dataset.drop(columns=['lvl'])

    raise ValueError("direction must be 'buy', 'sell', or 'both'")

@njit(cache=True, nogil=True, fastmath=True)
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
    """Label trends for a single or both directions.

    Parameters
    ----------
    dataset : pd.DataFrame
        Input price data with a ``close`` column.
    direction : {'buy', 'sell', 'both'}, optional
        When ``'both'`` the returned ``labels_main`` column contains ``0`` for
        buy trades and ``1`` for sell trades. Neutral periods are dropped.
    """

    smoothed_prices = savgol_filter(dataset['close'].values, window_length=rolling, polyorder=polyorder)
    trend = np.gradient(smoothed_prices)
    vol = dataset['close'].rolling(vol_window).std().values
    normalized_trend = np.where(vol != 0, trend / vol, np.nan)

    if direction in {'buy', 'sell'}:
        labels = calculate_labels_trend_one_direction(normalized_trend, threshold, direction)

        dataset = dataset.iloc[:len(labels)].copy()
        dataset['labels_main'] = labels
        dataset = dataset.dropna()
        return dataset

    if direction == 'both':
        labels_buy = calculate_labels_trend_one_direction(normalized_trend, threshold, 'buy')
        labels_sell = calculate_labels_trend_one_direction(normalized_trend, threshold, 'sell')

        n = min(len(labels_buy), len(labels_sell))
        labels = np.full(n, 2.0, dtype=np.float64)

        buy_sig = labels_buy[:n] == 1.0
        sell_sig = labels_sell[:n] == 1.0

        labels[buy_sig & ~sell_sig] = 0.0
        labels[sell_sig & ~buy_sig] = 1.0

        dataset = dataset.iloc[:n].copy()
        dataset['labels_main'] = labels
        dataset = dataset.dropna()
        dataset = dataset.drop(dataset[dataset.labels_main == 2.0].index)
        return dataset

    raise ValueError("direction must be 'buy', 'sell', or 'both'")

@njit(cache=True, nogil=True, fastmath=True)
def calculate_labels_filter_flat(close, lvl, q):
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

def get_labels_filter_flat(dataset, rolling=200, quantiles=[.45, .55], polyorder=3, decay_factor=0.95) -> pd.DataFrame:
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
    
    dataset['lvl'] = 1/weighted_diff # Add the weighted difference as 'lvl'

    # Remove any rows with NaN values 
    dataset = dataset.dropna()
    
    # Calculate the quantiles of the 'lvl' column (price deviation)
    q = dataset['lvl'].quantile(quantiles).to_list() 

    # Extract the closing prices and the calculated 'lvl' values as NumPy arrays
    close = dataset['close'].values
    lvl = dataset['lvl'].values
    
    # Calculate buy/sell labels using the 'calculate_labels_filter' function 
    labels = calculate_labels_filter_flat(close, lvl, q) 

    # Trim the dataset to match the length of the calculated labels
    dataset = dataset.iloc[:len(labels)].copy()
    
    # Add the calculated labels as a new 'labels' column to the DataFrame
    dataset['labels'] = labels
    
    # Remove any rows with NaN values
    dataset = dataset.dropna()
    
    # Remove rows where the 'labels' column has a value of 2.0 (sell signals)
    # dataset = dataset.drop(dataset[dataset.labels == 2.0].index)
    
    # Return the modified DataFrame with the 'lvl' column removed
    return dataset.drop(columns=['lvl'])

@njit(cache=True, nogil=True, fastmath=True)
def calculate_atr_simple(high, low, close, period=14):
    n   = len(close)
    tr  = np.empty(n)
    atr = np.empty(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i-1])
        lc = abs(low[i]  - close[i-1])
        tr[i] = max(hl, hc, lc)
    # promedio acumulado para i < period-1
    cumsum = tr[0]
    atr[0] = tr[0]
    for i in range(1, min(period, n)):
        cumsum += tr[i]
        atr[i] = cumsum / (i+1)
    if n <= period-1:
        return atr
    # primera media "oficial" (índice period-1)
    cumsum += tr[period-1]
    atr[period-1] = cumsum / period
    # Wilder a partir de aquí
    for i in range(period, n):
        atr[i] = (atr[i-1]*(period-1) + tr[i]) / period
    return atr

# ONE DIRECTION LABELING
@njit(cache=True, nogil=True, fastmath=True)
def calculate_labels_one_direction(high, low, close, markup, min_val, max_val, direction, atr_period=14):
    # Verificar que hay suficientes datos  
    n = len(close)
    if n <= max_val:
        return np.zeros(0, dtype=np.float64)

    # Calcular ATR
    atr = calculate_atr_simple(high, low, close, period=atr_period)

    # Modo aleatorio permanente
    result = np.zeros(n - max_val, dtype=np.float64)

    for i in range(n - max_val):
        # Seleccionar barra aleatoria dentro del rango
        rand_idx = np.random.randint(min_val, max_val + 1)
        future_price = close[i + rand_idx]

        # Calcular markup dinámico
        dyn_mk = markup * atr[i]

        # Verificar condición según dirección
        if direction == "buy":
            if future_price > close[i] + dyn_mk:
                result[i] = 1.0
        else:  # sell
            if future_price < close[i] - dyn_mk:
                result[i] = 1.0

    return result

def get_labels_one_direction(dataset, markup, min_val=1, max_val=15,
                             direction='buy', atr_period=14) -> pd.DataFrame:
    """Label trades for a single or both directions using ATR based distance.

    Parameters
    ----------
    dataset : pd.DataFrame
        Input OHLCV data.
    markup : float
        Multiplier for the ATR to define the distance to the target candle.
    min_val, max_val : int, optional
        Range (in candles) to evaluate the trade outcome.
    direction : {'buy', 'sell', 'both'}, optional
        Direction to evaluate. ``'both'`` returns labels for long and short
        trades simultaneously.
    atr_period : int, optional
        Period for the ATR calculation.

    Returns
    -------
    pd.DataFrame
        The original dataset trimmed and with a ``labels_main`` column.
        When ``direction`` is ``'both'`` the labels are ``0`` for buy and ``1``
        for sell trades. Rows without a clear signal are removed.
    """

    close_data = np.ascontiguousarray(dataset['close'].values)
    high_data = np.ascontiguousarray(dataset['high'].values)
    low_data = np.ascontiguousarray(dataset['low'].values)

    if direction in {'buy', 'sell'}:
        labels = calculate_labels_one_direction(
            high_data, low_data, close_data,
            markup, min_val, max_val, direction,
            atr_period
        )
        dataset = dataset.iloc[:len(labels)].copy()
        dataset['labels_main'] = labels
        dataset = dataset.dropna()
        return dataset

    if direction == 'both':
        labels_buy = calculate_labels_one_direction(
            high_data, low_data, close_data,
            markup, min_val, max_val, 'buy',
            atr_period
        )
        labels_sell = calculate_labels_one_direction(
            high_data, low_data, close_data,
            markup, min_val, max_val, 'sell',
            atr_period
        )

        n = min(len(labels_buy), len(labels_sell))
        labels = np.full(n, 2.0, dtype=np.float64)

        buy_sig = labels_buy[:n] == 1.0
        sell_sig = labels_sell[:n] == 1.0
        labels[buy_sig & ~sell_sig] = 0.0
        labels[sell_sig & ~buy_sig] = 1.0

        dataset = dataset.iloc[:n].copy()
        dataset['labels_main'] = labels
        dataset = dataset.dropna()
        dataset = dataset.drop(dataset[dataset.labels_main == 2.0].index)
        return dataset

    raise ValueError("direction must be 'buy', 'sell', or 'both'")

def sliding_window_clustering(
        dataset: pd.DataFrame,
        n_clusters: int,
        window_size: int = 100,
        step: int = None) -> pd.DataFrame:
    
    if dataset.empty:
        dataset["labels_meta"] = -1
        return dataset
    
    # Pre-cálculo
    n_rows = len(dataset)
    votes = np.zeros((n_rows, n_clusters + 1), dtype=np.uint8)
    # K-means global
    meta_X = dataset.filter(regex="meta_feature")
    if meta_X.shape[1] == 0:
        dataset["labels_meta"] = -1
        return dataset
    meta_X_np = meta_X.to_numpy(np.float32)
    global_km = KMeans(n_clusters=n_clusters, n_init="auto").fit(meta_X_np)
    global_ct = global_km.cluster_centers_
    
    def map_centroids(local_ct: np.ndarray) -> dict[int, int]:
        cost = np.linalg.norm(local_ct[:, None] - global_ct, axis=2)
        row, col = linear_sum_assignment(cost)
        return {int(r): int(c) + 1 for r, c in zip(row, col)}
    
    # Sliding windows
    starts = []
    ends = []
    i = 0
    while i + window_size <= n_rows:
        starts.append(i)
        ends.append(i + window_size)
        # Step optimizado o por defecto igual al tamaño de la ventana
        step_val = step if step is not None else window_size
        i += step_val
    
    # Procesar todas las ventanas válidas
    for start, end in zip(starts, ends):
        local_data = meta_X_np[start:end]
        if len(local_data) < n_clusters:
            continue
        local_km = KMeans(n_clusters, n_init="auto").fit(local_data)
        local_ct = local_km.cluster_centers_
        if local_ct.shape[0] < global_ct.shape[0]:
            continue
        mapping = map_centroids(local_ct)
        lbls = local_km.labels_
        indices = np.arange(start, end)
        mapped_ids = np.array([mapping.get(lab, 0) for lab in lbls])
        votes[indices, mapped_ids] += 1
    
    clusters = votes.argmax(axis=1).astype(np.int32) - 1
    ds_out = dataset.copy()
    ds_out["labels_meta"] = clusters
    return ds_out

def clustering_simple(dataset: pd.DataFrame,
               min_cluster_size: int = 20,
               min_samples: int | None = None) -> pd.DataFrame:
    # Si el dataset está vacío, devuelve un dataset con -1 en la columna labels_meta
    if dataset.empty:
        dataset["labels_meta"] = -1
        return dataset
    meta_X = dataset.filter(regex="meta_feature")
    if meta_X.shape[1] == 0:
        dataset["labels_meta"] = -1
        return dataset
    meta_X_np = meta_X.to_numpy(np.float32)
    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples
    ).fit(meta_X_np)

    dataset["labels_meta"] = clusterer.labels_.astype(int)
    return dataset

def markov_regime_switching_simple(dataset, n_regimes: int, model_type="GMMHMM", n_iter = 10, n_mix = 3) -> pd.DataFrame:
    # Si el dataset está vacío, devuelve un dataset con -1 en la columna labels_meta
    if dataset.empty:
        dataset["labels_meta"] = -1
        return dataset

    meta_X_np = dataset.filter(regex="meta_feature").to_numpy(np.float32)

    # Features normalization before training
    scaler = StandardScaler()
    try:
        X_scaled = scaler.fit_transform(meta_X_np)
    except Exception:
        dataset["labels_meta"] = -1
        return dataset
    
    # 3) Verificar columnas con nan/inf después del escalado
    if not np.isfinite(X_scaled).all():
        dataset["labels_meta"] = -1
        return dataset
    
    # 4) Verificar columnas (casi) constantes después del escalado
    stds = np.nanstd(X_scaled, axis=0)
    if np.any(stds < 1e-10):
        dataset["labels_meta"] = -1
        return dataset
    
    # Create and train the HMM model
    if model_type == "HMM":
        model = hmm.GaussianHMM(
            n_components=n_regimes,
            covariance_type="full",
            n_iter=n_iter,
            verbose=False
        )
    elif model_type == "GMMHMM":
        model = hmm.GMMHMM(
            n_components=n_regimes,
            covariance_type="full",
            n_iter=n_iter,
            n_mix=n_mix,
            verbose=False
        )
    elif model_type == "VARHMM":
        model = vhmm.VariationalGaussianHMM(
            n_components=n_regimes,
            covariance_type="full",
            n_iter=n_iter,
            verbose=False
        )
        
    # Fit the model
    try:
        model.fit(X_scaled)
        hidden_states = model.predict(X_scaled)
        # Assign states to clusters
        dataset['labels_meta'] = hidden_states
    except Exception:
        dataset["labels_meta"] = -1
        
    return dataset

def markov_regime_switching_advanced(dataset, n_regimes: int, model_type="HMM", n_iter=100, n_mix = 3) -> pd.DataFrame:
    # Si el dataset está vacío, devuelve un dataset con -1 en la columna labels_meta
    if dataset.empty:
        dataset["labels_meta"] = -1
        return dataset
    
    meta_X_np = dataset.filter(regex="meta_feature").to_numpy(np.float32)
    
    # Features normalization before training
    scaler = StandardScaler()
    try:
        X_scaled = scaler.fit_transform(meta_X_np)
    except Exception:
        dataset["labels_meta"] = -1
        return dataset
    
    # 3) Verificar columnas con nan/inf después del escalado
    if not np.isfinite(X_scaled).all():
        dataset["labels_meta"] = -1
        return dataset
    
    # 4) Verificar columnas (casi) constantes después del escalado
    stds = np.nanstd(X_scaled, axis=0)
    if np.any(stds < 1e-10):
        dataset["labels_meta"] = -1
        return dataset
    
    # Use k-means to cluster the data into n_regimes groups
    try:
        kmeans = KMeans(n_clusters=n_regimes, n_init='auto')
        cluster_labels = kmeans.fit_predict(X_scaled)
    except Exception:
        dataset["labels_meta"] = -1
        return dataset
    
    # Calculate cluster-specific means and covariances to use as priors
    prior_means = kmeans.cluster_centers_  # Shape: (n_regimes, n_features)
    
    # Calculate empirical covariance for each cluster
    prior_covs = []
    
    for i in range(n_regimes):
        cluster_data = X_scaled[cluster_labels == i]
        if len(cluster_data) > 1:  # Need at least 2 points for covariance
            try:
                cluster_cov = empirical_covariance(cluster_data)
                # Verificar que la covarianza sea definida positiva
                np.linalg.cholesky(cluster_cov)
                prior_covs.append(cluster_cov)
            except np.linalg.LinAlgError:
                # Fallback to overall covariance if cluster is too small or not positive definite
                prior_covs.append(empirical_covariance(X_scaled))
        else:
            # Fallback to overall covariance if cluster is too small
            prior_covs.append(empirical_covariance(X_scaled))
    
    prior_covs = np.array(prior_covs)  # Shape: (n_regimes, n_features, n_features)
    
    # Calculate initial state distribution from cluster proportions
    initial_probs = np.bincount(cluster_labels, minlength=n_regimes) / len(cluster_labels)
    
    # Calculate transition matrix based on cluster sequences
    trans_mat = np.zeros((n_regimes, n_regimes))
    for t in range(1, len(cluster_labels)):
        trans_mat[cluster_labels[t-1], cluster_labels[t]] += 1
        
    # Normalize rows to get probabilities
    row_sums = trans_mat.sum(axis=1, keepdims=True)
    # Avoid division by zero
    row_sums[row_sums == 0] = 1
    trans_mat = trans_mat / row_sums

    # Initialize model parameters based on model type
    if model_type == "HMM":
        model_params = {
            'n_components': n_regimes,
            'covariance_type': "full",
            'n_iter': n_iter,
            'init_params': '',  # Don't use default initialization
            'verbose': False
        }
        
        model = hmm.GaussianHMM(**model_params)
        
        # Set the model parameters directly with our k-means derived priors
        model.startprob_ = initial_probs
        model.transmat_ = trans_mat
        model.means_ = prior_means
        model.covars_ = prior_covs
        
    elif model_type == "GMMHMM":
        model_params = {
            'n_components': n_regimes,
            'covariance_type': "full",
            'n_iter': n_iter,
            'n_mix': n_mix,
            'init_params': '',
            'verbose': False
        }
        
        model = hmm.GMMHMM(**model_params)
        
        # Set initial state distribution and transition matrix
        model.startprob_ = initial_probs
        model.transmat_ = trans_mat
        
        # For GMMHMM, means_ should have shape (n_components, n_mix, n_features)
        # Currently prior_means has shape (n_regimes, n_features)
        # We need to reshape it properly for GMMHMM
        
        n_features = X_scaled.shape[1]
        n_mix = model_params['n_mix']
        
        # Initialize means with a different mixture mean for each component
        # Creating n_mix variations of each cluster center
        model.means_ = np.zeros((n_regimes, n_mix, n_features))
        for i in range(n_regimes):
            for j in range(n_mix):
                # Add some small variation for different mixtures
                model.means_[i, j] = prior_means[i] + (j - n_mix/2) * 0.2 * np.std(X_scaled, axis=0)
        
        # Similarly for covariances, shape should be (n_components, n_mix, n_features, n_features)
        model.covars_ = np.zeros((n_regimes, n_mix, n_features, n_features))
        for i in range(n_regimes):
            for j in range(n_mix):
                model.covars_[i, j] = prior_covs[i] + np.random.rand(n_features, n_features) * 0.01
        
    elif model_type == "VARHMM":
        # For VariationalGaussianHMM
        n_features = X_scaled.shape[1]
        
        model_params = {
            'n_components': n_regimes,
            'covariance_type': "full",
            'n_iter': n_iter,
            'init_params': '',
            # Set priors directly in the parameters
            'means_prior': prior_means,
            'beta_prior': np.ones(n_regimes),  # Shape: (n_components,)
            'dof_prior': np.ones(n_regimes) * (n_features + 2),  # Shape: (n_components,)
            'scale_prior': prior_covs,  # Shape: (n_components, n_features, n_features)
            'verbose': False
        }
        
        # Create the VARHMM model
        model = vhmm.VariationalGaussianHMM(**model_params)
        
        # Set initial state distribution and transition matrix
        model.startprob_ = initial_probs
        model.transmat_ = trans_mat

    # Fit the model
    try:
        model.fit(X_scaled)
        hidden_states = model.predict(X_scaled)
        
        # Assign states to clusters
        dataset['labels_meta'] = hidden_states
    except Exception:
        dataset["labels_meta"] = -1
        
    return dataset


def lgmm_clustering(dataset: pd.DataFrame, n_components: int,
                    covariance_type: str = "full",
                    max_iter: int = 100) -> pd.DataFrame:
    """Cluster meta features using a GaussianMixture model.

    Parameters
    ----------
    dataset : pd.DataFrame
        Input data with ``*_meta_feature`` columns.
    n_components : int
        Number of mixture components.
    covariance_type : str, optional
        Covariance type for ``GaussianMixture``.
    max_iter : int, optional
        Maximum EM iterations.

    Returns
    -------
    pd.DataFrame
        Dataset with an added ``labels_meta`` column containing cluster ids. If
        clustering fails the column will be ``-1``.
    """
    if dataset.empty:
        dataset["labels_meta"] = -1
        return dataset

    meta_X = dataset.filter(regex="meta_feature")
    if meta_X.shape[1] == 0:
        dataset["labels_meta"] = -1
        return dataset

    try:
        gm = GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            max_iter=max_iter,
            random_state=0,
        )
        labels = gm.fit_predict(meta_X.to_numpy(np.float32))
        dataset["labels_meta"] = labels.astype(int)
    except Exception:
        dataset["labels_meta"] = -1

    return dataset

