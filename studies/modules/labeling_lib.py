import os
import logging
import numpy as np
import pandas as pd
import cupy as cp
#import ot
from numba import njit, prange
from numba.typed import List
from hdbscan import HDBSCAN
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import empirical_covariance
from hmmlearn import hmm, vhmm
from scipy.optimize import linear_sum_assignment
from scipy.signal import savgol_filter
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import cdist
from typing import Tuple
import numpy.random as npr
from scipy.interpolate import UnivariateSpline
from scipy.signal import find_peaks
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from concurrent.futures import ProcessPoolExecutor, as_completed

# Configuración de logging
logging.getLogger('hmmlearn').setLevel(logging.ERROR)
logging.getLogger("tensorflow").setLevel(logging.ERROR)

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

@njit(cache=True, fastmath=True)
def std_manual(x):
    m = mean_manual(x)
    return np.sqrt(np.sum((x - m) ** 2) / (x.size - 1)) if x.size > 1 else 0.0

@njit(cache=True, fastmath=True)
def skew_manual(x):
    s = std_manual(x)
    if s == 0:
        return 0.0
    m = mean_manual(x)
    return mean_manual(((x - m) / s) ** 3)

@njit(cache=True, fastmath=True)
def kurt_manual(x):
    s = std_manual(x)
    if s == 0:
        return 0.0
    m = mean_manual(x)
    return mean_manual(((x - m) / s) ** 4) - 3.0

@njit(cache=True, fastmath=True)
def zscore_manual(x):
    s = std_manual(x)
    if s == 0:
        return 0.0
    m = mean_manual(x)
    return (x[-1] - m) / s

@njit(cache=True, fastmath=True)
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

@njit(cache=True, fastmath=True)
def mean_manual(x):
    if x.size == 0:
        return 0.0
    sum_val = 0.0
    for i in range(x.size):
        sum_val += x[i]
    return sum_val / x.size

@njit(cache=True, fastmath=True)
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

@njit(cache=True, fastmath=True)
def momentum_roc(x):
    if len(x) < 2: return 0.0
    ratio = x[0]/x[-1]
    return ratio - 1.0

@njit(cache=True, fastmath=True)
def fractal_dimension_manual(x):
    x = np.ascontiguousarray(x)
    eps = std_manual(x) / 4
    if eps == 0:
        return 1.0
    count = np.sum(np.abs(np.diff(x)) > eps)
    if count == 0:
        return 1.0
    return 1.0 + np.log(count) / np.log(len(x))

@njit(cache=True, fastmath=True)
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
        label_max_val = subseries[0]
        label_min_val = subseries[0]
        for j in range(1, subseries.size):
            if subseries[j] > label_max_val:
                label_max_val = subseries[j]
            if subseries[j] < label_min_val:
                label_min_val = subseries[j]
        r = label_max_val - label_min_val
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

@njit(cache=True, fastmath=True)
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

@njit(cache=True, fastmath=True)
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

@njit(cache=True, fastmath=True)
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

@njit(cache=True, fastmath=True)
def fisher_transform(x):
    return 0.5 * np.log((1 + x) / (1 - x))

@njit(cache=True, fastmath=True)
def chande_momentum(x):
    returns = np.diff(x)
    up = np.sum(returns[returns > 0])
    down = np.abs(np.sum(returns[returns < 0]))
    return (up - down) / (up + down) if (up + down) != 0 else 0.0

@njit(cache=True, fastmath=True)
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

@njit(cache=True, fastmath=True)
def efficiency_ratio(x):
    direction = x[-1] - x[0]
    volatility = np.sum(np.abs(np.diff(x)))
    return direction/volatility if volatility != 0 else 0.0

@njit(cache=True, fastmath=True)
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

@njit(cache=True, fastmath=True)
def correlation_skew_manual(x):
    lag = min(5, x.size // 2)
    if x.size < lag + 1:
        return 0.0
    corr_pos = corr_manual(x[:-lag], x[lag:])
    corr_neg = corr_manual(-x[:-lag], x[lag:])
    return corr_pos - corr_neg

@njit(cache=True, fastmath=True)
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

@njit(cache=True, fastmath=True)
def iqr_manual(a):
    n = a.size
    if n == 0:
        return 0.0
    b = a.copy()
    b.sort()
    q1_idx = int(0.25 * (n - 1))
    q3_idx = int(0.75 * (n - 1))
    return b[q3_idx] - b[q1_idx]

@njit(cache=True, fastmath=True)
def coeff_var_manual(a):
    m = mean_manual(a)
    if m == 0:
        return 0.0
    s = std_manual(a)
    return s / m

@njit(cache=True, fastmath=True)
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

@njit(cache=True, fastmath=True)
def volatility_skew(x):
    n = len(x)
    if n < 2:
        return 0.0
    up_vol = std_manual(np.maximum(x[1:] - x[:-1], 0))
    down_vol = std_manual(np.maximum(x[:-1] - x[1:], 0))
    return (up_vol - down_vol)/(up_vol + down_vol) if (up_vol + down_vol) != 0 else 0.0

# ───── HELPER PARA RETORNOS ─────
@njit(cache=True, fastmath=True)
def compute_returns(prices):
    """
    Calcula retornos logarítmicos de forma eficiente.
    Retorna un array de size-1 con los retornos.
    """
    if prices.size <= 1:
        return np.empty(0, dtype=np.float64)
    
    returns = np.empty(prices.size - 1, dtype=np.float64)
    for i in range(prices.size - 1):
        if prices[i] <= 0:
            returns[i] = 0.0  # Evitar log(0) o log(negativo)
        else:
            returns[i] = np.log(prices[i + 1] / prices[i])
    return returns

# ───── ESTADÍSTICOS QUE USAN RETORNOS ─────
@njit(cache=True, fastmath=True)
def should_use_returns(stat_name):
    """Determina si un estadístico debe usar retornos en lugar de precios."""
    return stat_name in ("mean", "median", "std", "iqr", "mad")

# Ingeniería de características
@njit(cache=True, fastmath=True)
def compute_features(close, periods_main, periods_meta, stats_main, stats_meta):
    n = len(close)
    # Calcular total de features considerando si hay meta o no
    total_features = len(periods_main) * len(stats_main)
    if len(periods_meta) > 0 and len(stats_meta) > 0:
        total_features += len(periods_meta) * len(stats_meta)
    features = np.full((n, total_features), np.nan)

    # ───── OPTIMIZACIÓN: Pre-calcular ventanas para evitar recálculos ─────
    col = 0
    
    # Procesar períodos main
    for win in periods_main:
        for s in stats_main:
            # ───── AJUSTAR VENTANA PARA ESTADÍSTICOS CON RETORNOS ─────
            if should_use_returns(s):
                window_size = win + 1  # +1 precio para obtener 'win' retornos
                start_idx = window_size
            else:
                window_size = win
                start_idx = win
                
            for i in range(start_idx, n):
                # ───── OPTIMIZACIÓN: Usar slice directo en lugar de [::-1] ─────
                window = close[i - window_size:i]
                
                # ───── NUEVA FUNCIONALIDAD: Usar retornos para estadísticos específicos ─────
                if should_use_returns(s):
                    if window.size <= 1:
                        features[i, col] = np.nan
                        continue
                    window_data = compute_returns(window)
                    if window_data.size == 0:
                        features[i, col] = np.nan
                        continue
                else:
                    window_data = window
                
                try:
                    if s == "std":
                        features[i, col] = std_manual(window_data)
                    elif s == "skew":
                        features[i, col] = skew_manual(window_data)
                    elif s == "kurt":
                        features[i, col] = kurt_manual(window_data)
                    elif s == "zscore":
                        features[i, col] = zscore_manual(window_data)
                    elif s == "range":
                        features[i, col] = np.max(window_data) - np.min(window_data)
                    elif s == "mean":
                        features[i, col] = mean_manual(window_data)
                    elif s == "median":
                        features[i, col] = median_manual(window_data)
                    elif s == "iqr":
                        features[i, col] = iqr_manual(window_data)
                    elif s == "cv":
                        features[i, col] = coeff_var_manual(window_data)
                    elif s == "mad":
                        m = mean_manual(window_data)
                        features[i, col] = mean_manual(np.abs(window_data - m))
                    elif s == "entropy":
                        features[i, col] = entropy_manual(window_data)
                    elif s == "slope":
                        features[i, col] = slope_manual(window_data)
                    elif s == "momentum":
                        features[i, col] = momentum_roc(window_data)
                    elif s == "fractal":
                        features[i, col] = fractal_dimension_manual(window_data)
                    elif s == "hurst":
                        features[i, col] = hurst_manual(window_data)
                    elif s == "autocorr":
                        features[i, col] = autocorr1_manual(window_data)
                    elif s == "maxdd":
                        features[i, col] = max_dd_manual(window_data)
                    elif s == "sharpe":
                        features[i, col] = sharpe_manual(window_data)
                    elif s == "fisher":
                        features[i, col] = fisher_transform(momentum_roc(window_data))
                    elif s == "chande":
                        features[i, col] = chande_momentum(window_data)
                    elif s == "var":
                        std = std_manual(window_data)
                        features[i, col] = std * std * (window_data.size - 1) / window_data.size
                    elif s == "approxentropy":
                        features[i, col] = approximate_entropy(window_data)
                    elif s == "effratio":
                        features[i, col] = efficiency_ratio(window_data)
                    elif s == "corrskew":
                        features[i, col] = correlation_skew_manual(window_data)
                    elif s == "jumpvol":
                        features[i, col] = jump_volatility_manual(window_data)
                    elif s == "volskew":
                        features[i, col] = volatility_skew(window_data)
                except:
                    return np.full((n, total_features), np.nan)
            col += 1

    # Procesar períodos meta solo si existen
    if len(periods_meta) > 0 and len(stats_meta) > 0:
        for win in periods_meta:
            for s in stats_meta:
                # ───── AJUSTAR VENTANA PARA ESTADÍSTICOS CON RETORNOS ─────
                if should_use_returns(s):
                    window_size = win + 1  # +1 precio para obtener 'win' retornos
                    start_idx = window_size
                else:
                    window_size = win
                    start_idx = win
                    
                for i in range(start_idx, n):
                    # ───── OPTIMIZACIÓN: Usar slice directo ─────
                    window = close[i - window_size:i]
                    
                    # ───── NUEVA FUNCIONALIDAD: Usar retornos para estadísticos específicos ─────
                    if should_use_returns(s):
                        if window.size <= 1:
                            features[i, col] = np.nan
                            continue
                        window_data = compute_returns(window)
                        if window_data.size == 0:
                            features[i, col] = np.nan
                            continue
                    else:
                        window_data = window
                    
                    try:
                        if s == "std":
                            features[i, col] = std_manual(window_data)
                        elif s == "skew":
                            features[i, col] = skew_manual(window_data)
                        elif s == "kurt":
                            features[i, col] = kurt_manual(window_data)
                        elif s == "zscore":
                            features[i, col] = zscore_manual(window_data)
                        elif s == "range":
                            features[i, col] = np.max(window_data) - np.min(window_data)
                        elif s == "mean":
                            features[i, col] = mean_manual(window_data)
                        elif s == "median":
                            features[i, col] = median_manual(window_data)
                        elif s == "iqr":
                            features[i, col] = iqr_manual(window_data)
                        elif s == "cv":
                            features[i, col] = coeff_var_manual(window_data)
                        elif s == "mad":
                            m = mean_manual(window_data)
                            features[i, col] = mean_manual(np.abs(window_data - m))
                        elif s == "entropy":
                            features[i, col] = entropy_manual(window_data)
                        elif s == "slope":
                            features[i, col] = slope_manual(window_data)
                        elif s == "momentum":
                            features[i, col] = momentum_roc(window_data)
                        elif s == "fractal":
                            features[i, col] = fractal_dimension_manual(window_data)
                        elif s == "hurst":
                            features[i, col] = hurst_manual(window_data)
                        elif s == "autocorr":
                            features[i, col] = autocorr1_manual(window_data)
                        elif s == "maxdd":
                            features[i, col] = max_dd_manual(window_data)
                        elif s == "sharpe":
                            features[i, col] = sharpe_manual(window_data)
                        elif s == "fisher":
                            features[i, col] = fisher_transform(momentum_roc(window_data))
                        elif s == "chande":
                            features[i, col] = chande_momentum(window_data)
                        elif s == "var":
                            std = std_manual(window_data)
                            features[i, col] = std * std * (window_data.size - 1) / window_data.size
                        elif s == "approxentropy":
                            features[i, col] = approximate_entropy(window_data)
                        elif s == "effratio":
                            features[i, col] = efficiency_ratio(window_data)
                        elif s == "corrskew":
                            features[i, col] = correlation_skew_manual(window_data)
                        elif s == "jumpvol":
                            features[i, col] = jump_volatility_manual(window_data)
                        elif s == "volskew":
                            features[i, col] = volatility_skew(window_data)
                    except:
                        return np.full((n, total_features), np.nan)
                col += 1

    return features

def get_features(data: pd.DataFrame, hp):
    close = data['close'].values
    index = data.index
    periods_main = hp["feature_main_periods"]
    stats_main = hp["feature_main_stats"]
    
    # Obtener períodos y estadísticas meta, siempre como listas (vacías si no existen)
    periods_meta = hp.get("feature_meta_periods", [])
    stats_meta = hp.get("feature_meta_stats", [])
    
    if len(stats_main) == 0:
        raise ValueError("La lista de estadísticas MAIN está vacía.")
    
    # ───── OPTIMIZACIÓN: Asegurar contigüidad y evitar conversiones ─────
    if not close.flags.c_contiguous:
        close = np.ascontiguousarray(close)
    
    # ───── OPTIMIZACIÓN: Convertir listas una sola vez ─────
    periods_main_t = List(periods_main)
    stats_main_t = List(stats_main)
    periods_meta_t = List(periods_meta)
    stats_meta_t = List(stats_meta)

    feats = compute_features(close, periods_main_t, periods_meta_t, stats_main_t, stats_meta_t)
    if np.isnan(feats).all():
        return pd.DataFrame(index=index)
    
    # ───── OPTIMIZACIÓN: Generar nombres de columnas de forma más eficiente ─────
    colnames = []
    for p in periods_main:
        for s in stats_main:
            colnames.append(f"{p}_{s}_main_feature")
    
    # Agregar nombres de columnas meta solo si existen
    if len(periods_meta) > 0 and len(stats_meta) > 0:
        for p in periods_meta:
            for s in stats_meta:
                colnames.append(f"{p}_{s}_meta_feature")
    
    # ───── OPTIMIZACIÓN: Crear DataFrame de forma más eficiente ─────
    df = pd.DataFrame(feats, columns=colnames, index=index)

    # 1) añadir OHLCV **antes** de limpiar para que sufran el mismo filtrado
    ohlcv = data.loc[index, ["open", "high", "low", "close", "volume"]]
    df = pd.concat([df, ohlcv], axis=1)

    # 2) limpiar Inf/NaN SOLO en columnas de features
    feat_cols = df.filter(like='_main_feature').columns
    df[feat_cols] = df[feat_cols].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=feat_cols)

    # 3) verificación — el close debe coincidir al 100 %
    if not np.allclose(df["close"].values,
                    ohlcv.loc[df.index, "close"].values):
        raise RuntimeError("❌ 'close' desalineado tras generar features")

    return df

@njit(cache=True, fastmath=True)
def safe_savgol_filter(x, l_window_size: int, polyorder: int):
    """Apply Savitzky-Golay label_filter safely.

    Parameters
    ----------
    x : array-like
        Input array.
    l_window_size : int
        Desired window size.
    polyorder : int
        Polynomial order.

    Returns
    -------
    np.ndarray
        Smoothed array. If the input is too short, the original array is
        returned without filtering.
    """

    x = np.asarray(x)
    n = len(x)
    if n <= polyorder:
        return x

    wl = int(l_window_size)
    if wl % 2 == 0:
        wl += 1
    max_wl = n if n % 2 == 1 else n - 1
    wl = min(wl, max_wl)
    if wl <= polyorder:
        wl = polyorder + 1 if (polyorder + 1) % 2 == 1 else polyorder + 2
        wl = min(wl, max_wl)
        if wl <= polyorder:
            return x

    return savgol_filter(x, window_length=wl, polyorder=polyorder)

@njit(cache=True, fastmath=True)
def calculate_labels_trend(normalized_trend, label_threshold):
    labels = np.empty(len(normalized_trend), dtype=np.float64)
    for i in range(len(normalized_trend)):
        if normalized_trend[i] > label_threshold:
            labels[i] = 0.0  # Buy (Up trend)
        elif normalized_trend[i] < -label_threshold:
            labels[i] = 1.0  # Sell (Down trend)
        else:
            labels[i] = 2.0  # No signal
    return labels

def get_labels_trend(dataset, label_rolling=200, polyorder=3, label_threshold=0.5, vol_window=50) -> pd.DataFrame:
    smoothed_prices = safe_savgol_filter(dataset['close'].values, window_length=label_rolling, polyorder=polyorder)
    trend = np.gradient(smoothed_prices)
    vol = dataset['close'].label_rolling(vol_window).std().values
    normalized_trend = np.where(vol != 0, trend / vol, np.nan)  # Set NaN where vol is 0
    labels = calculate_labels_trend(normalized_trend, label_threshold)
    dataset = dataset.iloc[:len(labels)].copy()
    dataset['labels_main'] = labels
    dataset = dataset.dropna()  # Remove rows with NaN
    return dataset

def plot_trading_signals(
    dataset: pd.DataFrame,
    label_rolling: int = 200,
    polyorder: int = 3,
    label_threshold: float = 0.5,
    vol_window: int = 50,
    figsize: tuple = (14, 7)
) -> None:
    """
    Visualizes price data with calculated indicators and trading signals in one integrated plot.
    
    Args:
        dataset: DataFrame with 'close' prices and datetime index
        label_rolling: Window size for Savitzky-Golay label_filter. Default 200
        polyorder: Polynomial order for smoothing. Default 3
        label_threshold: Signal generation label_threshold. Default 0.5
        vol_window: Volatility calculation window. Default 50
        figsize: Figure dimensions. Default (14,7)
    """
    # Copy and clean data
    df = dataset[['close']].copy().dropna()
    close_prices = df['close'].values
    
    # 1. Smooth prices using Savitzky-Golay label_filter
    smoothed = safe_savgol_filter(close_prices, window_length=label_rolling, polyorder=polyorder)
    
    # 2. Calculate trend gradient
    trend = np.gradient(smoothed)
    
    # 3. Compute volatility (label_rolling std)
    vol = df['close'].label_rolling(vol_window).std().values
    
    # 4. Normalize trend by volatility
    normalized_trend = np.zeros_like(trend)
    valid_mask = vol != 0  # Filter zero-volatility periods
    normalized_trend[valid_mask] = trend[valid_mask] / vol[valid_mask]
    
    # 5. Generate trading signals
    labels = np.full(len(normalized_trend), 2.0, dtype=np.float64)  # Default 2.0 (no signal)
    labels[normalized_trend > label_threshold] = 0.0  # Buy signals
    labels[normalized_trend < -label_threshold] = 1.0  # Sell signals
    
    # 6. Calculate label_threshold bands
    upper_band = smoothed + label_threshold * vol
    lower_band = smoothed - label_threshold * vol
    
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
             label=f'Savitzky-Golay ({label_rolling},{polyorder})', 
             color='#ff7f0e', 
             lw=2)
    
    # Fill between label_threshold bands
    plt.fill_between(df.index, upper_band, lower_band,
                     color='#e0e0e0', 
                     alpha=0.3, 
                     label=f'Threshold ±{label_threshold}σ')
    
    # Plot buy/sell signals with distinct markers
    buy_dates = df.index[labels == 0.0]
    sell_dates = df.index[labels == 1.0]
    
    plt.scatter(buy_dates, df.loc[buy_dates, 'close'],
                color='#2ca02c', 
                marker='^', 
                s=80,
                edgecolor='black',
                label=f'Buy Signal (>+{label_threshold}σ)',
                zorder=5)
    
    plt.scatter(sell_dates, df.loc[sell_dates, 'close'],
                color='#d62728', 
                marker='v', 
                s=80,
                edgecolor='black',
                label=f'Sell Signal (<-{label_threshold}σ)',
                zorder=5)
    
    # Configure plot aesthetics
    plt.title(f'Trading Signals Generation\n(Smoothing: {label_rolling} periods, Threshold: {label_threshold}σ)')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.grid(alpha=0.2, linestyle='--')
    plt.legend(loc='upper left', framealpha=0.9)
    plt.tight_layout()
    plt.show()

@njit(cache=True, fastmath=True)
def calculate_labels_trend_with_profit(close, atr, normalized_trend, label_threshold, label_markup, label_min_val, label_max_val):
    labels = np.empty(len(normalized_trend) - label_max_val, dtype=np.float64)
    for i in range(len(normalized_trend) - label_max_val):
        dyn_mk = label_markup * atr[i]
        if normalized_trend[i] > label_threshold:
            # Проверяем condición para Buy
            rand = np.random.randint(label_min_val, label_max_val + 1)
            future_pr = close[i + rand]
            if future_pr >= close[i] + dyn_mk:
                labels[i] = 0.0  # Buy (Profit reached)
            else:
                labels[i] = 2.0  # No profit
        elif normalized_trend[i] < -label_threshold:
            # Проверяем condición para Sell
            rand = np.random.randint(label_min_val, label_max_val + 1)
            future_pr = close[i + rand]
            if future_pr <= close[i] - dyn_mk:
                labels[i] = 1.0  # Sell (Profit reached)
            else:
                labels[i] = 2.0  # No profit
        else:
            labels[i] = 2.0  # No signal
    return labels

def get_labels_trend_with_profit(dataset, label_rolling=200, polyorder=3, label_threshold=0.5, 
                    vol_window=50, label_markup=0.5, label_min_val=1, label_max_val=15, label_atr_period=14) -> pd.DataFrame:
    # Smoothing and trend calculation
    smoothed_prices = safe_savgol_filter(dataset['close'].values, window_length=label_rolling, polyorder=polyorder)
    trend = np.gradient(smoothed_prices)
    
    # Normalizing the trend by volatility
    vol = dataset['close'].label_rolling(vol_window).std().values
    normalized_trend = np.where(vol != 0, trend / vol, np.nan)
    
    # Removing NaN and synchronizing data
    valid_mask = ~np.isnan(normalized_trend)
    normalized_trend_clean = normalized_trend[valid_mask]
    close_clean = dataset['close'].values[valid_mask]
    high = dataset["high"].values if "high" in dataset else dataset["close"].values
    low = dataset["low"].values if "low" in dataset else dataset["close"].values
    atr = calculate_atr_simple(high, low, dataset["close"].values, period=label_atr_period)
    atr_clean = atr[valid_mask]
    dataset_clean = dataset[valid_mask].copy()
    
    # Generating labels
    labels = calculate_labels_trend_with_profit(close_clean, atr_clean, normalized_trend_clean, label_threshold, label_markup, label_min_val, label_max_val)
    
    # Trimming the dataset and adding labels
    dataset_clean = dataset_clean.iloc[:len(labels)].copy()
    dataset_clean['labels_main'] = labels[: len(dataset_clean)]
    
    # Filtering the results
    dataset_clean = dataset_clean.dropna()    
    return dataset_clean

@njit(cache=True, fastmath=True)
def calculate_labels_trend_different_filters(close, atr, normalized_trend, label_threshold, label_markup, label_min_val, label_max_val):
    labels = np.empty(len(normalized_trend) - label_max_val, dtype=np.float64)
    for i in range(len(normalized_trend) - label_max_val):
        dyn_mk = label_markup * atr[i]
        if normalized_trend[i] > label_threshold:
            # Проверяем condición para Buy
            rand = np.random.randint(label_min_val, label_max_val + 1)
            future_pr = close[i + rand]
            if future_pr >= close[i] + dyn_mk:
                labels[i] = 0.0  # Buy (Profit reached)
            else:
                labels[i] = 2.0  # No profit
        elif normalized_trend[i] < -label_threshold:
            # Проверяем condición para Sell
            rand = np.random.randint(label_min_val, label_max_val + 1)
            future_pr = close[i + rand]
            if future_pr <= close[i] - dyn_mk:
                labels[i] = 1.0  # Sell (Profit reached)
            else:
                labels[i] = 2.0  # No profit
        else:
            labels[i] = 2.0  # No signal
    return labels

def get_labels_trend_with_profit_different_filters(dataset, label_filter='savgol', label_rolling=200, polyorder=3, label_threshold=0.5, 
                    vol_window=50, label_markup=0.5, label_min_val=1, label_max_val=15, label_atr_period=14) -> pd.DataFrame:
    # Smoothing and trend calculation
    close_prices = dataset['close'].values
    if label_filter == 'savgol':
        smoothed_prices = safe_savgol_filter(close_prices, window_length=label_rolling, polyorder=polyorder)
    elif label_filter == 'spline':
        x = np.arange(len(close_prices))
        spline = UnivariateSpline(x, close_prices, k=polyorder, s=label_rolling)
        smoothed_prices = spline(x)
    elif label_filter == 'sma':
        smoothed_series = pd.Series(close_prices).label_rolling(window=label_rolling).mean()
        smoothed_prices = smoothed_series.values
    elif label_filter == 'ema':
        smoothed_series = pd.Series(close_prices).ewm(span=label_rolling, adjust=False).mean()
        smoothed_prices = smoothed_series.values
    else:
        raise ValueError(f"Unknown smoothing label_filter: {label_filter}")
    
    trend = np.gradient(smoothed_prices)
    
    # Normalizing the trend by volatility
    vol = dataset['close'].label_rolling(vol_window).std().values
    normalized_trend = np.where(vol != 0, trend / vol, np.nan)
    
    # Removing NaN and synchronizing data
    valid_mask = ~np.isnan(normalized_trend)
    normalized_trend_clean = normalized_trend[valid_mask]
    close_clean = dataset['close'].values[valid_mask]
    dataset_clean = dataset[valid_mask].copy()
    high = dataset["high"].values if "high" in dataset else dataset["close"].values
    low = dataset["low"].values if "low" in dataset else dataset["close"].values
    atr = calculate_atr_simple(high, low, dataset["close"].values, period=label_atr_period)
    atr_clean = atr[valid_mask]
    
    # Generating labels
    labels = calculate_labels_trend_different_filters(close_clean, atr_clean, normalized_trend_clean, label_threshold, label_markup, label_min_val, label_max_val)
    
    # Trimming the dataset and adding labels
    dataset_clean = dataset_clean.iloc[:len(labels)].copy()
    dataset_clean['labels_main'] = labels[: len(dataset_clean)]
    
    # Filtering the results
    dataset_clean = dataset_clean.dropna()    
    return dataset_clean

@njit(cache=True, fastmath=True)
def calculate_labels_trend_multi(close, atr, normalized_trends, label_threshold, label_markup, label_min_val, label_max_val):
    num_periods = normalized_trends.shape[0]  # Number of periods
    labels = np.empty(len(close) - label_max_val, dtype=np.float64)
    for i in range(len(close) - label_max_val):
        dyn_mk = label_markup * atr[i]
        # Select a random number of bars forward once for all periods
        rand = np.random.randint(label_min_val, label_max_val + 1)
        buy_signals = 0
        sell_signals = 0
        # Check conditions for each period
        for j in range(num_periods):
            if normalized_trends[j, i] > label_threshold:
                if close[i + rand] >= close[i] + dyn_mk:
                    buy_signals += 1
            elif normalized_trends[j, i] < -label_threshold:
                if close[i + rand] <= close[i] - dyn_mk:
                    sell_signals += 1
        # Combine signals
        if buy_signals > 0 and sell_signals == 0:
            labels[i] = 0.0  # Buy
        elif sell_signals > 0 and buy_signals == 0:
            labels[i] = 1.0  # Sell
        else:
            labels[i] = 2.0  # No signal or conflict
    return labels

def get_labels_trend_with_profit_multi(dataset, label_filter='savgol', rolling_periods=[10, 20, 30], polyorder=3, label_threshold=0.5, 
                                       vol_window=50, label_markup=0.5, label_min_val=1, label_max_val=15, label_atr_period=14) -> pd.DataFrame:
    """
    Generates labels for trading signals (Buy/Sell) based on the normalized trend,
    calculated for multiple smoothing periods.

    Args:
        dataset (pd.DataFrame): DataFrame with data, containing the 'close' column.
        label_filter (str): Smoothing label_filter ('savgol', 'spline', 'sma', 'ema').
        rolling_periods (list): List of smoothing window sizes. Default is [200].
        polyorder (int): Polynomial order for 'savgol' and 'spline' methods.
        label_threshold (float): Threshold for the normalized trend.
        vol_window (int): Window for volatility calculation.
        label_markup (float): Minimum profit to confirm the signal.
        label_min_val (int): Minimum number of bars forward.
        label_max_val (int): Maximum number of bars forward.

    Returns:
        pd.DataFrame: DataFrame with added 'labels_main' column:
                      - 0.0: Buy
                      - 1.0: Sell
                      - 2.0: No signal
    """
    close_prices = dataset['close'].values
    normalized_trends = []

    # Calculate normalized trend for each period
    for label_rolling in rolling_periods:
        if label_filter == 'savgol':
            smoothed_prices = safe_savgol_filter(close_prices, window_length=label_rolling, polyorder=polyorder)
        elif label_filter == 'spline':
            x = np.arange(len(close_prices))
            spline = UnivariateSpline(x, close_prices, k=polyorder, s=label_rolling)
            smoothed_prices = spline(x)
        elif label_filter == 'sma':
            smoothed_series = pd.Series(close_prices).label_rolling(window=label_rolling).mean()
            smoothed_prices = smoothed_series.values
        elif label_filter == 'ema':
            smoothed_series = pd.Series(close_prices).ewm(span=label_rolling, adjust=False).mean()
            smoothed_prices = smoothed_series.values
        else:
            raise ValueError(f"Unknown smoothing label_filter: {label_filter}")
        
        trend = np.gradient(smoothed_prices)
        vol = pd.Series(close_prices).label_rolling(vol_window).std().values
        normalized_trend = np.where(vol != 0, trend / vol, np.nan)
        normalized_trends.append(normalized_trend)

    # Transform list into 2D array
    normalized_trends_array = np.vstack(normalized_trends)

    # Remove rows with NaN
    valid_mask = ~np.isnan(normalized_trends_array).any(axis=0)
    normalized_trends_clean = normalized_trends_array[:, valid_mask]
    close_clean = close_prices[valid_mask]
    dataset_clean = dataset[valid_mask].copy()

    high = dataset["high"].values if "high" in dataset else dataset["close"].values
    low = dataset["low"].values if "low" in dataset else dataset["close"].values
    atr = calculate_atr_simple(high, low, dataset["close"].values, period=label_atr_period)
    atr_clean = atr[valid_mask]
    # Generate labels
    labels = calculate_labels_trend_multi(close_clean, atr_clean, normalized_trends_clean, label_threshold, label_markup, label_min_val, label_max_val)

    # Trim data and add labels
    dataset_clean = dataset_clean.iloc[:len(labels)].copy()
    dataset_clean['labels_main'] = labels[: len(dataset_clean)]

    # Remove remaining NaN
    dataset_clean = dataset_clean.dropna()
    return dataset_clean

@njit(cache=True, fastmath=True)
def calculate_labels_clusters(close_data, atr, clusters, label_markup):
    labels = []
    current_cluster = clusters[0]
    last_price = close_data[0]
    for i in range(1, len(close_data)):
        next_cluster = clusters[i]
        dyn_mk = label_markup * atr[i]
        if next_cluster != current_cluster and (abs(close_data[i] - last_price) > dyn_mk):
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

def get_labels_clusters(dataset, label_markup, l_n_clusters=20, label_atr_period=14) -> pd.DataFrame:
    kmeans = KMeans(n_clusters=l_n_clusters, n_init='auto')
    dataset['cluster'] = kmeans.fit_predict(dataset[['close']])

    close_data = dataset['close'].values
    clusters = dataset['cluster'].values

    high = dataset["high"].values if "high" in dataset else dataset["close"].values
    low = dataset["low"].values if "low" in dataset else dataset["close"].values
    atr = calculate_atr_simple(high, low, dataset["close"].values, period=label_atr_period)
    labels = calculate_labels_clusters(close_data, atr, clusters, label_markup)

    dataset['labels_main'] = labels
    dataset = dataset.drop(columns=['cluster'])
    return dataset

@njit(cache=True, fastmath=True)
def calculate_signals(prices, window_sizes, threshold_pct):
    max_window = max(window_sizes)
    signals = []
    for i in range(max_window, len(prices)):
        long_signals = 0
        short_signals = 0
        for l_window_size in window_sizes:
            window = prices[i-l_window_size:i]
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
    window_sizes_t = List(window_sizes)
    signals = calculate_signals(prices, window_sizes_t, threshold_pct)
    signals = [2.0] * max(window_sizes) + signals
    dataset = dataset.iloc[: len(signals)].copy()
    dataset['labels_main'] = signals[: len(dataset)]
    return dataset

@njit(cache=True, fastmath=True)
def calculate_labels_validated_levels(prices, l_window_size, threshold_pct, min_touches):
    resistance_touches = {}
    support_touches = {}
    labels = []
    for i in range(l_window_size, len(prices)):
        window = prices[i-l_window_size:i]
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

def get_labels_validated_levels(dataset, l_window_size=20, threshold_pct=0.02, min_touches=2) -> pd.DataFrame:
    prices = dataset['close'].values
    
    labels = calculate_labels_validated_levels(prices, l_window_size, threshold_pct, min_touches)
    
    labels = [2.0] * l_window_size + labels
    dataset['labels_main'] = labels
    return dataset

@njit(cache=True, fastmath=True)
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
                                           used to label_filter out insignificant fluctuations. 
                                           Defaults to 0.1.

    Returns:
        pd.DataFrame: The original DataFrame with a new 'labels_main' column and filtered rows:
                       - 'labels_main' column: 
                            - 0: Buy
                            - 1: Sell
                       - Rows with missing values (NaN) are removed.
    """

    # Find peaks and troughs in the closing prices
    peaks, _ = find_peaks(dataset['close'], prominence=peak_prominence)
    troughs, _ = find_peaks(-dataset['close'], prominence=peak_prominence)
    
    # Calculate buy/sell labels using the new zigzag-based labeling function
    labels = calculate_labels_zigzag(peaks, troughs, len(dataset)) 

    # Add the calculated labels as a new 'labels_main' column to the DataFrame
    dataset['labels_main'] = labels
        
    # Return the modified DataFrame 
    return dataset

# MEAN REVERSION WITH RESTRICTIONS BASED LABELING
@njit(cache=True, fastmath=True)
def calculate_labels_mean_reversion(close, atr, lvl, label_markup, label_min_val, label_max_val, q):
    labels = np.empty(len(close) - label_max_val, dtype=np.float64)
    for i in range(len(close) - label_max_val):
        dyn_mk = label_markup * atr[i]
        rand = np.random.randint(label_min_val, label_max_val + 1)
        curr_pr = close[i]
        curr_lvl = lvl[i]
        future_pr = close[i + rand]

        if curr_lvl > q[1] and (future_pr + dyn_mk) < curr_pr:
            labels[i] = 1.0
        elif curr_lvl < q[0] and (future_pr - dyn_mk) > curr_pr:
            labels[i] = 0.0
        else:
            labels[i] = 2.0
    return labels

def get_labels_mean_reversion(dataset, label_markup, label_min_val=1, label_max_val=15, label_rolling=0.5, quantiles=[.45, .55], label_filter='spline', decay_factor=0.95, shift=0, label_atr_period=14) -> pd.DataFrame:
    """
    Generates labels for a financial dataset based on mean reversion principles.

    This function calculates trading signals (buy/sell) based on the deviation of
    the price from a chosen moving average or smoothing label_filter. It identifies
    potential buy opportunities when the price deviates significantly below its 
    smoothed trend, anticipating a reversion to the mean.

    Args:
        dataset (pd.DataFrame): DataFrame containing financial data with a 'close' column.
        label_markup (float): The percentage label_markup used to determine buy signals.
        label_min_val (int, optional): Minimum number of consecutive days the label_markup must hold. Defaults to 1.
        label_max_val (int, optional): Maximum number of consecutive days the label_markup is considered. Defaults to 15.
        label_rolling (float, optional): Rolling window size for smoothing/averaging. 
                                     If label_filter='spline', this controls the spline smoothing factor.
                                     Defaults to 0.5.
        quantiles (list, optional): Quantiles to define the "reversion zone". Defaults to [.45, .55].
        label_filter (str, optional): Method for calculating the price deviation:
                                 - 'mean': Deviation from the label_rolling mean.
                                 - 'spline': Deviation from a smoothed spline.
                                 - 'savgol': Deviation from a Savitzky-Golay label_filter.
                                 Defaults to 'spline'.
        shift (int, optional): Shift the smoothed price data forward (positive) or backward (negative).
                                 Useful for creating a lag/lead effect. Defaults to 0.

    Returns:
        pd.DataFrame: The original DataFrame with a new 'labels_main' column and filtered rows:
                       - 'labels_main' column: 
                            - 0: Buy
                            - 1: Sell
                       - Rows with missing values (NaN) are removed.
                       - The temporary 'lvl' column is removed. 
    """

    # Calculate the price deviation ('lvl') based on the chosen label_filter
    if label_filter == 'mean':
        diff = (dataset['close'] - dataset['close'].label_rolling(label_rolling).mean())
        weighted_diff = diff * np.exp(np.arange(len(diff)) * decay_factor / len(diff)) 
        dataset['lvl'] = weighted_diff # Add the weighted difference as 'lvl'
    elif label_filter == 'spline':
        x = np.array(range(dataset.shape[0]))
        y = dataset['close'].values
        spl = UnivariateSpline(x, y, k=3, s=label_rolling) 
        yHat = spl(np.linspace(min(x), max(x), num=x.shape[0]))
        yHat_shifted = np.roll(yHat, shift=shift) # Apply the shift
        diff = dataset['close'] - yHat_shifted
        weighted_diff = diff * np.exp(np.arange(len(diff)) * decay_factor / len(diff)) 
        dataset['lvl'] = weighted_diff # Add the weighted difference as 'lvl'
        dataset = dataset.dropna()  # Remove NaN values potentially introduced by spline/shift
    elif label_filter == 'savgol':
        smoothed_prices = safe_savgol_filter(dataset['close'].values, window_length=int(label_rolling), polyorder=3)
        diff = dataset['close'] - smoothed_prices
        weighted_diff = diff * np.exp(np.arange(len(diff)) * decay_factor / len(diff)) 
        dataset['lvl'] = weighted_diff # Add the weighted difference as 'lvl'

    dataset = dataset.dropna()  # Remove NaN values before proceeding

    # Ensure label_max_val does not exceed dataset length
    label_max_val = min(int(label_max_val), max(len(dataset) - 1, 1))
    if len(dataset) <= label_max_val:
        return pd.DataFrame()
    q = tuple(dataset['lvl'].quantile(quantiles).to_list())  # Calculate quantiles for the 'reversion zone'

    # Prepare data for label calculation
    close = dataset['close'].values
    lvl = dataset['lvl'].values
    
    # Calculate buy/sell labels 
    high = dataset["high"].values if "high" in dataset else close
    low = dataset["low"].values if "low" in dataset else close
    atr = calculate_atr_simple(high, low, close, period=label_atr_period)
    labels = calculate_labels_mean_reversion(close, atr, lvl, label_markup, label_min_val, label_max_val, q)

    # Process the dataset and labels
    dataset = dataset.iloc[:len(labels)].copy()
    dataset['labels_main'] = labels
    dataset = dataset.dropna()
    return dataset.drop(columns=['lvl'])  # Remove the temporary 'lvl' column 

@njit(cache=True, fastmath=True)
def calculate_labels_mean_reversion_multi(close_data, atr, lvl_data, q_list, label_markup, label_min_val, label_max_val, window_sizes):
    labels = []
    n_win = len(window_sizes)
    for i in range(len(close_data) - label_max_val):
        dyn_mk = label_markup * atr[i]
        rand = np.random.randint(label_min_val, label_max_val + 1)
        curr_pr = close_data[i]
        future_pr = close_data[i + rand]

        buy_condition = True
        sell_condition = True
        for qq in range(n_win):
            curr_lvl = lvl_data[i, qq]
            q_low, q_high = q_list[qq]
            if curr_lvl >= q_high:
                pass
            else:
                sell_condition = False
            if curr_lvl <= q_low:
                pass
            else:
                buy_condition = False

        if sell_condition and (future_pr + dyn_mk) < curr_pr:
            labels.append(1.0)
        elif buy_condition and (future_pr - dyn_mk) > curr_pr:
            labels.append(0.0)
        else:
            labels.append(2.0)
    return labels

def get_labels_mean_reversion_multi(dataset, label_markup, label_min_val=1, label_max_val=15, window_sizes=[0.2, 0.3, 0.5], quantiles=[.45, .55], label_atr_period=14) -> pd.DataFrame:
    q = np.empty((len(window_sizes), 2))  # Initialize as 2D NumPy array
    lvl_data = np.empty((dataset.shape[0], len(window_sizes)))

    for i, label_rolling in enumerate(window_sizes):
        x = np.arange(dataset.shape[0])
        y = dataset['close'].values
        spl = UnivariateSpline(x, y, k=3, s=label_rolling)
        yHat = spl(np.linspace(x.min(), x.max(), x.shape[0]))
        lvl_data[:, i] = dataset['close'] - yHat
        # Store quantiles directly into the NumPy array
        quantile_values = np.quantile(lvl_data[:, i], quantiles)
        q[i, 0] = quantile_values[0]
        q[i, 1] = quantile_values[1]

    dataset = dataset.dropna()
    label_max_val = min(int(label_max_val), max(len(dataset) - 1, 1))
    if len(dataset) <= label_max_val:
        return pd.DataFrame()
    close_data = dataset['close'].values

    high = dataset["high"].values if "high" in dataset else close_data
    low = dataset["low"].values if "low" in dataset else close_data
    atr = calculate_atr_simple(high, low, close_data, period=label_atr_period)

    # Convert parameters to Numba typed.List to avoid repeated JIT compilations
    windows_t = List(window_sizes)
    q_t = List([(float(q[i,0]), float(q[i,1])) for i in range(len(window_sizes))])
    labels = calculate_labels_mean_reversion_multi(close_data, atr, lvl_data, q_t, label_markup, label_min_val, label_max_val, windows_t)

    dataset = dataset.iloc[:len(labels)].copy()
    dataset['labels_main'] = labels
    dataset = dataset.dropna()
    
    return dataset

@njit(cache=True, fastmath=True)
def calculate_labels_mean_reversion_v(close_data, atr, lvl_data, volatility_group, quantile_groups_low, quantile_groups_high, label_markup, label_min_val, label_max_val):
    labels = []
    for i in range(len(close_data) - label_max_val):
        dyn_mk = label_markup * atr[i]
        rand = np.random.randint(label_min_val, label_max_val + 1)
        curr_pr = close_data[i]
        curr_lvl = lvl_data[i]
        curr_vol_group = volatility_group[i]
        future_pr = close_data[i + rand]

        # Access quantiles directly from arrays
        low_q = quantile_groups_low[int(curr_vol_group)]
        high_q = quantile_groups_high[int(curr_vol_group)]

        if curr_lvl > high_q and (future_pr + dyn_mk) < curr_pr:
            labels.append(1.0)
        elif curr_lvl < low_q and (future_pr - dyn_mk) > curr_pr:
            labels.append(0.0)
        else:
            labels.append(2.0)
    return labels

def get_labels_mean_reversion_v(dataset, label_markup, label_min_val=1, label_max_val=15, label_rolling=0.5, quantiles=[.45, .55], label_filter='spline', shift=1, vol_window=20, label_atr_period=14) -> pd.DataFrame:
    """
    Generates trading labels based on mean reversion principles, incorporating
    volatility-based adjustments to identify buy opportunities.

    This function calculates trading signals (buy/sell), taking into account the 
    volatility of the asset. It groups the data into volatility bands and calculates 
    quantiles for each band. This allows for more dynamic "reversion zones" that 
    adjust to changing market conditions.

    Args:
        dataset (pd.DataFrame): DataFrame containing financial data with a 'close' column.
        label_markup (float): The percentage label_markup used to determine buy signals.
        label_min_val (int, optional): Minimum number of consecutive days the label_markup must hold. Defaults to 1.
        label_max_val (int, optional): Maximum number of consecutive days the label_markup is considered. Defaults to 15.
        label_rolling (float, optional): Rolling window size or spline smoothing factor (see 'label_filter'). 
                                     Defaults to 0.5.
        quantiles (list, optional): Quantiles to define the "reversion zone". Defaults to [.45, .55].
        label_filter (str, optional): Method for calculating the price deviation:
                                 - 'mean': Deviation from the label_rolling mean.
                                 - 'spline': Deviation from a smoothed spline.
                                 - 'savgol': Deviation from a Savitzky-Golay label_filter.
                                 Defaults to 'spline'.
        shift (int, optional): Shift the smoothed price data forward (positive) or backward (negative).
                                 Useful for creating a lag/lead effect. Defaults to 1.
        vol_window (int, optional): Window size for calculating volatility. Defaults to 20.

    Returns:
        pd.DataFrame: The original DataFrame with a new 'labels_main' column and filtered rows:
                       - 'labels_main' column: 
                            - 0: Buy
                            - 1: Sell
                       - Rows with missing values (NaN) are removed.
                       - Temporary 'lvl', 'volatility', 'volatility_group' columns are removed.
    """

    # Calculate Volatility
    dataset['volatility'] = dataset['close'].pct_change().label_rolling(window=vol_window).std()
    vol = dataset['volatility'].dropna()
    if vol.nunique() < 2:
        # No se puede hacer qcut, todos los valores son iguales
        return pd.DataFrame()
    # Divide into 20 groups by volatility 
    dataset['volatility_group'] = pd.qcut(dataset['volatility'], q=20, labels=False, duplicates='drop')
    
    # Calculate price deviation ('lvl') based on the chosen label_filter
    if label_filter == 'mean':
        dataset['lvl'] = (dataset['close'] - dataset['close'].label_rolling(label_rolling).mean())
    elif label_filter == 'spline':
        x = np.array(range(dataset.shape[0]))
        y = dataset['close'].values
        spl = UnivariateSpline(x, y, k=3, s=label_rolling)
        yHat = spl(np.linspace(min(x), max(x), num=x.shape[0]))
        yHat_shifted = np.roll(yHat, shift=shift) # Apply the shift 
        dataset['lvl'] = dataset['close'] - yHat_shifted
        dataset = dataset.dropna() 
    elif label_filter == 'savgol':
        smoothed_prices = safe_savgol_filter(dataset['close'].values, window_length=label_rolling, polyorder=5)
        dataset['lvl'] = dataset['close'] - smoothed_prices

    dataset = dataset.dropna()

    # Ensure label_max_val does not exceed dataset length
    label_max_val = min(int(label_max_val), max(len(dataset) - 1, 1))
    if len(dataset) <= label_max_val:
        return pd.DataFrame()

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
    high = dataset["high"].values if "high" in dataset else close_data
    low = dataset["low"].values if "low" in dataset else close_data
    atr = calculate_atr_simple(high, low, close_data, period=label_atr_period)
    quantile_groups_high = np.array(quantile_groups_high)

    # Calculate buy/sell labels
    labels = calculate_labels_mean_reversion_v(close_data, atr, lvl_data, volatility_group, quantile_groups_low, quantile_groups_high, label_markup, label_min_val, label_max_val)
    
    # Process dataset and labels
    dataset = dataset.iloc[:len(labels)].copy()
    dataset['labels_main'] = labels
    dataset = dataset.dropna()
    
    # Remove temporary columns and return
    return dataset.drop(columns=['lvl', 'volatility', 'volatility_group'])

# FILTERING BASED LABELING W/O RESTRICTIONS
@njit(cache=True, fastmath=True)
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

def get_labels_filter(dataset, label_rolling=200, quantiles=[.45, .55], polyorder=3, decay_factor=0.95) -> pd.DataFrame:
    """
    Generates labels for a financial dataset based on price deviation from a Savitzky-Golay label_filter,
    with exponential weighting applied to prioritize recent data. Optionally incorporates a 
    cyclical component to the price deviation.

    Args:
        dataset (pd.DataFrame): DataFrame containing financial data with a 'close' column.
        label_rolling (int, optional): Window size for the Savitzky-Golay label_filter. Defaults to 200.
        quantiles (list, optional): Quantiles to define the "reversion zone". Defaults to [.45, .55].
        polyorder (int, optional): Polynomial order for the Savitzky-Golay label_filter. Defaults to 3.
        decay_factor (float, optional): Exponential decay factor for weighting past data. 
                                        Lower values prioritize recent data more. Defaults to 0.95.
        cycle_period (int, optional): Period of the cycle in number of data points. If None, 
                                     no cycle is applied. Defaults to None.
        cycle_amplitude (float, optional): Amplitude of the cycle. If None, no cycle is applied. 
                                          Defaults to None.

    Returns:
        pd.DataFrame: The original DataFrame with a new 'labels_main' column and filtered rows:
                       - 'labels_main' column: 
                            - 0: Buy
                            - 1: Sell
                       - Rows with missing values (NaN) are removed.
                       - The temporary 'lvl' column is removed. 
    """

    # Calculate smoothed prices using the Savitzky-Golay label_filter
    smoothed_prices = safe_savgol_filter(dataset['close'].values, window_length=label_rolling, polyorder=polyorder)
    
    # Calculate the difference between the actual closing prices and the smoothed prices
    diff = dataset['close'] - smoothed_prices
    
    # Apply exponential weighting to the 'diff' values
    weighted_diff = diff * np.exp(np.arange(len(diff)) * decay_factor / len(diff)) 
    
    dataset['lvl'] = weighted_diff # Add the weighted difference as 'lvl'

    # Remove any rows with NaN values 
    dataset = dataset.dropna()
    
    # Calculate the quantiles of the 'lvl' column (price deviation)
    q = tuple(dataset['lvl'].quantile(quantiles).to_list())

    # Extract the closing prices and the calculated 'lvl' values as NumPy arrays
    close = dataset['close'].values
    lvl = dataset['lvl'].values
    
    # Calculate buy/sell labels using the 'calculate_labels_filter' function 
    labels = calculate_labels_filter(close, lvl, q) 

    # Trim the dataset to match the length of the calculated labels
    dataset = dataset.iloc[:len(labels)].copy()
    
    # Add the calculated labels as a new 'labels_main' column to the DataFrame
    dataset['labels_main'] = labels
    
    # Remove any rows with NaN values
    dataset = dataset.dropna()
    
    # Return the modified DataFrame with the 'lvl' column removed
    return dataset.drop(columns=['lvl']) 

@njit(cache=True, fastmath=True)
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

def get_labels_multiple_filters(dataset, rolling_periods=[200, 400, 600], quantiles=[.45, .55], l_window_size=100, polyorder=3) -> pd.DataFrame:
    """
    Generates trading signals (buy/sell) based on price deviation from multiple 
    smoothed price trends calculated using a Savitzky-Golay label_filter with different
    label_rolling periods and label_rolling quantiles. 

    This function applies a Savitzky-Golay label_filter to the closing prices for each 
    specified 'rolling_period'. It then calculates the price deviation from these
    smoothed trends and determines dynamic "reversion zones" using label_rolling quantiles.
    Buy signals are generated when the price is within these reversion zones 
    across multiple timeframes.

    Args:
        dataset (pd.DataFrame): DataFrame containing financial data with a 'close' column.
        rolling_periods (list, optional): List of label_rolling l_window_size sizes for the Savitzky-Golay label_filter. 
                                           Defaults to [200, 400, 600].
        quantiles (list, optional): Quantiles to define the "reversion zone". Defaults to [.45, .55].
        l_window_size (int, optional): Window size for calculating label_rolling quantiles. Defaults to 100.
        polyorder (int, optional): Polynomial order for the Savitzky-Golay label_filter. Defaults to 3.

    Returns:
        pd.DataFrame: The original DataFrame with a new 'labels_main' column and filtered rows:
                       - 'labels_main' column: 
                            - 0: Buy
                            - 1: Sell
                       - Rows with missing values (NaN) are removed. 
    """
    
    # Create a copy of the dataset to avoid modifying the original
    dataset = dataset.copy()
    
    # Lists to store price deviation levels and quantiles for each label_rolling period
    all_levels = []
    all_quantiles = []
    
    # Calculate smoothed price trends and label_rolling quantiles for each label_rolling period
    for label_rolling in rolling_periods:
        # Calculate smoothed prices using the Savitzky-Golay label_filter
        smoothed_prices = safe_savgol_filter(dataset['close'].values,
                                      window_length=label_rolling,
                                      polyorder=polyorder)
        # Calculate the price deviation from the smoothed prices
        diff = dataset['close'] - smoothed_prices
        
        # Create a temporary DataFrame to calculate label_rolling quantiles
        temp_df = pd.DataFrame({'diff': diff})
        
        # Calculate label_rolling quantiles for the price deviation
        q_low = temp_df['diff'].label_rolling(window=l_window_size).quantile(quantiles[0])
        q_high = temp_df['diff'].label_rolling(window=l_window_size).quantile(quantiles[1])
        
        # Store the price deviation and quantiles for the current label_rolling period
        all_levels.append(diff)
        all_quantiles.append([q_low.values, q_high.values])
    
    # Convert lists to NumPy arrays for faster calculations (potentially using Numba)
    lvls_array = np.array(all_levels)
    qs_array = np.array(all_quantiles)

    # Calculate buy/sell labels using the 'calc_labels_multiple_filters' function 
    labels = calc_labels_multiple_filters(dataset['close'].values, lvls_array, qs_array)
    
    # Add the calculated labels to the DataFrame
    dataset['labels_main'] = labels
    
    # Remove rows with NaN values
    dataset = dataset.dropna()
        
    # Return the DataFrame with the new 'labels_main' column
    return dataset

@njit(cache=True, fastmath=True)
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
        rolling1 (int, optional): Window size for the first Savitzky-Golay label_filter. Defaults to 200.
        rolling2 (int, optional): Window size for the second Savitzky-Golay label_filter. Defaults to 200.
        quantiles (list, optional): Quantiles to define the "reversion zones". Defaults to [.45, .55].
        polyorder (int, optional): Polynomial order for both Savitzky-Golay filters. Defaults to 3.

    Returns:
        pd.DataFrame: The original DataFrame with a new 'labels_main' column and filtered rows:
                       - 'labels_main' column: 
                            - 0: Buy
                            - 1: Sell
                       - Rows with missing values (NaN) are removed.
                       - Temporary 'lvl1' and 'lvl2' columns are removed.
    """

    # Apply the first Savitzky-Golay label_filter (forward direction)
    smoothed_prices = safe_savgol_filter(dataset['close'].values, window_length=rolling1, polyorder=polyorder)
    
    # Apply the second Savitzky-Golay label_filter (could be in reverse direction if rolling2 is negative)
    smoothed_prices2 = safe_savgol_filter(dataset['close'].values, window_length=rolling2, polyorder=polyorder)

    # Calculate price deviations from both smoothed price series
    diff1 = dataset['close'] - smoothed_prices
    diff2 = dataset['close'] - smoothed_prices2

    # Add price deviations as new columns to the DataFrame
    dataset['lvl1'] = diff1
    dataset['lvl2'] = diff2
    
    # Remove rows with NaN values 
    dataset = dataset.dropna()

    # Calculate quantiles for the "reversion zones" for both price deviation series
    q1 = tuple(dataset['lvl1'].quantile(quantiles).to_list())
    q2 = tuple(dataset['lvl2'].quantile(quantiles).to_list())

    # Extract relevant data for label calculation
    close = dataset['close'].values
    lvl1 = dataset['lvl1'].values
    lvl2 = dataset['lvl2'].values
    
    # Calculate buy/sell labels using the 'calc_labels_bidirectional' function
    labels = calc_labels_bidirectional(close, lvl1, lvl2, q1, q2)

    # Process the dataset and labels
    dataset = dataset.iloc[:len(labels)].copy()
    dataset['labels_main'] = labels
    dataset = dataset.dropna()
    
    # Return the DataFrame with temporary columns removed
    return dataset.drop(columns=['lvl1', 'lvl2']) 

@njit(cache=True, fastmath=True)
def calculate_labels_filter_one_direction(close, lvl, q, direction_int):
    labels = np.empty(len(close), dtype=np.float64)
    for i in range(len(close)):
        curr_lvl = lvl[i]
        if direction_int == 1:  # sell
            if curr_lvl > q[1]:
                labels[i] = 1.0
            else:
                labels[i] = 0.0
        if direction_int == 0:  # buy
            if curr_lvl < q[0]:
                labels[i] = 1.0
            else:
                labels[i] = 0.0
    return labels

def get_labels_filter_one_direction(dataset, label_rolling=200, quantiles=[.45, .55], polyorder=3, direction='buy') -> pd.DataFrame:
    smoothed_prices = safe_savgol_filter(dataset['close'].values, window_length=label_rolling, polyorder=polyorder)
    diff = dataset['close'] - smoothed_prices
    dataset['lvl'] = diff
    dataset = dataset.dropna()
    q = tuple(dataset['lvl'].quantile(quantiles).to_list())
    close = dataset['close'].values
    lvl = dataset['lvl'].values
    direction_map = {'buy': 0, 'sell': 1}
    if direction in {'buy', 'sell'}:
        direction_int = direction_map.get(direction, 0)
        labels = calculate_labels_filter_one_direction(close, lvl, q, direction_int)
        dataset = dataset.iloc[:len(labels)].copy()
        dataset['labels_main'] = labels
        dataset = dataset.dropna()
        return dataset.drop(columns=['lvl'])
    if direction == 'both':
        labels_buy = calculate_labels_filter_one_direction(close, lvl, q, 0)
        labels_sell = calculate_labels_filter_one_direction(close, lvl, q, 1)
        n = min(len(labels_buy), len(labels_sell))
        labels = np.full(n, 2.0, dtype=np.float64)
        buy_sig = labels_buy[:n] == 1.0
        sell_sig = labels_sell[:n] == 1.0
        labels[buy_sig & ~sell_sig] = 0.0
        labels[sell_sig & ~buy_sig] = 1.0
        dataset = dataset.iloc[:n].copy()
        dataset['labels_main'] = labels
        dataset = dataset.dropna()
        return dataset.drop(columns=['lvl'])
    raise ValueError("direction must be 'buy', 'sell', or 'both'")

@njit(cache=True, fastmath=True)
def calculate_labels_trend_one_direction(normalized_trend, label_threshold, direction_int):
    labels = np.empty(len(normalized_trend), dtype=np.float64)
    for i in range(len(normalized_trend)):
        if direction_int == 0:  # buy
            if normalized_trend[i] > label_threshold:
                labels[i] = 1.0  # Buy (Up trend)
            else:
                labels[i] = 0.0
        if direction_int == 1:  # sell
            if normalized_trend[i] < -label_threshold:
                labels[i] = 1.0  # Sell (Down trend)
            else:
                labels[i] = 0.0  # No signal
    return labels

def get_labels_trend_one_direction(dataset, label_rolling=50, polyorder=3, label_threshold=0.001, vol_window=50, direction='buy') -> pd.DataFrame:
    smoothed_prices = safe_savgol_filter(dataset['close'].values, window_length=label_rolling, polyorder=polyorder)
    trend = np.gradient(smoothed_prices)
    vol = dataset['close'].label_rolling(vol_window).std().values
    normalized_trend = np.where(vol != 0, trend / vol, np.nan)
    direction_map = {'buy': 0, 'sell': 1}
    if direction in {'buy', 'sell'}:
        direction_int = direction_map.get(direction, 0)
        labels = calculate_labels_trend_one_direction(normalized_trend, label_threshold, direction_int)
        dataset = dataset.iloc[:len(labels)].copy()
        dataset['labels_main'] = labels
        dataset = dataset.dropna()
        return dataset
    if direction == 'both':
        labels_buy = calculate_labels_trend_one_direction(normalized_trend, label_threshold, 0)
        labels_sell = calculate_labels_trend_one_direction(normalized_trend, label_threshold, 1)
        n = min(len(labels_buy), len(labels_sell))
        labels = np.full(n, 2.0, dtype=np.float64)
        buy_sig = labels_buy[:n] == 1.0
        sell_sig = labels_sell[:n] == 1.0
        labels[buy_sig & ~sell_sig] = 0.0
        labels[sell_sig & ~buy_sig] = 1.0
        dataset = dataset.iloc[:n].copy()
        dataset['labels_main'] = labels
        dataset = dataset.dropna()
        return dataset
    raise ValueError("direction must be 'buy', 'sell', or 'both'")

@njit(cache=True, fastmath=True)
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

def get_labels_filter_flat(dataset, label_rolling=200, quantiles=[.45, .55], polyorder=3, decay_factor=0.95) -> pd.DataFrame:
    """
    Generates labels for a financial dataset based on price deviation from a Savitzky-Golay label_filter,
    with exponential weighting applied to prioritize recent data. Optionally incorporates a 
    cyclical component to the price deviation.

    Args:
        dataset (pd.DataFrame): DataFrame containing financial data with a 'close' column.
        label_rolling (int, optional): Window size for the Savitzky-Golay label_filter. Defaults to 200.
        quantiles (list, optional): Quantiles to define the "reversion zone". Defaults to [.45, .55].
        polyorder (int, optional): Polynomial order for the Savitzky-Golay label_filter. Defaults to 3.
        decay_factor (float, optional): Exponential decay factor for weighting past data. 
                                        Lower values prioritize recent data more. Defaults to 0.95.
        cycle_period (int, optional): Period of the cycle in number of data points. If None, 
                                     no cycle is applied. Defaults to None.
        cycle_amplitude (float, optional): Amplitude of the cycle. If None, no cycle is applied. 
                                          Defaults to None.

    Returns:
        pd.DataFrame: The original DataFrame with a new 'labels_main' column and filtered rows:
                       - 'labels_main' column: 
                            - 0: Buy
                            - 1: Sell
                       - Rows with missing values (NaN) are removed.
                       - The temporary 'lvl' column is removed. 
    """

    # Calculate smoothed prices using the Savitzky-Golay label_filter
    smoothed_prices = safe_savgol_filter(dataset['close'].values, window_length=label_rolling, polyorder=polyorder)
    
    # Calculate the difference between the actual closing prices and the smoothed prices
    diff = dataset['close'] - smoothed_prices
    
    # Apply exponential weighting to the 'diff' values
    weighted_diff = diff * np.exp(np.arange(len(diff)) * decay_factor / len(diff)) 
    
    dataset['lvl'] = 1/weighted_diff # Add the weighted difference as 'lvl'

    # Remove any rows with NaN values 
    dataset = dataset.dropna()
    
    # Calculate the quantiles of the 'lvl' column (price deviation)
    q = tuple(dataset['lvl'].quantile(quantiles).to_list())

    # Extract the closing prices and the calculated 'lvl' values as NumPy arrays
    close = dataset['close'].values
    lvl = dataset['lvl'].values
    
    # Calculate buy/sell labels using the 'calculate_labels_filter' function 
    labels = calculate_labels_filter_flat(close, lvl, q) 

    # Trim the dataset to match the length of the calculated labels
    dataset = dataset.iloc[:len(labels)].copy()
    
    # Add the calculated labels as a new 'labels_main' column to the DataFrame
    dataset['labels_main'] = labels
    
    # Remove any rows with NaN values
    dataset = dataset.dropna()

    # Return the modified DataFrame with the 'lvl' column removed
    return dataset.drop(columns=['lvl'])

@njit(cache=True, fastmath=True)
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
@njit(cache=True, fastmath=True)
def calculate_labels_one_direction(high, low, close, label_markup, label_min_val, label_max_val, direction_int, label_atr_period=14, method_int=5):
    n = len(close)
    if n <= label_max_val:
        return np.zeros(0, dtype=np.float64)
    atr = calculate_atr_simple(high, low, close, period=label_atr_period)
    result = np.zeros(n - label_max_val, dtype=np.float64)
    for i in range(n - label_max_val):
        window = close[i + label_min_val : i + label_max_val + 1]
        if window.size == 0:
            continue
        if method_int == 0:  # first
            future_price = close[i + label_min_val]
        elif method_int == 1:  # last
            future_price = close[i + label_max_val]
        elif method_int == 2:  # mean
            future_price = np.mean(window)
        elif method_int == 3:  # max
            future_price = np.max(window)
        elif method_int == 4:  # min
            future_price = np.min(window)
        else:  # random/otro
            rand = np.random.randint(label_min_val, label_max_val + 1)
            future_price = close[i + rand]
        dyn_mk = label_markup * atr[i]
        if direction_int == 0:  # buy
            if future_price > close[i] + dyn_mk:
                result[i] = 1.0
        elif direction_int == 1:  # sell
            if future_price < close[i] - dyn_mk:
                result[i] = 1.0
    return result

def get_labels_one_direction(dataset, label_markup=0.5, label_min_val=1, label_max_val=5,
                             direction='buy', label_atr_period=14, label_method='random') -> pd.DataFrame:
    close_data = np.ascontiguousarray(dataset['close'].values)
    high_data = np.ascontiguousarray(dataset['high'].values)
    low_data = np.ascontiguousarray(dataset['low'].values)
    direction_map = {'buy': 0, 'sell': 1}
    method_map = {'first': 0, 'last': 1, 'mean': 2, 'max': 3, 'min': 4, 'random': 5}
    if direction in {'buy', 'sell'}:
        direction_int = direction_map.get(direction, 0)
        method_int = method_map.get(label_method, 5)
        labels = calculate_labels_one_direction(
            high_data, low_data, close_data,
            label_markup, label_min_val, label_max_val, direction_int,
            label_atr_period, method_int
        )
        dataset = dataset.iloc[:len(labels)].copy()
        dataset['labels_main'] = labels
        dataset = dataset.dropna()
        return dataset
    if direction == 'both':
        method_int = method_map.get(label_method, 5)
        labels_buy = calculate_labels_one_direction(
            high_data, low_data, close_data,
            label_markup, label_min_val, label_max_val, 0,
            label_atr_period, method_int
        )
        labels_sell = calculate_labels_one_direction(
            high_data, low_data, close_data,
            label_markup, label_min_val, label_max_val, 1,
            label_atr_period, method_int
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

    if len(dataset) < n_clusters:
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
            max_iter=max_iter
        )
        labels = gm.fit_predict(meta_X.to_numpy(np.float32))
        dataset["labels_meta"] = labels.astype(int)
    except Exception:
        dataset["labels_meta"] = -1

    return dataset

def _wasserstein2_gpu(x: np.ndarray, y: np.ndarray) -> float:
    """
    Wasserstein-2 exacto entre dos muestras empíricas usando POT + CuPy.
    Regresa W2 (no W2²).
    """
    # Pasar a GPU (float64 conserva precisión)
    X = cp.asarray(x, dtype=cp.float64)
    Y = cp.asarray(y, dtype=cp.float64)

    nx, ny = X.shape[0], Y.shape[0]
    a = cp.full(nx, 1.0 / nx, dtype=cp.float64)
    b = cp.full(ny, 1.0 / ny, dtype=cp.float64)

    # Matriz de costes (distancias euclidianas al cuadrado)
    M = ot.utils.dist(X, Y, metric="sqeuclidean")

    # Transporte óptimo exacto (linear programming) en GPU
    # cost = W2²
    cost = ot.emd2(a, b, M, numItermax=10_000)

    return float(cp.sqrt(cost).get())

def _wasserstein2_cpu(x: np.ndarray, y: np.ndarray) -> float:
    """
    Wasserstein-2 exacto en CPU usando POT y NumPy.
    """
    nx, ny = x.shape[0], y.shape[0]
    a = np.full(nx, 1.0 / nx)
    b = np.full(ny, 1.0 / ny)
    M = ot.utils.dist(x, y, metric="sqeuclidean")
    cost = ot.emd2(a, b, M, numItermax=10_000)   # W2²
    return float(np.sqrt(cost))

@njit(cache=True, fastmath=True)
def _kmedoids_pam(
    dist_mat: np.ndarray, k: int, max_iter: int = 100
) -> tuple:
    """
    Implementación minimalista de k-MEDOIDS (PAM) sobre una matriz de distancias
    pre-calculada. Devuelve (labels 0..k-1, índices de los medoids).
    """
    n = dist_mat.shape[0]

    medoids = np.random.choice(n, k, replace=False)
    labels = np.empty(n, dtype=np.int64)

    for _ in range(max_iter):
        # 1) asignación
        for i in prange(n):
            min_dist = 1e20
            min_j = 0
            for j in range(k):
                d = dist_mat[i, medoids[j]]
                if d < min_dist:
                    min_dist = d
                    min_j = j
            labels[i] = min_j

        # 2) actualización de medoids
        new_medoids = medoids.copy()
        for j in range(k):
            # idx = np.where(labels == j)[0]
            count = 0
            for ii in range(n):
                if labels[ii] == j:
                    count += 1
            idx = np.empty(count, dtype=np.int64)
            c = 0
            for ii in range(n):
                if labels[ii] == j:
                    idx[c] = ii
                    c += 1
            if count == 0:  # clúster vacío → re-seed
                new_medoids[j] = np.random.randint(n)
            else:
                # costs = dist_mat[np.ix_(idx, idx)].sum(axis=1)
                min_cost = 1e20
                min_idx = idx[0]
                for m in range(count):
                    cost = 0.0
                    for l in range(count):
                        cost += dist_mat[idx[m], idx[l]]
                    if cost < min_cost:
                        min_cost = cost
                        min_idx = idx[m]
                new_medoids[j] = min_idx

        # check convergence
        converged = True
        for j in range(k):
            if new_medoids[j] != medoids[j]:
                converged = False
                break
        if converged:
            break
        medoids = new_medoids

    # asignación final
    for i in prange(n):
        min_dist = 1e20
        min_j = 0
        for j in range(k):
            d = dist_mat[i, medoids[j]]
            if d < min_dist:
                min_dist = d
                min_j = j
        labels[i] = min_j

    return labels, medoids

@njit(parallel=True, fastmath=True)
def _euclidean_matrix_numba(window_sizes):
    n = window_sizes.shape[0]
    dist_mat = np.zeros((n, n), dtype=np.float64)
    for i in prange(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(window_sizes[i] - window_sizes[j])
            dist_mat[i, j] = dist_mat[j, i] = d
    return dist_mat

@njit(fastmath=True)
def _wasserstein1d_numba(x, y):
    x = np.sort(x)
    y = np.sort(y)
    return np.mean(np.abs(x - y))

@njit(parallel=True, fastmath=True)
def _wasserstein1d_matrix(window_sizes):
    n = window_sizes.shape[0]
    dist_mat = np.zeros((n, n), dtype=np.float64)
    for i in prange(n):
        for j in range(i + 1, n):
            # Wasserstein 1D: ordenar y sumar diferencias absolutas
            x = np.sort(window_sizes[i].ravel())
            y = np.sort(window_sizes[j].ravel())
            d = np.mean(np.abs(x - y))
            dist_mat[i, j] = dist_mat[j, i] = d
    return dist_mat

@njit(parallel=True, fastmath=True)
def _sliced_wasserstein_numba(window_sizes, n_proj=50):
    n = window_sizes.shape[0]
    d = window_sizes.shape[2] if window_sizes.ndim == 3 else 1
    dist_mat = np.zeros((n, n), dtype=np.float64)
    # Numba does not support np.random.normal with 'size' argument in nopython mode.
    # Instead, generate the random vectors manually using Box-Muller.
    vecs = np.empty((n_proj, d), dtype=np.float64)
    for k in range(n_proj):
        for dd in range(d):
            # Box-Muller transform for standard normal
            u1 = np.random.random()
            u2 = np.random.random()
            z = np.sqrt(-2.0 * np.log(u1 + 1e-12)) * np.cos(2.0 * np.pi * u2)
            vecs[k, dd] = z
    for k in range(n_proj):
        v = vecs[k]
        norm = np.sqrt((v ** 2).sum())
        if norm > 0:
            v = v / norm
        for i in prange(n):
            for j in range(i + 1, n):
                if d == 1:
                    x_proj = window_sizes[i].ravel()
                    y_proj = window_sizes[j].ravel()
                else:
                    x_proj = window_sizes[i] @ v
                    y_proj = window_sizes[j] @ v
                d_w = _wasserstein1d_numba(x_proj, y_proj)
                dist_mat[i, j] += d_w
                dist_mat[j, i] += d_w
    dist_mat /= n_proj
    return dist_mat

@njit(fastmath=True)
def _mmd_rbf_numba(x, y, gamma):
    n = x.shape[0]
    m = y.shape[0]
    k_xx = 0.0
    for i in range(n):
        for j in range(n):
            if i != j:
                d2 = 0.0
                for d in range(x.shape[1]):
                    d2 += (x[i, d] - x[j, d]) ** 2
                k_xx += np.exp(-gamma * d2)
    k_yy = 0.0
    for i in range(m):
        for j in range(m):
            if i != j:
                d2 = 0.0
                for d in range(y.shape[1]):
                    d2 += (y[i, d] - y[j, d]) ** 2
                k_yy += np.exp(-gamma * d2)
    k_xy = 0.0
    for i in range(n):
        for j in range(m):
            d2 = 0.0
            for d in range(x.shape[1]):
                d2 += (x[i, d] - y[j, d]) ** 2
            k_xy += np.exp(-gamma * d2)
    mmd2 = (k_xx / (n * (n - 1) + 1e-12)) + (k_yy / (m * (m - 1) + 1e-12)) - 2.0 * (k_xy / (n * m))
    return np.sqrt(max(mmd2, 0.0))

@njit(parallel=True, fastmath=True)
def _mmd_matrix_numba(window_sizes, bandwidth):
    n = window_sizes.shape[0]
    dist_mat = np.zeros((n, n), dtype=np.float64)
    gamma = 1.0 / (2.0 * bandwidth ** 2)
    for i in prange(n):
        for j in range(i + 1, n):
            d = _mmd_rbf_numba(window_sizes[i], window_sizes[j], gamma)
            dist_mat[i, j] = dist_mat[j, i] = d
    return dist_mat

def _distance_matrix(key: tuple,
                     metric: str,
                     bandwidth: float | None,
                     n_proj: int,
                     window_sizes: np.ndarray) -> np.ndarray:
    n = window_sizes.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)
    dist_mat = None  # Inicializa dist_mat para evitar UnboundLocalError

    if metric == 'euclidean':
        dist_mat = _euclidean_matrix_numba(window_sizes)
    elif metric == 'wasserstein':
        # Si es 1D (shape: n_windows, window, 1), usar la versión 1D optimizada
        if window_sizes.ndim == 3 and window_sizes.shape[2] == 1:
            arr1d = window_sizes.reshape(window_sizes.shape[0], window_sizes.shape[1])
            dist_mat = _wasserstein1d_matrix(arr1d)
        # Si es multidimensional (shape: n_windows, window, d)
        elif window_sizes.ndim == 3 and window_sizes.shape[2] > 1:
            n = window_sizes.shape[0]
            dist_mat = np.zeros((n, n), dtype=float)
            for i in range(n):
                for j in range(i + 1, n):
                    x = window_sizes[i]
                    y = window_sizes[j]
                    if cp is not None and cp.cuda.is_available():
                        d = _wasserstein2_gpu(x, y)
                    else:
                        d = _wasserstein2_cpu(x, y)
                    dist_mat[i, j] = dist_mat[j, i] = d
        else:
            raise ValueError("Forma de ventana no soportada para 'wasserstein'")
    elif metric == 'sliced_w':
        dist_mat = _sliced_wasserstein_numba(window_sizes, n_proj)
    elif metric == 'mmd':
        if bandwidth is None:
            centers = np.array([w.mean(axis=0) for w in window_sizes])
            dists = np.zeros((n, n))
            for i in range(n):
                for j in range(i + 1, n):
                    d2 = 0.0
                    for d in range(centers.shape[1]):
                        d2 += (centers[i, d] - centers[j, d]) ** 2
                    dists[i, j] = dists[j, i] = np.sqrt(d2)
            med = np.median(dists[dists > 0])
            bandwidth = med if med > 0 else 1.0
        dist_mat = _mmd_matrix_numba(window_sizes, bandwidth)
    else:
        raise ValueError(f"Métrica desconocida: {metric}")

    if dist_mat is None:
        raise RuntimeError("No se pudo calcular la matriz de distancias: 'dist_mat' no fue asignada.")

    return dist_mat
    
def wkmeans_clustering(
    ds: pd.DataFrame,
    n_clusters: int = 4,
    window_size: int = 60,
    metric: str = "wasserstein",
    step: int = 1,
    bandwidth: float | None = None,
    n_proj: int = 50,
    max_iter: int = 100,
) -> pd.DataFrame:
    """
    Devuelve `ds` con UNA sola columna nueva: labels_meta
    (regímenes de mercado detectados por WK-means / MMDK-means)
    """
    # 0) early-exit si el DF está vacío --------------------------------------
    if ds.empty:
        return ds.copy()

    # 1) matriz de trabajo ---------------------------------------------------
    if "close" not in ds.columns:
        raise ValueError("El DataFrame debe contener una columna 'close'.")

    work = pd.DataFrame({"ret": np.log(ds["close"]).diff()}).dropna()
    X   = work.to_numpy()
    idx = work.index

    # 2) ventanas ------------------------------------------------------------
    starts = np.arange(0, X.shape[0] - window_size + 1, step)
    ends   = starts + window_size - 1
    window_sizes   = [X[s:e + 1] for s, e in zip(starts, ends)]
    win_times = [idx[e] for e in ends]

    if len(window_sizes) < n_clusters:
        raise ValueError("No hay suficientes ventanas para el número de clusters.")

    # 3) matriz de distancias -------------------------------------------------
    # Convertir a array 3-D para evitar múltiples firmas
    window_sizes_array = np.stack(window_sizes)
    dm_key = (window_size, step, metric, bandwidth, n_proj, len(window_sizes))
    dist_mat = _distance_matrix(
        dm_key,
        metric=metric,
        bandwidth=bandwidth,
        n_proj=n_proj,
        window_sizes=window_sizes_array,
    )

    # 4) k-medoids -----------------------------------------------------------
    labels, _ = _kmedoids_pam(dist_mat, n_clusters, max_iter)

    # 5) ensamblar resultado -------------------------------------------------
    res = ds.copy()
    res["labels_meta"] = np.nan
    for i, t in enumerate(win_times):
        res.loc[t, "labels_meta"] = labels[i]

    res["labels_meta"] = res["labels_meta"].ffill()
    return res

@njit(cache=True, fastmath=True)
def calculate_symmetric_correlation_dynamic(data, min_window_size, max_window_size):
    """
    Calcula correlación simétrica dinámica para detectar patrones fractales.
    
    Args:
        data: Array de precios de cierre
        min_window_size: Tamaño mínimo de ventana para patrones
        max_window_size: Tamaño máximo de ventana para patrones
    
    Returns:
        correlations: Array de correlaciones máximas para cada punto
        best_window_sizes: Array de tamaños de ventana correspondientes
    """
    n = len(data)
    min_w = max(2, min_window_size)
    max_w = max(min_w, max_window_size)
    num_correlations = max(0, n - min_w + 1)

    if num_correlations == 0:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.int64)

    correlations = np.zeros(num_correlations, dtype=np.float64)
    best_window_sizes = np.full(num_correlations, -1, dtype=np.int64)

    for i in range(num_correlations):
        max_abs_corr_for_i = -1.0
        best_corr_for_i = 0.0
        current_best_w = -1
        current_max_w = min(max_w, n - i)
        start_w = min_w
        if start_w % 2 != 0:
            start_w += 1

        for w in range(start_w, current_max_w + 1, 2):
            if w < 2 or i + w > n:
                continue
            half_window = w // 2
            window = data[i : i + w]
            first_half = window[:half_window]
            second_half = (window[half_window:] * -1)[::-1]
            
            std1 = np.std(first_half)
            std2 = np.std(second_half)

            if std1 > 1e-9 and std2 > 1e-9:
                mean1 = np.mean(first_half)
                mean2 = np.mean(second_half)
                cov = np.mean((first_half - mean1) * (second_half - mean2))
                corr = cov / (std1 * std2)
                if abs(corr) > max_abs_corr_for_i:
                    max_abs_corr_for_i = abs(corr)
                    best_corr_for_i = corr
                    current_best_w = w
        
        correlations[i] = best_corr_for_i
        best_window_sizes[i] = current_best_w
    
    return correlations, best_window_sizes


@njit(cache=True, fastmath=True)
def generate_future_outcome_labels_for_patterns(
    close_data_len,
    correlations_at_window_start,
    window_sizes_at_window_start,
    source_close_data,
    correlation_threshold,
    min_future_horizon,
    max_future_horizon,
    markup_points,
    direction_int=-1
):
    """
    Genera etiquetas basadas en resultados futuros para patrones fractales.
    
    Args:
        close_data_len: Longitud total de los datos
        correlations_at_window_start: Array de correlaciones
        window_sizes_at_window_start: Array de tamaños de ventana
        source_close_data: Array completo de precios de cierre
        correlation_threshold: Umbral de correlación para considerar patrón válido
        min_future_horizon: Horizonte mínimo de predicción
        max_future_horizon: Horizonte máximo de predicción
        markup_points: Puntos de markup para determinar cambio significativo
        direction_int: Dirección (-1=both, 0=buy, 1=sell)
    
    Returns:
        labels: Array de etiquetas (both: 0.0=compra, 1.0=venta, 2.0=neutral | single: 1.0=señal, 0.0=no señal)
    """
    if direction_int == -1:  # both directions
        labels = np.full(close_data_len, 2.0, dtype=np.float64)  # 2.0: no signal/neutral
    else:  # single direction
        labels = np.full(close_data_len, 0.0, dtype=np.float64)  # 0.0: no signal
    num_potential_windows = len(correlations_at_window_start)

    for idx_window_start in range(num_potential_windows):
        corr_value = correlations_at_window_start[idx_window_start]
        w = window_sizes_at_window_start[idx_window_start]

        # Condición 1: La correlación debe ser suficientemente fuerte
        if abs(corr_value) < correlation_threshold:
            continue

        # Condición 2: Debe encontrarse una ventana válida
        if w < 2:
            continue

        # Momento en el tiempo (índice) cuando el patrón de correlación está completamente formado
        signal_time_idx = idx_window_start + w - 1

        if signal_time_idx >= close_data_len:  # Teóricamente no debería ocurrir
            continue
            
        # Array para almacenar etiquetas de todo el patrón (tanto parte izquierda como derecha)
        pattern_labels = []
        
        # Calculamos etiquetas individuales para todos los puntos del patrón
        for point_idx in range(idx_window_start, signal_time_idx + 1):
            # Precio actual para este punto específico
            current_price = source_close_data[point_idx]
            
            # Determinamos el horizonte para la predicción
            current_horizon = min_future_horizon
            if max_future_horizon > min_future_horizon:
                current_horizon = np.random.randint(min_future_horizon, max_future_horizon + 1)
            
            # Índice del precio futuro relativo al punto actual
            future_price_idx = point_idx + current_horizon
            
            if future_price_idx >= close_data_len:
                continue
                
            future_price = source_close_data[future_price_idx]
            
            # Determinamos la etiqueta para el punto actual
            current_label = 2.0  # Neutral por defecto
            if direction_int == -1:  # both directions (comportamiento original)
                if future_price > current_price + markup_points:
                    current_label = 0.0  # Precio subió
                elif future_price < current_price - markup_points:
                    current_label = 1.0  # Precio bajó
                # Agregamos la etiqueta al array si no es neutral
                if current_label != 2.0:
                    pattern_labels.append(current_label)
            elif direction_int == 0:  # buy only
                if future_price > current_price + markup_points:
                    current_label = 1.0  # Señal válida
                else:
                    current_label = 0.0  # No señal
                pattern_labels.append(current_label)
            elif direction_int == 1:  # sell only
                if future_price < current_price - markup_points:
                    current_label = 1.0  # Señal válida
                else:
                    current_label = 0.0  # No señal
                pattern_labels.append(current_label)
        
        # Si no hay etiquetas significativas en el patrón, pasamos al siguiente
        if len(pattern_labels) == 0:
            continue
            
        # Calculamos la etiqueta promedio de todos los puntos del patrón
        avg_label = 0.0
        for l in pattern_labels:
            avg_label += l
        avg_label /= len(pattern_labels)
        
        # Determinamos la etiqueta general para todo el patrón
        if direction_int == -1:  # both directions
            pattern_label = 0.0 if avg_label < 0.5 else 1.0
        else:  # single direction
            pattern_label = 1.0 if avg_label >= 0.5 else 0.0
        
        # Asignamos esta etiqueta a todos los puntos del patrón
        for i in range(idx_window_start, signal_time_idx + 1):
            labels[i] = pattern_label
    
    return labels


def get_labels_fractal_patterns(
    dataset,
    min_window_size=6,
    max_window_size=60,
    correlation_threshold=0.7,
    min_future_horizon=5,
    max_future_horizon=5,
    markup_points=0.00010,
    direction='both'
) -> pd.DataFrame:
    """
    Genera etiquetas basadas en patrones fractales simétricos.
    
    Args:
        dataset: DataFrame con columna 'close'
        min_window_size: Tamaño mínimo de ventana para patrones
        max_window_size: Tamaño máximo de ventana para patrones
        correlation_threshold: Umbral de correlación para patrones válidos
        min_future_horizon: Horizonte mínimo de predicción en barras
        max_future_horizon: Horizonte máximo de predicción en barras
        markup_points: Puntos de markup para cambios significativos
        direction: Dirección de las señales ('buy', 'sell', 'both')
    
    Returns:
        DataFrame con columna 'labels_main' agregada
    """
    if 'close' not in dataset.columns:
        raise ValueError("Dataset must contain a 'close' column.")

    close_data = dataset['close'].values
    n_data = len(close_data)

    if min_window_size < 2:
        min_window_size = 2
    if max_window_size < min_window_size:
        max_window_size = min_window_size
    if min_future_horizon <= 0:
        raise ValueError("min_future_horizon must be > 0")
    if max_future_horizon < min_future_horizon:
        raise ValueError("max_future_horizon must be >= min_future_horizon")
    
    correlations_at_start, best_window_sizes_at_start = calculate_symmetric_correlation_dynamic(
        close_data,
        min_window_size,
        max_window_size,
    )

    direction_map = {'buy': 0, 'sell': 1, 'both': -1}
    direction_int = direction_map.get(direction, -1)
    
    if direction in {'buy', 'sell'}:
        # Modo una dirección - etiquetas binarias
        labels = generate_future_outcome_labels_for_patterns(
            n_data,
            correlations_at_start,
            best_window_sizes_at_start,
            close_data,
            correlation_threshold,
            min_future_horizon,
            max_future_horizon,
            markup_points,
            direction_int
        )
        result_df = dataset.copy()
        result_df['labels_main'] = pd.Series(labels, index=dataset.index)
        return result_df
        
    elif direction == 'both':
        # Modo bidireccional - etiquetas ternarias (comportamiento original)
        labels = generate_future_outcome_labels_for_patterns(
            n_data,
            correlations_at_start,
            best_window_sizes_at_start,
            close_data,
            correlation_threshold,
            min_future_horizon,
            max_future_horizon,
            markup_points,
            -1
        )
        result_df = dataset.copy()
        result_df['labels_main'] = pd.Series(labels, index=dataset.index)
        return result_df
    else:
        raise ValueError("direction must be 'buy', 'sell', or 'both'")