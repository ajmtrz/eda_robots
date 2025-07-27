import os
import logging
import numpy as np
import pandas as pd
import cupy as cp
#import ot
import bottleneck as bn
from numba import njit, prange
from numba.typed import List
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

# Logging configuration
logging.getLogger('hmmlearn').setLevel(logging.ERROR)
logging.getLogger("tensorflow").setLevel(logging.ERROR)

##### PRICES #####

def get_prices(symbol, timeframe, history_path) -> pd.DataFrame:
    history_file = os.path.join(history_path, f"{symbol}_{timeframe}.csv")
    p = pd.read_csv(history_file, sep=r"\s+")
    # Create DataFrame with all necessary columns
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
    return pFixed

##### FEATURE ENGINEERING #####

@njit(cache=True)
def std_manual(x):
    m = mean_manual(x)
    return np.sqrt(np.sum((x - m) ** 2) / (x.size - 1)) if x.size > 1 else 0.0

@njit(cache=True)
def skew_manual(x):
    s = std_manual(x)
    if s == 0:
        return 0.0
    m = mean_manual(x)
    return mean_manual(((x - m) / s) ** 3)

@njit(cache=True)
def kurt_manual(x):
    s = std_manual(x)
    if s == 0:
        return 0.0
    m = mean_manual(x)
    return mean_manual(((x - m) / s) ** 4) - 3.0

@njit(cache=True)
def zscore_manual(x):
    s = std_manual(x)
    if s == 0:
        return 0.0
    m = mean_manual(x)
    return (x[-1] - m) / s

@njit(cache=True)
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

@njit(cache=True)
def mean_manual(x):
    if x.size == 0:
        return 0.0
    sum_val = 0.0
    for i in range(x.size):
        sum_val += x[i]
    return sum_val / x.size

@njit(cache=True)
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

@njit(cache=True)
def momentum_roc(x):
    if len(x) < 2: return 0.0
    ratio = x[0]/x[-1]
    return ratio - 1.0

@njit(cache=True)
def fractal_dimension_manual(x):
    eps = std_manual(x) / 4
    if eps == 0:
        return 1.0
    count = np.sum(np.abs(np.diff(x)) > eps)
    if count == 0:
        return 1.0
    return 1.0 + np.log(count) / np.log(len(x))

@njit(cache=True)
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

@njit(cache=True)
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

@njit(cache=True)
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

@njit(cache=True)
def sharpe_manual(x):
    mean = mean_manual(x)
    std = std_manual(x)
    return mean / std if std != 0 else 0.0

@njit(cache=True)
def fisher_transform(x):
    return 0.5 * np.log((1 + x) / (1 - x))

@njit(cache=True)
def chande_momentum(x):
    returns = np.diff(x)
    up = np.sum(returns[returns > 0])
    down = np.abs(np.sum(returns[returns < 0]))
    return (up - down) / (up + down) if (up + down) != 0 else 0.0

@njit(cache=True)
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

@njit(cache=True)
def efficiency_ratio(x):
    direction = x[-1] - x[0]
    volatility = np.sum(np.abs(np.diff(x)))
    return direction/volatility if volatility != 0 else 0.0

@njit(cache=True)
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

@njit(cache=True)
def correlation_skew_manual(x):
    lag = min(5, x.size // 2)
    if x.size < lag + 1:
        return 0.0
    corr_pos = corr_manual(x[:-lag], x[lag:])
    corr_neg = corr_manual(-x[:-lag], x[lag:])
    return corr_pos - corr_neg

@njit(cache=True)
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

@njit(cache=True)
def iqr_manual(a):
    n = a.size
    if n == 0:
        return 0.0
    b = a.copy()
    b.sort()
    q1_idx = int(0.25 * (n - 1))
    q3_idx = int(0.75 * (n - 1))
    return b[q3_idx] - b[q1_idx]

@njit(cache=True)
def coeff_var_manual(a):
    m = mean_manual(a)
    if m == 0:
        return 0.0
    s = std_manual(a)
    return s / m

@njit(cache=True)
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

@njit(cache=True)
def volatility_skew(x):
    n = len(x)
    if n < 2:
        return 0.0
    up_vol = std_manual(np.maximum(x[1:] - x[:-1], 0))
    down_vol = std_manual(np.maximum(x[:-1] - x[1:], 0))
    return (up_vol - down_vol)/(up_vol + down_vol) if (up_vol + down_vol) != 0 else 0.0

@njit(cache=True)
def compute_returns(prices):
    n = len(prices)
    if n <= 1:
        return np.empty(0, dtype=np.float64)
    
    returns = np.empty(n - 1, dtype=np.float64)
    for i in range(n - 1):
        if prices[i] <= 0 or prices[i + 1] <= 0:
            returns[i] = 0.0
        else:
            ratio = prices[i + 1] / prices[i]
            returns[i] = np.log(ratio)
    return returns

@njit(cache=True)
def should_use_returns(stat_name):
    """Determina si un estadístico debe usar retornos en lugar de precios."""
    return stat_name in ("mean", "median", "std", "iqr", "mad", "sharpe", "autocorr")

@njit(cache=True)
def compute_features(close, periods_main, periods_meta, stats_main, stats_meta):
    n = len(close)
    # Calcular total de features
    total_features = len(periods_main) * len(stats_main)
    
    # Solo añadir meta features si existen
    has_meta = len(periods_meta) > 0 and len(stats_meta) > 0
    if has_meta:
        total_features += len(periods_meta) * len(stats_meta)
    
    features = np.full((n, total_features), np.nan)
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
    if has_meta:
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

def get_features(data: pd.DataFrame, hp, decimal_precision=6):
    close = data['close'].values
    index = data.index
    
    # Verificar datos vacíos
    if len(close) == 0:
        return pd.DataFrame(index=index)
    
    periods_main = hp["feature_main_periods"]
    stats_main = hp["feature_main_stats"]
    
    # Manejar meta features
    periods_meta = hp.get("feature_meta_periods", [])
    stats_meta = hp.get("feature_meta_stats", [])
    
    if len(stats_main) == 0:
        raise ValueError("La lista de estadísticas MAIN está vacía.")
    
    # Crear listas tipadas para MAIN
    periods_main_t = List(periods_main)
    stats_main_t = List(stats_main)
    
    # SOLUCIÓN CRÍTICA: Crear listas tipadas para META con tipo específico
    if periods_meta and stats_meta:
        periods_meta_t = List(periods_meta)
        stats_meta_t = List(stats_meta)
    else:
        # Crear listas vacías con tipos explícitos
        periods_meta_t = List()
        stats_meta_t = List()
        
        # Forzar tipado añadiendo y removiendo elementos ficticios
        if len(periods_main) > 0:
            periods_meta_t.append(periods_main[0])
            periods_meta_t.pop()
        if len(stats_main) > 0:
            stats_meta_t.append(stats_main[0])
            stats_meta_t.pop()

    # Calcular features
    feats = compute_features(
        close,
        periods_main_t,
        periods_meta_t,
        stats_main_t,
        stats_meta_t
    )
    if np.isnan(feats).all():
        return pd.DataFrame(index=index)
    feats = np.round(feats, decimal_precision)
    # ───── OPTIMIZACIÓN: Generar nombres de columnas de forma más eficiente ─────
    colnames = []
    for p in periods_main:
        for s in stats_main:
            colnames.append(f"{p}_{s}_main_feature")
    
    # Agregar nombres de columnas meta solo si existen
    if periods_meta and stats_meta:
        for p in periods_meta:
            for s in stats_meta:
                colnames.append(f"{p}_{s}_meta_feature")
    
    # ───── OPTIMIZACIÓN: Crear DataFrame de forma más eficiente ─────
    df = pd.DataFrame(feats, columns=colnames, index=index)

    # 1) añadir OHLCV **antes** de limpiar para que sufran el mismo filtrado
    ohlcv = data.loc[index, ["open", "high", "low", "close", "volume"]]
    df = pd.concat([df, ohlcv], axis=1)

    # 2) limpiar Inf/NaN en TODAS las columnas de features (main + meta)
    main_feat_cols = df.filter(like='_main_feature').columns
    meta_feat_cols = df.filter(like='_meta_feature').columns
    all_feat_cols = list(main_feat_cols) + list(meta_feat_cols)
    df[all_feat_cols] = df[all_feat_cols].replace([np.inf, -np.inf], np.nan)

    # 3) verificación — el close debe coincidir al 100 %
    if not np.allclose(df["close"].values, ohlcv.loc[df.index, "close"].values):
        raise RuntimeError("❌ 'close' desalineado tras generar features")

    return df

##### NON-RELIABILITY BASED LABELING #####

@njit(cache=True)
def calculate_atr_simple(high, low, close, period=14):
    """
    Calcula el Average True Range (ATR) de forma eficiente usando Numba.
    
    El ATR es una medida de volatilidad que considera:
    - High - Low
    - |High - Close_prev|
    - |Low - Close_prev|

    Args:
        high: Array de precios máximos
        low: Array de precios mínimos  
        close: Array de precios de cierre
        period: Período para el cálculo de ATR (default: 14)

    Returns:
        Array de valores ATR calculados usando el método de Wilder
    """
    n   = len(close)
    tr  = np.empty(n)
    atr = np.empty(n)
    
    # ✅ OPTIMIZACIÓN: Inicialización directa del primer TR
    tr[0] = high[0] - low[0]
    
    # Calcular True Range para cada período
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i-1])
        lc = abs(low[i]  - close[i-1])
        tr[i] = max(hl, hc, lc)
    
    # ✅ MEJORA: Promedio acumulado para períodos iniciales (más estable)
    cumsum = tr[0]
    atr[0] = tr[0]
    for i in range(1, min(period, n)):
        cumsum += tr[i]
        atr[i] = cumsum / (i+1)
    
    if n <= period-1:
        return atr
        
    # Primera media "oficial" (índice period-1)
    atr[period-1] = cumsum / period
    
    # ✅ OPTIMIZACIÓN: Aplicar método de Wilder de forma eficiente
    # ATR[i] = ((ATR[i-1] * (period-1)) + TR[i]) / period
    for i in range(period, n):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    
    return atr

@njit(cache=True)
def calculate_labels_random(close, atr, label_markup, label_min_val, label_max_val, direction_int, method_int=5):
    """
    Etiquetado random con soporte para bidireccional (2) y unidireccional (0=buy, 1=sell) siguiendo la convención:
    - Direcciones únicas: 1.0=éxito direccional, 0.0=fracaso, 2.0=no confiable
    - Ambas: 0.0=buy, 1.0=sell, 2.0=no confiable
    """
    n = len(close)
    if n <= label_max_val:
        return np.full(0, 2.0, dtype=np.float64)
    result = np.full(n - label_max_val, 2.0, dtype=np.float64)
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
        if direction_int == 2:
            # Bidireccional: 0.0=buy, 1.0=sell, 2.0=no confiable
            is_buy = future_price > close[i] + dyn_mk
            is_sell = future_price < close[i] - dyn_mk
            if is_buy and not is_sell:
                result[i] = 0.0
            elif is_sell and not is_buy:
                result[i] = 1.0
            else:
                result[i] = 2.0
        elif direction_int == 0:
            # Solo buy: 1.0=éxito, 0.0=fracaso, 2.0=no confiable
            if future_price > close[i] + dyn_mk:
                result[i] = 1.0
            elif future_price < close[i] - dyn_mk:
                result[i] = 0.0
            else:
                result[i] = 2.0
        elif direction_int == 1:
            # Solo sell: 1.0=éxito, 0.0=fracaso, 2.0=no confiable
            if future_price < close[i] - dyn_mk:
                result[i] = 1.0
            elif future_price > close[i] + dyn_mk:
                result[i] = 0.0
            else:
                result[i] = 2.0
        else:
            result[i] = 2.0  # fallback
    return result

def get_labels_random(dataset, label_markup=0.5, label_min_val=1, label_max_val=5,
                        label_atr_period=14, label_method_random='random', direction=2) -> pd.DataFrame:
    """
    Etiquetado random con soporte para bidireccional (2) y unidireccional (0=buy, 1=sell) siguiendo la convención:
    - Direcciones únicas: 1.0=éxito direccional, 0.0=fracaso, 2.0=no confiable
    - Ambas: 0.0=buy, 1.0=sell, 2.0=no confiable

    direction debe venir ya mapeada a int: 0=buy, 1=sell, 2=both
    """
    close = dataset['close'].values
    high = dataset['high'].values
    low = dataset['low'].values
    atr = calculate_atr_simple(high, low, close, period=label_atr_period)
    method_map = {'first': 0, 'last': 1, 'mean': 2, 'max': 3, 'min': 4, 'random': 5}
    method_int = method_map.get(label_method_random, 5)
    labels = calculate_labels_random(
        close, atr,
        label_markup, label_min_val, label_max_val, direction,
        method_int
    )
    # ✅ CORRECCIÓN CRÍTICA: Alineación correcta de etiquetas
    # El array de etiquetas tiene longitud (n - label_max_val), debe alinearse correctamente
    labels_series = pd.Series(labels, index=dataset.index[:len(labels)])
    labels_aligned = labels_series.reindex(dataset.index, fill_value=2.0)
    dataset['labels_main'] = labels_aligned
    dataset['labels_main'] = dataset['labels_main'].fillna(2.0)
    return dataset

@njit(cache=True)
def calculate_labels_filter(close, lvl, q, direction=2):
    """
    direction: 0=buy only, 1=sell only, 2=both
    For buy: 1.0=éxito direccional, 0.0=fracaso, 2.0=no confiable
    For sell: 1.0=éxito direccional, 0.0=fracaso, 2.0=no confiable
    For both: 0.0=buy, 1.0=sell, 2.0=no confiable
    """
    labels = np.empty(len(close), dtype=np.float64)
    for i in range(len(close)):
        curr_lvl = lvl[i]
        if direction == 2:  # both
            if curr_lvl > q[1]:
                labels[i] = 1.0  # Sell
            elif curr_lvl < q[0]:
                labels[i] = 0.0  # Buy
            else:
                labels[i] = 2.0  # No confiable
        elif direction == 0:  # solo buy
            if curr_lvl < q[0]:
                labels[i] = 1.0  # Éxito direccional (compra)
            elif curr_lvl > q[1]:
                labels[i] = 0.0  # Fracaso direccional (venta)
            else:
                labels[i] = 2.0  # Patrón no confiable
        elif direction == 1:  # solo sell
            if curr_lvl > q[1]:
                labels[i] = 1.0  # Éxito direccional (venta)
            elif curr_lvl < q[0]:
                labels[i] = 0.0  # Fracaso direccional (compra)
            else:
                labels[i] = 2.0  # Patrón no confiable
        else:
            labels[i] = 2.0  # fallback
    return labels

def get_labels_filter(
    dataset, 
    label_rolling=200, 
    label_quantiles=[.45, .55], 
    label_polyorder=3, 
    label_decay_factor=0.95, 
    label_method_trend='normal',
    direction=2,
) -> pd.DataFrame:
    """
    Generates labels for a financial dataset based on price deviation from a Savitzky-Golay label_filter,
    with exponential weighting applied to prioritize recent data. Supports both normal and inverse methods.

    Args:
        dataset (pd.DataFrame): DataFrame containing financial data with a 'close' column.
        label_rolling (int, optional): Window size for the Savitzky-Golay label_filter. Defaults to 200.
        label_quantiles (list, optional): Quantiles to define the "reversion zone". Defaults to [.45, .55].
        label_polyorder (int, optional): Polynomial order for the Savitzky-Golay label_filter. Defaults to 3.
        label_decay_factor (float, optional): Exponential decay factor for weighting past data. 
                                        Lower values prioritize recent data more. Defaults to 0.95.
        direction (int, optional): 0=buy only, 1=sell only, 2=both. Defaults to 2.
        method (str, optional): 'normal' or 'inverse'. Defaults to 'normal'.

    Returns:
        pd.DataFrame: The original DataFrame with a new 'labels_main' column and filtered rows:
                       - 'labels_main' column: 
                            - 0: Buy (or fracaso direccional en modo unidireccional)
                            - 1: Sell (o éxito direccional en modo unidireccional)
                            - 2: Patrón no confiable
                       - Rows with missing values (NaN) are removed.
                       - The temporary 'lvl' column is removed. 
    """

    # Calculate smoothed prices using the Savitzky-Golay label_filter
    smoothed_prices, filtering_successful = safe_savgol_filter(dataset['close'].values, label_rolling=label_rolling, label_polyorder=label_polyorder)
    if not filtering_successful:
        return pd.DataFrame()
    # Calculate the difference between the actual closing prices and the smoothed prices
    diff = dataset['close'] - smoothed_prices
    # Apply exponential weighting to the 'diff' values
    weighted_diff = diff * np.exp(np.arange(len(diff)) * label_decay_factor / len(diff)) 
    # Apply method-specific transformation
    if label_method_trend == 'inverse':
        dataset['lvl'] = 1 / weighted_diff  # Inverse method
    else:
        dataset['lvl'] = weighted_diff  # Normal method
    # Calculate the quantiles of the 'lvl' column (price deviation)
    q = tuple(dataset['lvl'].quantile(label_quantiles).to_list())
    # Extract the closing prices and the calculated 'lvl' values as NumPy arrays
    close = dataset['close'].values
    lvl = dataset['lvl'].values
    # Calculate buy/sell labels using the 'calculate_labels_filter' function 
    labels = calculate_labels_filter(close, lvl, q, direction=direction) 
    # Trimming the dataset and adding labels
    dataset['labels_main'] = labels
    dataset['labels_main'] = dataset['labels_main'].fillna(2.0)
    # Return the modified DataFrame with the 'lvl' column removed
    return dataset.drop(columns=['lvl'])

@njit(cache=True)
def calc_labels_binary(close, lvl1, lvl2, q1, q2, direction=2):
    """
    Etiquetado bidireccional/unidireccional con dos filtros.
    direction: 0=solo buy, 1=solo sell, 2=ambos
    Para direcciones únicas: 1.0=éxito direccional, 0.0=fracaso, 2.0=no confiable
    Para ambas: 0.0=buy, 1.0=sell, 2.0=no confiable
    """
    labels = np.empty(len(close), dtype=np.float64)
    for i in range(len(close)):
        curr_lvl1 = lvl1[i]
        curr_lvl2 = lvl2[i]
        if direction == 2:  # both
            if curr_lvl1 > q1[1]:
                labels[i] = 1.0  # Sell
            elif curr_lvl2 < q2[0]:
                labels[i] = 0.0  # Buy
            else:
                labels[i] = 2.0  # No confiable
        elif direction == 0:  # solo buy
            if curr_lvl2 < q2[0]:
                labels[i] = 1.0  # Éxito direccional (compra)
            elif curr_lvl1 > q1[1]:
                labels[i] = 0.0  # Fracaso direccional (venta)
            else:
                labels[i] = 2.0  # Patrón no confiable
        elif direction == 1:  # solo sell
            if curr_lvl1 > q1[1]:
                labels[i] = 1.0  # Éxito direccional (venta)
            elif curr_lvl2 < q2[0]:
                labels[i] = 0.0  # Fracaso direccional (compra)
            else:
                labels[i] = 2.0  # Patrón no confiable
        else:
            labels[i] = 2.0  # fallback
    return labels

def get_labels_filter_binary(
    dataset, 
    label_rolling=200, 
    label_rolling2=200, 
    label_quantiles=[.45, .55], 
    label_polyorder=3, 
    direction=2
) -> pd.DataFrame:
    """
    Generates trading labels based on price deviation from two Savitzky-Golay filters applied
    in opposite directions (forward and reversed) to the closing price data.

    Supports bidirectional (both) and unidirectional (buy/sell only) labeling.
    For unidirectional: 1.0=éxito direccional, 0.0=fracaso, 2.0=no confiable.
    For both: 0.0=buy, 1.0=sell, 2.0=no confiable.

    Args:
        dataset (pd.DataFrame): DataFrame containing financial data with a 'close' column.
        label_rolling (int, optional): Window size for the first Savitzky-Golay label_filter. Defaults to 200.
        label_rolling2 (int, optional): Window size for the second Savitzky-Golay label_filter. Defaults to 200.
        label_quantiles (list, optional): Quantiles to define the "reversion zones". Defaults to [.45, .55].
        label_polyorder (int, optional): Polynomial order for both Savitzky-Golay filters. Defaults to 3.
        direction (int, optional): 0=buy only, 1=sell only, 2=both. Defaults to 2.

    Returns:
        pd.DataFrame: The original DataFrame with a new 'labels_main' column and filtered rows:
                       - 'labels_main' column: 
                            - 0: Buy (o fracaso direccional en modo unidireccional)
                            - 1: Sell (o éxito direccional en modo unidireccional)
                            - 2: Patrón no confiable
                       - Rows with missing values (NaN) are removed.
                       - Temporary 'lvl1' and 'lvl2' columns are removed.
    """

    close = dataset['close'].values
    # Apply the first Savitzky-Golay label_filter (forward direction)
    smoothed_prices, filtering_successful1 = safe_savgol_filter(close, label_rolling=label_rolling, label_polyorder=label_polyorder)
    # Apply the second Savitzky-Golay label_filter (could be in reverse direction if rolling2 is negative)
    smoothed_prices2, filtering_successful2 = safe_savgol_filter(close, label_rolling=label_rolling2, label_polyorder=label_polyorder)
    if not filtering_successful1 or not filtering_successful2:
        return pd.DataFrame()
    # Calculate price deviations from both smoothed price series
    diff1 = dataset['close'] - smoothed_prices
    diff2 = dataset['close'] - smoothed_prices2
    # Add price deviations as new columns to the DataFrame
    dataset['lvl1'] = diff1
    dataset['lvl2'] = diff2
    # Calculate quantiles for the "reversion zones" for both price deviation series
    q1 = tuple(dataset['lvl1'].quantile(label_quantiles).to_list())
    q2 = tuple(dataset['lvl2'].quantile(label_quantiles).to_list())
    # Extract relevant data for label calculation
    lvl1 = dataset['lvl1'].values
    lvl2 = dataset['lvl2'].values
    # Calculate buy/sell labels using the 'calc_labels_binary' function
    labels = calc_labels_binary(close, lvl1, lvl2, q1, q2, direction=direction)
    # Trimming the dataset and adding labels
    dataset['labels_main'] = labels
    dataset['labels_main'] = dataset['labels_main'].fillna(2.0)
    return dataset.drop(columns=['lvl1', 'lvl2'])

@njit(cache=True)
def calc_labels_filter_multi(close, lvls, qs, direction=2):
    """
    direction: 0=solo buy, 1=solo sell, 2=ambas
    Para direcciones únicas: 1.0=éxito direccional, 0.0=fracaso, 2.0=no confiable
    Para ambas: 0.0=buy, 1.0=sell, 2.0=no confiable
    """
    labels = np.empty(len(close), dtype=np.float64)
    for i in range(len(close)):
        buy_signals = 0
        sell_signals = 0
        for j in range(len(lvls)):
            curr_lvl = lvls[j][i]
            curr_q_low = qs[j][0][i]
            curr_q_high = qs[j][1][i]
            if curr_lvl > curr_q_high:
                sell_signals += 1
            elif curr_lvl < curr_q_low:
                buy_signals += 1

        if direction == 2:
            # Esquema clásico: 0.0=buy, 1.0=sell, 2.0=no confiable
            if buy_signals > 0 and sell_signals == 0:
                labels[i] = 0.0
            elif sell_signals > 0 and buy_signals == 0:
                labels[i] = 1.0
            else:
                labels[i] = 2.0
        elif direction == 0:
            # Solo buy: 1.0=éxito, 0.0=fracaso, 2.0=no confiable
            if buy_signals > 0 and sell_signals == 0:
                labels[i] = 1.0  # Éxito direccional (buy)
            elif sell_signals > 0 and buy_signals == 0:
                labels[i] = 0.0  # Fracaso direccional (no buy, sino sell)
            else:
                labels[i] = 2.0  # Patrón no confiable
        elif direction == 1:
            # Solo sell: 1.0=éxito, 0.0=fracaso, 2.0=no confiable
            if sell_signals > 0 and buy_signals == 0:
                labels[i] = 1.0  # Éxito direccional (sell)
            elif buy_signals > 0 and sell_signals == 0:
                labels[i] = 0.0  # Fracaso direccional (no sell, sino buy)
            else:
                labels[i] = 2.0  # Patrón no confiable
        else:
            labels[i] = 2.0  # fallback
    return labels

def get_labels_filter_multi(
    dataset,
    label_rolling_periods_big=[200, 400, 600],
    label_quantiles=[.45, .55],
    label_window_size=100,
    label_polyorder=3,
    direction=2
) -> pd.DataFrame:
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
        label_rolling_periods_big (list, optional): List of label_rolling window_size sizes for the Savitzky-Golay label_filter. 
                                           Defaults to [200, 400, 600].
        quantiles (list, optional): Quantiles to define the "reversion zone". Defaults to [.45, .55].
        window_size (int, optional): Window size for calculating label_rolling quantiles. Defaults to 100.
        label_polyorder (int, optional): Polynomial order for the Savitzky-Golay label_filter. Defaults to 3.
        direction (int, optional): 0=buy only, 1=sell only, 2=both. Defaults to 2.

    Returns:
        pd.DataFrame: The original DataFrame with a new 'labels_main' column and filtered rows:
                       - 'labels_main' column: 
                            - 0: Buy (o fracaso direccional en modo unidireccional)
                            - 1: Sell (o éxito direccional en modo unidireccional)
                            - 2: Patrón no confiable
                       - Rows with missing values (NaN) are removed. 
    """
    
    # Lists to store price deviation levels and quantiles for each label_rolling period
    all_levels = []
    all_quantiles = []
    # Calculate smoothed price trends and label_rolling quantiles for each label_rolling period
    for label_rolling in label_rolling_periods_big:
                # Calculate smoothed prices using the Savitzky-Golay label_filter
        smoothed_prices, filtering_successful = safe_savgol_filter(dataset['close'].values,
                                       label_rolling=label_rolling,
                                       label_polyorder=label_polyorder)
        if not filtering_successful:
            return pd.DataFrame()
        # Calculate the price deviation from the smoothed prices
        diff = dataset['close'] - smoothed_prices
        # Create a temporary DataFrame to calculate label_rolling quantiles
        temp_df = pd.DataFrame({'diff': diff})
        # Calculate quantiles using pandas rolling for better compatibility
        q_low = temp_df['diff'].rolling(window=label_window_size, min_periods=1).quantile(label_quantiles[0])
        q_high = temp_df['diff'].rolling(window=label_window_size, min_periods=1).quantile(label_quantiles[1])
        # Store the price deviation and quantiles for the current label_rolling period
        all_levels.append(diff)
        all_quantiles.append([q_low.values, q_high.values])
    # Convert lists to NumPy arrays for faster calculations (potentially using Numba)
    lvls_array = np.array(all_levels)
    qs_array = np.array(all_quantiles)
    # Calculate buy/sell labels using the 'calc_labels_multiple_filters' function 
    labels = calc_labels_filter_multi(dataset['close'].values, lvls_array, qs_array, direction=direction)
    # Add the calculated labels to the DataFrame
    dataset['labels_main'] = labels
    dataset['labels_main'] = dataset['labels_main'].fillna(2.0)
    return dataset

######### RELIABILITY BASED LABELING #########

@njit(cache=True)
def calculate_symmetric_correlation_dynamic(data, min_window_size, max_window_size):
    """
    Calcula correlación simétrica dinámica para detectar patrones fractales.
    
    ✅ OPTIMIZACIÓN: Implementación eficiente que busca patrones simétricos
    comparando la primera mitad de una ventana con la segunda mitad invertida y negada.
    
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

    # ✅ OPTIMIZACIÓN: Pre-calcular constantes para evitar recálculos
    epsilon = 1e-9  # Umbral de precisión numérica

    for i in range(num_correlations):
        max_abs_corr_for_i = -1.0
        best_corr_for_i = 0.0
        current_best_w = -1
        current_max_w = min(max_w, n - i)
        start_w = min_w
        
        # ✅ ASEGURAR VENTANAS PARES: Solo procesar ventanas de tamaño par
        if start_w % 2 != 0:
            start_w += 1

        for w in range(start_w, current_max_w + 1, 2):
            if w < 2 or i + w > n:
                continue
                
            half_window = w // 2
            window = data[i : i + w]
            first_half = window[:half_window]
            
            # ✅ OPTIMIZACIÓN: Calcular segunda mitad de forma eficiente
            second_half = (window[half_window:] * -1)[::-1]
            
            # ✅ VALIDACIÓN NUMÉRICA: Verificar variabilidad antes de calcular correlación
            std1 = np.std(first_half)
            std2 = np.std(second_half)

            if std1 > epsilon and std2 > epsilon:
                # ✅ OPTIMIZACIÓN: Cálculo directo de correlación
                mean1 = np.mean(first_half)
                mean2 = np.mean(second_half)
                
                # Covarianza normalizada
                cov = np.mean((first_half - mean1) * (second_half - mean2))
                corr = cov / (std1 * std2)
                
                # ✅ CRITERIO DE SELECCIÓN: Mantener correlación con mayor valor absoluto
                if abs(corr) > max_abs_corr_for_i:
                    max_abs_corr_for_i = abs(corr)
                    best_corr_for_i = corr
                    current_best_w = w
        
        correlations[i] = best_corr_for_i
        best_window_sizes[i] = current_best_w
    
    return correlations, best_window_sizes


@njit(cache=True)
def calculate_future_outcome_labels_for_patterns(
    close_data_len,
    correlations_at_window_start,
    window_sizes_at_window_start,
    source_close_data,
    atr,
    correlation_threshold,
    min_future_horizon,
    max_future_horizon,
    markup_multiplier,
    direction=2
):
    """
    Genera etiquetas basadas en resultados futuros para patrones fractales usando ATR * markup.
    Siguiendo el enfoque del artículo MQL5:
    - Etiqueta TODOS los patrones fractales encontrados
    - 0.0/1.0: Patrones confiables (correlación >= threshold) con señal direccional clara
    - 2.0: Patrones no confiables O confiables sin direccionalidad clara
    
    Args:
        close_data_len: Longitud total de los datos
        correlations_at_window_start: Array de correlaciones
        window_sizes_at_window_start: Array de tamaños de ventana
        source_close_data: Array completo de precios de cierre
        atr: Array de valores ATR
        correlation_threshold: Umbral de correlación para considerar patrón válido
        min_future_horizon: Horizonte mínimo de predicción
        max_future_horizon: Horizonte máximo de predicción
        markup_multiplier: Multiplicador de ATR para determinar cambio significativo
        direction: Dirección (0=buy, 1=sell, 2=both)
    
    Returns:
        labels: Array de etiquetas siguiendo el enfoque MQL5:
                - both: 0.0=compra, 1.0=venta, 2.0=no confiable
                - single: 1.0=éxito direccional, 0.0=fracaso direccional, 2.0=no confiable
    """
    # Inicialización: SIEMPRE empezar con 2.0 (no confiable) para cualquier dirección
    # Siguiendo el enfoque del artículo MQL5: primero evaluar confiabilidad
    labels = np.full(close_data_len, 2.0, dtype=np.float64)  # 2.0: no confiable
    num_potential_windows = len(correlations_at_window_start)

    for idx_window_start in range(num_potential_windows):
        corr_value = correlations_at_window_start[idx_window_start]
        w = window_sizes_at_window_start[idx_window_start]

        # Verificar que se encontró una ventana válida
        if w < 2:
            continue

        # Momento en el tiempo (índice) cuando el patrón de correlación está completamente formado
        signal_time_idx = idx_window_start + w - 1

        if signal_time_idx >= close_data_len:  # Teóricamente no debería ocurrir
            continue
        
        # ✅ ENFOQUE ARTÍCULO MQL5: Verificar confiabilidad del patrón
        is_reliable_pattern = abs(corr_value) >= correlation_threshold
        
        if not is_reliable_pattern:
            # Patrón no confiable - SIEMPRE etiquetar como 2.0 (no confiable)
            # Esto aplica para cualquier dirección (buy, sell, both)
            for i in range(idx_window_start, signal_time_idx + 1):
                labels[i] = 2.0  # No confiable
            continue
            
        # Patrón confiable - evaluar direccionalidad basada en movimientos futuros
        pattern_labels = []
        
        # Calculamos etiquetas individuales para todos los puntos del patrón
        for point_idx in range(idx_window_start, signal_time_idx + 1):
            # ✅ CORRECCIÓN CRÍTICA: Verificar límites del array ATR
            if point_idx >= len(atr):
                continue
                
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
            
            # ✅ NUEVA FUNCIONALIDAD: Usar ATR dinámico
            dynamic_markup = markup_multiplier * atr[point_idx]
            
            # Determinamos la etiqueta para el punto actual
            current_label = 2.0  # Neutral por defecto
            if direction == 0:  # buy only
                if future_price > current_price + dynamic_markup:
                    current_label = 1.0  # Señal válida
                else:
                    current_label = 0.0  # No señal
                pattern_labels.append(current_label)
            elif direction == 1:  # sell only
                if future_price < current_price - dynamic_markup:
                    current_label = 1.0  # Señal válida
                else:
                    current_label = 0.0  # No señal
                pattern_labels.append(current_label)
            elif direction == 2:  # both directions (comportamiento original)
                if future_price > current_price + dynamic_markup:
                    current_label = 0.0  # Precio subió
                elif future_price < current_price - dynamic_markup:
                    current_label = 1.0  # Precio bajó
                # Agregamos la etiqueta al array si no es neutral
                if current_label != 2.0:
                    pattern_labels.append(current_label)
        # Si no hay etiquetas significativas en el patrón, etiquetar como no confiable
        if len(pattern_labels) == 0:
            # Patrón confiable pero sin direccionalidad clara - etiquetar como 2.0
            for i in range(idx_window_start, signal_time_idx + 1):
                labels[i] = 2.0  # No confiable (sin direccionalidad clara)
            continue
            
        # Calculamos la etiqueta promedio de todos los puntos del patrón
        avg_label = 0.0
        for l in pattern_labels:
            avg_label += l
        avg_label /= len(pattern_labels)
        
        # Determinamos la etiqueta general para todo el patrón
        if direction == 2:  # both directions
            pattern_label = 0.0 if avg_label < 0.5 else 1.0
        else:  # single direction
            pattern_label = 1.0 if avg_label >= 0.5 else 0.0
        
        # Asignamos esta etiqueta a todos los puntos del patrón
        for i in range(idx_window_start, signal_time_idx + 1):
            labels[i] = pattern_label
    
    return labels

def get_labels_fractal_patterns(
    dataset,
    label_min_window=6,
    label_max_window=60,
    label_corr_threshold=0.7,
    label_min_val=5,
    label_max_val=5,
    label_markup=0.5,
    label_atr_period=14,
    direction=2
) -> pd.DataFrame:
    """
    Genera etiquetas basadas en patrones fractales simétricos con profit target escalado por ATR.
    
    Args:
        dataset: DataFrame con columna 'close'
        label_min_window: Tamaño mínimo de ventana para patrones
        label_max_window: Tamaño máximo de ventana para patrones
        label_corr_threshold: Umbral de correlación para patrones válidos
        label_min_val: Horizonte mínimo de predicción en barras
        label_max_val: Horizonte máximo de predicción en barras
        label_markup: Multiplicador de ATR para profit target
        label_atr_period: Período para cálculo de ATR
        direction: Dirección de las señales (0=buy, 1=sell, 2=both)
    
    Returns:
        DataFrame con columna 'labels_main' agregada
    """
    # Prepare data for label calculation
    close = dataset['close'].values
    high = dataset['high'].values
    low = dataset['low'].values
    atr = calculate_atr_simple(high, low, close, period=label_atr_period)
    # Calculate correlations and best window sizes
    correlations_at_start, best_window_sizes_at_start = calculate_symmetric_correlation_dynamic(
        close,
        label_min_window,
        label_max_window,
    )
    # Calculate labels
    labels = calculate_future_outcome_labels_for_patterns(
        len(close),
        correlations_at_start,
        best_window_sizes_at_start,
        close,
        atr,
        label_corr_threshold,
        label_min_val,
        label_max_val,
        label_markup,
        direction
    )
    # ✅ CORRECCIÓN CRÍTICA: Alineación correcta de etiquetas
    # Las etiquetas se generan para toda la longitud de datos, alinear correctamente
    labels_series = pd.Series(labels, index=dataset.index)
    dataset['labels_main'] = labels_series
    dataset['labels_main'] = dataset['labels_main'].fillna(2.0)
    return dataset

def safe_savgol_filter(x, label_rolling: int, label_polyorder: int):
    """
    Aplica filtro Savitzky-Golay de forma segura con validaciones robustas.
    
    Este filtro suaviza los datos usando regresión polinómica local, útil para:
    - Eliminar ruido manteniendo características importantes
    - Calcular derivadas suavizadas
    - Preservar picos y valles significativos

    Parameters
    ----------
    x : array-like
        Array de entrada (generalmente precios de cierre)
    label_rolling : int
        Tamaño de ventana deseado (se ajustará automáticamente si es necesario)
    label_polyorder : int
        Orden polinomial para el ajuste

    Returns
    -------
    tuple
        (filtered_array, filtering_successful)
        - filtered_array: Array suavizado o array original si falló el filtrado
        - filtering_successful: Boolean indicando si Savitzky-Golay se aplicó exitosamente
    """
    n = len(x)
    
    # ✅ VALIDACIÓN MEJORADA: Verificar datos suficientes
    if n <= label_polyorder:
        return x, False  # No hay suficientes datos

    # ✅ OPTIMIZACIÓN: Ajustar ventana para que sea impar
    wl = int(label_rolling)
    if wl % 2 == 0:
        wl += 1
    
    # ✅ VALIDACIÓN ROBUSTA: Calcular ventana máxima permitida
    max_wl = n if n % 2 == 1 else n - 1
    wl = min(wl, max_wl)
    
    # ✅ CORRECCIÓN CRÍTICA: Asegurar ventana mínima válida
    if wl <= label_polyorder:
        wl = label_polyorder + 1 if (label_polyorder + 1) % 2 == 1 else label_polyorder + 2
        wl = min(wl, max_wl)
        if wl <= label_polyorder:
            return x, False  # Ventana demasiado pequeña

    try:
        # ✅ APLICACIÓN SEGURA: Filtro con manejo de excepciones
        filtered_data = savgol_filter(x, window_length=wl, polyorder=label_polyorder)
        
        # ✅ VALIDACIÓN DE RESULTADOS: Verificar que no hay NaN/Inf
        if np.any(np.isnan(filtered_data)) or np.any(np.isinf(filtered_data)):
            return x, False
            
        return filtered_data, True  # Filtrado exitoso
        
    except Exception:
        # ✅ MANEJO DE ERRORES: Retornar datos originales si algo falla
        return x, False

@njit(cache=True)
def calculate_labels_trend(
    close, atr, normalized_trend, label_threshold, label_markup, label_min_val, label_max_val, direction, method_int=5
):
    """
    Etiquetado de tendencia con profit, soportando dirección.

    Procedimiento de etiquetado:
    1. Se calcula la tendencia normalizada (por volatilidad) para cada punto.
    2. Para cada punto, se evalúa si la tendencia supera un umbral positivo (buy) o negativo (sell).
    3. Si la señal es válida según la dirección (buy/sell/both), se busca el precio futuro en una ventana [min_val, max_val] usando el método elegido:
       - 'first', 'last', 'mean', 'max', 'min', o 'random' (ver parámetro method_int).
    4. Se compara el precio futuro con el precio actual más/menos un objetivo de profit dinámico (ATR * markup).
    5. Para direcciones únicas (buy/sell): 
       - 1.0 = éxito direccional (profit alcanzado en la dirección)
       - 0.0 = fracaso direccional (no se alcanza el profit)
       - 2.0 = patrón no confiable (la tendencia no supera el umbral)
    6. Para dirección 'both':
       - 0.0 = éxito buy, 1.0 = éxito sell, 2.0 = no hay señal o no se alcanza el profit

    direction: 0=solo buy, 1=solo sell, 2=both
    method_int: 0=first, 1=last, 2=mean, 3=max, 4=min, 5=random
    Usa ATR como filtro de profit.
    """
    labels = np.empty(len(normalized_trend) - label_max_val, dtype=np.float64)
    for i in range(len(normalized_trend) - label_max_val):
        dyn_mk = label_markup * atr[i]
        if direction == 0:  # solo buy
            if normalized_trend[i] > label_threshold:
                window = close[i + label_min_val : i + label_max_val + 1]
                if window.size == 0:
                    future_pr = close[i + label_min_val] if i + label_min_val < len(close) else close[i]
                elif method_int == 0:  # first
                    future_pr = close[i + label_min_val]
                elif method_int == 1:  # last
                    future_pr = close[i + label_max_val]
                elif method_int == 2:  # mean
                    future_pr = np.mean(window)
                elif method_int == 3:  # max
                    future_pr = np.max(window)
                elif method_int == 4:  # min
                    future_pr = np.min(window)
                else:  # random
                    rand = np.random.randint(label_min_val, label_max_val + 1)
                    future_pr = close[i + rand]
                if future_pr >= close[i] + dyn_mk:
                    labels[i] = 1.0  # Éxito direccional (buy)
                else:
                    labels[i] = 0.0  # Fracaso direccional (buy)
            else:
                labels[i] = 2.0  # Patrón no confiable
        elif direction == 1:  # solo sell
            if normalized_trend[i] < -label_threshold:
                window = close[i + label_min_val : i + label_max_val + 1]
                if window.size == 0:
                    future_pr = close[i + label_min_val] if i + label_min_val < len(close) else close[i]
                elif method_int == 0:  # first
                    future_pr = close[i + label_min_val]
                elif method_int == 1:  # last
                    future_pr = close[i + label_max_val]
                elif method_int == 2:  # mean
                    future_pr = np.mean(window)
                elif method_int == 3:  # max
                    future_pr = np.max(window)
                elif method_int == 4:  # min
                    future_pr = np.min(window)
                else:  # random
                    rand = np.random.randint(label_min_val, label_max_val + 1)
                    future_pr = close[i + rand]
                if future_pr <= close[i] - dyn_mk:
                    labels[i] = 1.0  # Éxito direccional (sell)
                else:
                    labels[i] = 0.0  # Fracaso direccional (sell)
            else:
                labels[i] = 2.0  # Patrón no confiable
        else:  # both
            if normalized_trend[i] > label_threshold:
                window = close[i + label_min_val : i + label_max_val + 1]
                if window.size == 0:
                    future_pr = close[i + label_min_val] if i + label_min_val < len(close) else close[i]
                elif method_int == 0:  # first
                    future_pr = close[i + label_min_val]
                elif method_int == 1:  # last
                    future_pr = close[i + label_max_val]
                elif method_int == 2:  # mean
                    future_pr = np.mean(window)
                elif method_int == 3:  # max
                    future_pr = np.max(window)
                elif method_int == 4:  # min
                    future_pr = np.min(window)
                else:  # random
                    rand = np.random.randint(label_min_val, label_max_val + 1)
                    future_pr = close[i + rand]
                if future_pr >= close[i] + dyn_mk:
                    labels[i] = 0.0  # Buy (Profit reached)
                else:
                    labels[i] = 2.0  # No profit
            elif normalized_trend[i] < -label_threshold:
                window = close[i + label_min_val : i + label_max_val + 1]
                if window.size == 0:
                    future_pr = close[i + label_min_val] if i + label_min_val < len(close) else close[i]
                elif method_int == 0:  # first
                    future_pr = close[i + label_min_val]
                elif method_int == 1:  # last
                    future_pr = close[i + label_max_val]
                elif method_int == 2:  # mean
                    future_pr = np.mean(window)
                elif method_int == 3:  # max
                    future_pr = np.max(window)
                elif method_int == 4:  # min
                    future_pr = np.min(window)
                else:  # random
                    rand = np.random.randint(label_min_val, label_max_val + 1)
                    future_pr = close[i + rand]
                if future_pr <= close[i] - dyn_mk:
                    labels[i] = 1.0  # Sell (Profit reached)
                else:
                    labels[i] = 2.0  # No profit
            else:
                labels[i] = 2.0  # No signal
    return labels

def get_labels_trend(
    dataset,
    label_rolling=200,
    label_polyorder=3,
    label_threshold=0.5,
    label_vol_window=50,
    label_markup=0.5,
    label_min_val=1,
    label_max_val=15,
    label_atr_period=14,
    direction=2,  # 0=buy, 1=sell, 2=both
    label_method_random='random'
) -> pd.DataFrame:
    """
    Etiquetado de tendencia normalizada con validación de profit objetivo usando ATR y markup.

    Descripción del procedimiento de etiquetado:
    1. Se suaviza la serie de precios de cierre usando un filtro Savitzky-Golay.
    2. Se calcula la tendencia como el gradiente de la serie suavizada.
    3. Se normaliza la tendencia dividiéndola por la volatilidad local (desviación estándar rolling).
    4. Se eliminan los valores NaN y se sincronizan los datos.
    5. Se calcula el ATR para cada punto.
    6. Para cada punto, se evalúa si la tendencia normalizada supera el umbral (positivo para buy, negativo para sell).
    7. Si la señal es válida, se busca el precio futuro en una ventana [min_val, max_val] usando el método seleccionado ('first', 'last', 'mean', 'max', 'min', 'random').
    8. Se compara el precio futuro con el precio actual más/menos el objetivo de profit (ATR * markup).
    9. Se asigna la etiqueta según el esquema:
       - Para direcciones únicas (buy/sell): 1.0 = éxito direccional, 0.0 = fracaso, 2.0 = patrón no confiable.
       - Para dirección 'both': 0.0 = éxito buy, 1.0 = éxito sell, 2.0 = no señal/no profit.

    Parámetros principales:
    - direction: 0=solo buy, 1=solo sell, 2=ambas direcciones
    - label_method_random: método de selección del precio objetivo en la ventana futura
    """
    # Smoothing and trend calculation
    smoothed_prices, filtering_successful = safe_savgol_filter(
        dataset['close'].values, label_rolling=label_rolling, label_polyorder=label_polyorder
    )
    if not filtering_successful:
        return pd.DataFrame()
    trend = np.gradient(smoothed_prices)

    # Removing NaN and synchronizing data
    vol = bn.move_std(dataset['close'].values, window=label_vol_window, min_count=1)
    normalized_trend = np.where(vol != 0, trend / vol, np.nan)
    valid_mask = ~np.isnan(normalized_trend)
    normalized_trend_clean = normalized_trend[valid_mask]
    close_clean = dataset['close'].values[valid_mask]

    high = dataset['high'].values
    low = dataset['low'].values
    close = dataset['close'].values
    atr = calculate_atr_simple(high, low, close, period=label_atr_period)
    atr_clean = atr[valid_mask]

    # Generating labels
    method_map = {'first': 0, 'last': 1, 'mean': 2, 'max': 3, 'min': 4, 'random': 5}
    method_int = method_map.get(label_method_random, 5)
    labels = calculate_labels_trend(
        close_clean,
        atr_clean,
        normalized_trend_clean,
        label_threshold,
        label_markup,
        label_min_val,
        label_max_val,
        direction,
        method_int
    )

    # ✅ CORRECCIÓN CRÍTICA: Alineación correcta de etiquetas
    # El array de etiquetas tiene longitud (len(close_clean) - label_max_val)
    # Debe alinearse con los índices válidos originales
    if len(labels) > 0:
        valid_indices = dataset.index[valid_mask]
        label_indices = valid_indices[:len(labels)]
        labels_series = pd.Series(labels, index=label_indices)
        labels_aligned = labels_series.reindex(dataset.index, fill_value=2.0)
    else:
        labels_aligned = pd.Series(2.0, index=dataset.index)

    # Trimming the dataset and adding labels
    dataset['labels_main'] = labels_aligned
    dataset['labels_main'] = dataset['labels_main'].fillna(2.0)
    return dataset

@njit(cache=True)
def calculate_labels_trend_multi(
    close, atr, normalized_trends, label_threshold, label_markup, label_min_val, label_max_val, direction=2, method_int=5
):
    """
    Etiquetado multi-período con soporte para direcciones únicas o ambas.
    direction: 0=solo buy, 1=solo sell, 2=ambas
    method_int: 0=first, 1=last, 2=mean, 3=max, 4=min, 5=random
    """
    num_periods = normalized_trends.shape[0]  # Number of periods
    labels = np.empty(len(close) - label_max_val, dtype=np.float64)
    for i in range(len(close) - label_max_val):
        dyn_mk = label_markup * atr[i]
        # Select the target price using the specified method
        window = close[i + label_min_val : i + label_max_val + 1]
        if window.size == 0:
            future_price = close[i + label_min_val] if i + label_min_val < len(close) else close[i]
        elif method_int == 0:  # first
            future_price = close[i + label_min_val]
        elif method_int == 1:  # last
            future_price = close[i + label_max_val]
        elif method_int == 2:  # mean
            future_price = np.mean(window)
        elif method_int == 3:  # max
            future_price = np.max(window)
        elif method_int == 4:  # min
            future_price = np.min(window)
        else:  # random
            rand = np.random.randint(label_min_val, label_max_val + 1)
            future_price = close[i + rand]
        buy_signals = 0
        sell_signals = 0
        # Check conditions for each period
        for j in range(num_periods):
            if normalized_trends[j, i] > label_threshold:
                if future_price >= close[i] + dyn_mk:
                    buy_signals += 1
            elif normalized_trends[j, i] < -label_threshold:
                if future_price <= close[i] - dyn_mk:
                    sell_signals += 1
        # Etiquetado según dirección
        if direction == 2:
            # Esquema clásico: 0.0=buy, 1.0=sell, 2.0=no señal/conflicto
            if buy_signals > 0 and sell_signals == 0:
                labels[i] = 0.0  # Buy
            elif sell_signals > 0 and buy_signals == 0:
                labels[i] = 1.0  # Sell
            else:
                labels[i] = 2.0  # No signal or conflict
        elif direction == 0:
            # Solo buy: 1.0=éxito, 0.0=fracaso, 2.0=no confiable
            if buy_signals > 0 and sell_signals == 0:
                labels[i] = 1.0  # Éxito direccional (buy)
            elif sell_signals > 0 and buy_signals == 0:
                labels[i] = 0.0  # Fracaso direccional (no buy, sino sell)
            else:
                labels[i] = 2.0  # Patrón no confiable
        elif direction == 1:
            # Solo sell: 1.0=éxito, 0.0=fracaso, 2.0=no confiable
            if sell_signals > 0 and buy_signals == 0:
                labels[i] = 1.0  # Éxito direccional (sell)
            elif buy_signals > 0 and sell_signals == 0:
                labels[i] = 0.0  # Fracaso direccional (no sell, sino buy)
            else:
                labels[i] = 2.0  # Patrón no confiable
        else:
            labels[i] = 2.0  # fallback
    return labels

def get_labels_trend_multi(
    dataset,
    label_filter='savgol',
    label_rolling_periods_small=[10, 20, 30],
    label_polyorder=3,
    label_threshold=0.5,
    label_vol_window=50,
    label_markup=0.5,
    label_min_val=1,
    label_max_val=15,
    label_atr_period=14,
    direction=2,
    label_method_random='random'
) -> pd.DataFrame:
    """
    Generates labels for trading signals (Buy/Sell) based on the normalized trend,
    calculated for multiple smoothing periods, con soporte para direcciones únicas o ambas.

    Args:
        dataset (pd.DataFrame): DataFrame with data, containing the 'close' column.
        label_filter (str): Smoothing label_filter ('savgol', 'spline', 'sma', 'ema').
        label_rolling_periods_small (list): List of smoothing window sizes. Default is [10, 20, 30].
        label_polyorder (int): Polynomial order for 'savgol' and 'spline' methods.
        label_threshold (float): Threshold for the normalized trend.
        label_vol_window (int): Window for volatility calculation.
        label_markup (float): Minimum profit to confirm the signal.
        label_min_val (int): Minimum number of bars forward.
        label_max_val (int): Maximum number of bars forward.
        label_atr_period (int): ATR period.
        direction (int): 0=solo buy, 1=solo sell, 2=ambas (default).

    Returns:
        pd.DataFrame: DataFrame with added 'labels_main' column:
                      - 0.0/1.0/2.0 según esquema fractal MQL5.
    """
    close_prices = dataset['close'].values
    normalized_trends = []

    # Calculate normalized trend for each period
    for label_rolling in label_rolling_periods_small:
        if label_filter == 'savgol':
            smoothed_prices, filtering_successful = safe_savgol_filter(close_prices, label_rolling=label_rolling, label_polyorder=label_polyorder)
            if not filtering_successful:
                return pd.DataFrame()
        elif label_filter == 'spline':
            x = np.arange(len(close_prices))
            spline = UnivariateSpline(x, close_prices, k=label_polyorder, s=label_rolling)
            smoothed_prices = spline(x)
        elif label_filter == 'sma':
            smoothed_prices = bn.move_mean(close_prices, window=label_rolling, min_count=1)
        elif label_filter == 'ema':
            smoothed_prices = bn.move_exp_mean(close_prices, window=label_rolling, min_count=1)
        else:
            raise ValueError(f"Unknown smoothing label_filter: {label_filter}")
        
        trend = np.gradient(smoothed_prices)
        vol = bn.move_std(close_prices, window=label_vol_window, min_count=1)
        normalized_trend = np.where(vol != 0, trend / vol, np.nan)
        normalized_trends.append(normalized_trend)

    # Transform list into 2D array
    normalized_trends_array = np.vstack(normalized_trends)

    # Remove rows with NaN
    valid_mask = ~np.isnan(normalized_trends_array).any(axis=0)
    normalized_trends_clean = normalized_trends_array[:, valid_mask]
    close_clean = close_prices[valid_mask]

    high = dataset['high'].values
    low = dataset['low'].values
    atr = calculate_atr_simple(high, low, close_prices, period=label_atr_period)
    atr_clean = atr[valid_mask]
    # Generate labels
    method_map = {'first': 0, 'last': 1, 'mean': 2, 'max': 3, 'min': 4, 'random': 5}
    method_int = method_map.get(label_method_random, 5)
    labels = calculate_labels_trend_multi(
        close_clean,
        atr_clean,
        normalized_trends_clean,
        label_threshold,
        label_markup,
        label_min_val,
        label_max_val,
        direction=direction,
        method_int=method_int
    )

    # Asignar etiquetas alineadas y rellenar con 2.0 usando pandas para evitar padding manual
    labels = pd.Series(labels, index=dataset.index[valid_mask][:-label_max_val]).reindex_like(dataset).fillna(2.0)

    # Trimming the dataset and adding labels
    dataset['labels_main'] = labels
    dataset['labels_main'] = dataset['labels_main'].fillna(2.0)
    return dataset

@njit(cache=True)
def calculate_labels_trend_filters(close, atr, normalized_trend, label_threshold, label_markup, label_min_val, label_max_val, direction=2, method_int=5):
    """
    Etiquetado de tendencia con profit, soportando dirección.
    
    direction: 0=solo buy, 1=solo sell, 2=ambas
    method_int: 0=first, 1=last, 2=mean, 3=max, 4=min, 5=random
    
    Etiquetas:
      - Direccional único: 1.0=éxito direccional, 0.0=fracaso direccional, 2.0=patrón no confiable
      - Ambas: 0.0=éxito buy, 1.0=éxito sell, 2.0=sin señal/no profit
    """
    labels = np.empty(len(normalized_trend) - label_max_val, dtype=np.float64)
    for i in range(len(normalized_trend) - label_max_val):
        dyn_mk = label_markup * atr[i]
        # Calcular future_pr directamente según method_int
        window = close[i + label_min_val : i + label_max_val + 1]
        if window.size == 0:
            future_pr = close[i + label_min_val] if i + label_min_val < len(close) else close[i]
        elif method_int == 0:  # first
            future_pr = close[i + label_min_val]
        elif method_int == 1:  # last
            future_pr = close[i + label_max_val]
        elif method_int == 2:  # mean
            future_pr = np.mean(window)
        elif method_int == 3:  # max
            future_pr = np.max(window)
        elif method_int == 4:  # min
            future_pr = np.min(window)
        else:  # random
            rand = np.random.randint(label_min_val, label_max_val + 1)
            future_pr = close[i + rand]

        if direction == 0:  # solo buy
            if normalized_trend[i] > label_threshold:
                if future_pr >= close[i] + dyn_mk:
                    labels[i] = 1.0  # Éxito direccional (buy)
                else:
                    labels[i] = 0.0  # Fracaso direccional (buy)
            else:
                labels[i] = 2.0  # Patrón no confiable
        elif direction == 1:  # solo sell
            if normalized_trend[i] < -label_threshold:
                if future_pr <= close[i] - dyn_mk:
                    labels[i] = 1.0  # Éxito direccional (sell)
                else:
                    labels[i] = 0.0  # Fracaso direccional (sell)
            else:
                labels[i] = 2.0  # Patrón no confiable
        else:  # both
            if normalized_trend[i] > label_threshold:
                if future_pr >= close[i] + dyn_mk:
                    labels[i] = 0.0  # Buy (Profit reached)
                else:
                    labels[i] = 2.0  # No profit
            elif normalized_trend[i] < -label_threshold:
                if future_pr <= close[i] - dyn_mk:
                    labels[i] = 1.0  # Sell (Profit reached)
                else:
                    labels[i] = 2.0  # No profit
            else:
                labels[i] = 2.0  # No signal
    return labels

def get_labels_trend_filters(dataset, label_filter='savgol', label_rolling=200, label_polyorder=3, label_threshold=0.5, 
                    label_vol_window=50, label_markup=0.5, label_min_val=1, label_max_val=15, label_atr_period=14, label_method_random='random', direction=2) -> pd.DataFrame:
    """
    Etiquetado de tendencia con profit usando diferentes filtros de suavizado, con soporte para direcciones únicas o ambas.
    
    Args:
        dataset (pd.DataFrame): DataFrame con datos, conteniendo la columna 'close'.
        label_filter (str): Filtro de suavizado ('savgol', 'spline', 'sma', 'ema').
        label_rolling (int): Tamaño de ventana para suavizado.
        label_polyorder (int): Orden polinomial para 'savgol' y 'spline'.
        label_threshold (float): Umbral para la tendencia normalizada.
        label_vol_window (int): Ventana para cálculo de volatilidad.
        label_markup (float): Profit mínimo para confirmar la señal.
        label_min_val (int): Número mínimo de barras hacia adelante.
        label_max_val (int): Número máximo de barras hacia adelante.
        label_atr_period (int): Período ATR.
        label_method_random (str): Método de selección del precio objetivo.
        direction (int): 0=solo buy, 1=solo sell, 2=ambas (default).
    
    Returns:
        pd.DataFrame: DataFrame con columna 'labels_main' añadida:
                      - Para direcciones únicas: 1.0=éxito direccional, 0.0=fracaso direccional, 2.0=patrón no confiable
                      - Para ambas: 0.0=éxito buy, 1.0=éxito sell, 2.0=sin señal/no profit
    """
    # Smoothing and trend calculation
    close_prices = dataset['close'].values
    if label_filter == 'savgol':
        smoothed_prices, filtering_successful = safe_savgol_filter(close_prices, label_rolling=label_rolling, label_polyorder=label_polyorder)
        if not filtering_successful:
            return pd.DataFrame()
    elif label_filter == 'spline':
        x = np.arange(len(close_prices))
        spline = UnivariateSpline(x, close_prices, k=label_polyorder, s=label_rolling)
        smoothed_prices = spline(x)
    elif label_filter == 'sma':
        smoothed_prices = bn.move_mean(close_prices, window=label_rolling, min_count=1)
    elif label_filter == 'ema':
        smoothed_series = pd.Series(close_prices).ewm(span=label_rolling, adjust=False).mean()
        smoothed_prices = smoothed_series.values
    else:
        raise ValueError(f"Unknown smoothing label_filter: {label_filter}")
    
    trend = np.gradient(smoothed_prices)
    
    # Normalizing the trend by volatility (usando bottleneck para acelerar)
    vol = bn.move_std(dataset['close'].values, window=label_vol_window, min_count=1)
    normalized_trend = np.where(vol != 0, trend / vol, np.nan)
    
    # Removing NaN and synchronizing data
    valid_mask = ~np.isnan(normalized_trend)
    normalized_trend_clean = normalized_trend[valid_mask]
    close_clean = dataset['close'].values[valid_mask]
    dataset_clean = dataset[valid_mask].copy()
    high = dataset['high'].values
    low = dataset['low'].values
    close = dataset['close'].values
    atr = calculate_atr_simple(high, low, close, period=label_atr_period)
    atr_clean = atr[valid_mask]
    
    # Generating labels
    method_map = {'first': 0, 'last': 1, 'mean': 2, 'max': 3, 'min': 4, 'random': 5}
    method_int = method_map.get(label_method_random, 5)
    labels = calculate_labels_trend_filters(close_clean, atr_clean, normalized_trend_clean, label_threshold, label_markup, label_min_val, label_max_val, direction, method_int)
    
    # Asignar etiquetas alineadas y rellenar con 2.0 usando pandas para evitar padding manual
    labels = pd.Series(labels, index=dataset.index[valid_mask][:-label_max_val]).reindex_like(dataset).fillna(2.0)
    
    # Trimming the dataset and adding labels
    dataset['labels_main'] = labels
    dataset['labels_main'] = dataset['labels_main'].fillna(2.0)
    return dataset

@njit(cache=True)
def calculate_labels_clusters(close_data, atr, clusters, label_markup, direction=2):
    """
    Etiquetado de saltos de cluster con soporte para direcciones únicas o ambas.
    
    ✅ METODOLOGÍA: Detecta cambios significativos de cluster acompañados 
    de movimientos de precio superiores al umbral dinámico (ATR * markup).
    
    Args:
        close_data: Array de precios de cierre
        atr: Array de valores ATR
        clusters: Array de asignaciones de cluster
        label_markup: Multiplicador de ATR para umbral de movimiento
        direction: 0=solo buy, 1=solo sell, 2=ambas
        
    Returns:
        Array de etiquetas:
        - Direccional único: 1.0=éxito direccional, 0.0=fracaso direccional, 2.0=patrón no confiable
        - Ambas: 0.0=salto alcista, 1.0=salto bajista, 2.0=sin señal
    """
    n = len(close_data)
    labels = np.full(n, 2.0, dtype=np.float64)  # ✅ INICIALIZACIÓN: Tamaño correcto
    
    if n < 2:
        return labels
    
    current_cluster = clusters[0]
    last_price = close_data[0]
    
    # ✅ OPTIMIZACIÓN: Procesar desde el segundo elemento
    for i in range(1, n):
        next_cluster = clusters[i]
        dyn_mk = label_markup * atr[i] if i < len(atr) else label_markup * atr[-1]
        price_diff = close_data[i] - last_price
        
        # ✅ CRITERIO DE SALTO: Cambio de cluster + movimiento significativo
        jump = (next_cluster != current_cluster) and (abs(price_diff) > dyn_mk)
        
        if direction == 0:  # solo buy
            if jump:
                if price_diff > 0:
                    labels[i] = 1.0  # Éxito direccional (sube tras salto)
                else:
                    labels[i] = 0.0  # Fracaso direccional (baja tras salto)
                current_cluster = next_cluster
                last_price = close_data[i]
            else:
                labels[i] = 2.0  # Patrón no confiable
        elif direction == 1:  # solo sell
            if jump:
                if price_diff < 0:
                    labels[i] = 1.0  # Éxito direccional (baja tras salto)
                else:
                    labels[i] = 0.0  # Fracaso direccional (sube tras salto)
                current_cluster = next_cluster
                last_price = close_data[i]
            else:
                labels[i] = 2.0  # Patrón no confiable
        else:  # both
            if jump:
                if price_diff > 0:
                    labels[i] = 0.0  # Salto alcista
                else:
                    labels[i] = 1.0  # Salto bajista
                current_cluster = next_cluster
                last_price = close_data[i]
            else:
                labels[i] = 2.0  # Sin señal
    
    return labels

def get_labels_clusters(
    dataset, 
    label_markup, 
    label_n_clusters=20, 
    label_atr_period=14, 
    direction=2  # 0=buy, 1=sell, 2=both
) -> pd.DataFrame:
    """
    Etiquetado de saltos de cluster con soporte para direcciones únicas o ambas.
    direction: 0=solo buy, 1=solo sell, 2=ambas
    """

    kmeans = KMeans(n_clusters=label_n_clusters, n_init='auto')
    dataset['cluster'] = kmeans.fit_predict(dataset[['close']])
    clusters = dataset['cluster'].values
    high = dataset['high'].values
    low = dataset['low'].values
    close = dataset['close'].values
    atr = calculate_atr_simple(high, low, close, period=label_atr_period)
    labels = calculate_labels_clusters(close, atr, clusters, label_markup, direction=direction)
    # Trimming the dataset and adding labels
    dataset['labels_main'] = labels
    dataset['labels_main'] = dataset['labels_main'].fillna(2.0)
    return dataset.drop(columns=['cluster'])

@njit(cache=True)
def calculate_labels_multi_window(prices, atr, window_sizes, label_markup, label_min_val, label_max_val, direction=2, method_int=5):
    """
    Etiquetado multi-ventana con profit target basado en ATR * markup.
    direction: 0=solo buy, 1=solo sell, 2=ambas
    method_int: 0=first, 1=last, 2=mean, 3=max, 4=min, 5=random
    Etiquetas:
      - Direccional único: 1.0=éxito direccional, 0.0=fracaso direccional, 2.0=patrón no confiable
      - Ambas: 0.0=buy, 1.0=sell, 2.0=sin señal
    """
    max_window = max(window_sizes)
    signals = []
    for i in range(max_window, len(prices) - label_max_val):
        dyn_mk = label_markup * atr[i]
        long_signals = 0
        short_signals = 0
        
        for window_size in window_sizes:
            window = prices[i-window_size:i]
            resistance = np.max(window)
            support = np.min(window)
            current_price = prices[i]
            
            # Usar ATR dinámico en lugar de porcentaje fijo
            if current_price > resistance + dyn_mk:
                long_signals += 1
            elif current_price < support - dyn_mk:
                short_signals += 1

        # Validar con profit futuro
        window = prices[i + label_min_val : i + label_max_val + 1]
        if window.size == 0:
            future_price = prices[i + label_min_val] if i + label_min_val < len(prices) else prices[i]
        elif method_int == 0:  # first
            future_price = prices[i + label_min_val]
        elif method_int == 1:  # last
            future_price = prices[i + label_max_val]
        elif method_int == 2:  # mean
            future_price = np.mean(window)
        elif method_int == 3:  # max
            future_price = np.max(window)
        elif method_int == 4:  # min
            future_price = np.min(window)
        else:  # random
            rand = np.random.randint(label_min_val, label_max_val + 1)
            future_price = prices[i + rand]
        
        if direction == 2:  # both
            if long_signals > short_signals and future_price >= current_price + dyn_mk:
                signals.append(0.0)  # buy with profit validation
            elif short_signals > long_signals and future_price <= current_price - dyn_mk:
                signals.append(1.0)  # sell with profit validation
            else:
                signals.append(2.0)
        elif direction == 0:  # solo buy
            if long_signals > 0 and future_price >= current_price + dyn_mk:
                signals.append(1.0)  # éxito direccional (señal de compra)
            elif long_signals > 0:
                signals.append(0.0)  # fracaso direccional (no profit)
            else:
                signals.append(2.0)  # patrón no confiable
        elif direction == 1:  # solo sell
            if short_signals > 0 and future_price <= current_price - dyn_mk:
                signals.append(1.0)  # éxito direccional (señal de venta)
            elif short_signals > 0:
                signals.append(0.0)  # fracaso direccional (no profit)
            else:
                signals.append(2.0)  # patrón no confiable
    return signals

def get_labels_multi_window(
    dataset, 
    label_window_sizes_int=[20, 50, 100], 
    label_markup=0.5,
    label_min_val=1,
    label_max_val=15,
    label_atr_period=14,
    label_method_random='random',
    direction=2,
) -> pd.DataFrame:
    """
    Etiquetado multi-ventana con profit target basado en ATR * markup.
    direction: 0=solo buy, 1=solo sell, 2=ambas
    """

    # Calculate ATR
    high = dataset['high'].values
    low = dataset['low'].values
    close = dataset['close'].values
    atr = calculate_atr_simple(high, low, close, period=label_atr_period)
    # Calculate labels
    window_sizes_t = List(label_window_sizes_int)
    method_map = {'first': 0, 'last': 1, 'mean': 2, 'max': 3, 'min': 4, 'random': 5}
    method_int = method_map.get(label_method_random, 5)
    labels = calculate_labels_multi_window(
        close, atr, window_sizes_t, label_markup, label_min_val, label_max_val, direction, method_int
    )
    # Align labels and fill with 2.0 using pandas to avoid manual padding
    labels = pd.Series(labels, index=dataset.index[max(label_window_sizes_int):]).reindex_like(dataset).fillna(2.0)
    # Add labels to dataset
    dataset['labels_main'] = labels
    dataset['labels_main'] = dataset['labels_main'].fillna(2.0)
    return dataset

@njit(cache=True)
def calculate_labels_validated_levels(prices, atr, window_size, label_markup, min_touches, label_min_val, label_max_val, direction=2):
    """
    Etiquetado de rupturas de niveles validados con profit target basado en ATR * markup.
    direction: 0=solo buy, 1=solo sell, 2=ambas
    Etiquetas:
      - Direccional único: 1.0=éxito direccional, 0.0=fracaso direccional, 2.0=patrón no confiable
      - Ambas: 0.0=buy, 1.0=sell, 2.0=sin señal
    """
    resistance_touches = {}
    support_touches = {}
    labels = []
    
    for i in range(window_size, len(prices) - label_max_val):
        window = prices[i-window_size:i]
        current_price = prices[i]
        dyn_mk = label_markup * atr[i]

        potential_resistance = np.max(window)
        potential_support = np.min(window)

        # Usar ATR dinámico para determinar toques de niveles
        for level in resistance_touches:
            if abs(current_price - level) <= dyn_mk:
                resistance_touches[level] += 1

        for level in support_touches:
            if abs(current_price - level) <= dyn_mk:
                support_touches[level] += 1

        if potential_resistance not in resistance_touches:
            resistance_touches[potential_resistance] = 1
        if potential_support not in support_touches:
            support_touches[potential_support] = 1

        valid_resistance = [level for level, touches in resistance_touches.items() if touches >= min_touches]
        valid_support = [level for level, touches in support_touches.items() if touches >= min_touches]

        # Usar ATR dinámico para detectar rupturas
        broke_resistance = valid_resistance and current_price > min(valid_resistance) + dyn_mk
        broke_support = valid_support and current_price < max(valid_support) - dyn_mk

        # Validar con profit futuro
        rand = np.random.randint(label_min_val, label_max_val + 1)
        future_price = prices[i + rand]

        if direction == 2:  # both
            if broke_resistance and future_price >= current_price + dyn_mk:
                labels.append(0.0)  # buy with profit validation
            elif broke_support and future_price <= current_price - dyn_mk:
                labels.append(1.0)  # sell with profit validation
            else:
                labels.append(2.0)  # sin señal
        elif direction == 0:  # solo buy
            if broke_resistance and future_price >= current_price + dyn_mk:
                labels.append(1.0)  # éxito direccional (señal de compra)
            elif broke_resistance:
                labels.append(0.0)  # fracaso direccional (no profit)
            else:
                labels.append(2.0)  # patrón no confiable
        elif direction == 1:  # solo sell
            if broke_support and future_price <= current_price - dyn_mk:
                labels.append(1.0)  # éxito direccional (señal de venta)
            elif broke_support:
                labels.append(0.0)  # fracaso direccional (no profit)
            else:
                labels.append(2.0)  # patrón no confiable

    return labels

def get_labels_validated_levels(
    dataset, 
    label_window_size=20, 
    label_markup=0.5,
    label_min_touches=2,
    label_min_val=1,
    label_max_val=15,
    label_atr_period=14,
    direction=2
) -> pd.DataFrame:
    """
    Etiquetado de rupturas de niveles validados con profit target basado en ATR * markup.
    direction: 0=solo buy, 1=solo sell, 2=ambas
    """
    # Calculate ATR
    high = dataset["high"].values
    low = dataset["low"].values
    close = dataset["close"].values
    atr = calculate_atr_simple(high, low, close, period=label_atr_period)
    # Calculate labels
    labels = calculate_labels_validated_levels(
        close, atr, label_window_size, label_markup, label_min_touches,
        label_min_val, label_max_val, direction
    )
    # Align labels and fill with 2.0 using pandas to avoid manual padding
    labels = pd.Series(labels, index=dataset.index[label_window_size:]).reindex_like(dataset).fillna(2.0)
    # Add labels to dataset
    dataset['labels_main'] = labels
    dataset['labels_main'] = dataset['labels_main'].fillna(2.0)
    return dataset

@njit(cache=True)
def calculate_labels_zigzag(peaks, troughs, close, atr, label_markup, label_min_val, label_max_val, direction=2):
    """
    Generates labels based on peaks and troughs with profit target validation using ATR * markup.

    Args:
        peaks (np.array): Indices of the peaks in the data.
        troughs (np.array): Indices of the troughs in the data.
        close (np.array): Array of close prices.
        atr (np.array): Array of ATR values.
        label_markup (float): Markup multiplier for ATR.
        label_min_val (int): Minimum bars for future validation.
        label_max_val (int): Maximum bars for future validation.
        direction (int): 0=solo buy, 1=solo sell, 2=ambas

    Returns:
        np.array: An array of labels with profit validation.
    """
    len_close = len(close) - label_max_val
    labels = np.empty(len_close, dtype=np.float64)
    labels.fill(2.0)  # Initialize all labels to 2.0 (no signal)
    
    last_peak_type = None  # None for the start, 'up' for peak, 'down' for trough
    
    for i in range(len_close):
        is_peak = False
        is_trough = False
        
        # Check if current index is a peak
        for j in range(len(peaks)):
            if i == peaks[j]:
                is_peak = True
                break
        
        # Check if current index is a trough
        for j in range(len(troughs)):
            if i == troughs[j]:
                is_trough = True
                break

        dyn_mk = label_markup * atr[i]
        rand = np.random.randint(label_min_val, label_max_val + 1)
        future_price = close[i + rand]
        current_price = close[i]

        if direction == 2:  # both
            if is_peak and future_price <= current_price - dyn_mk:
                labels[i] = 1.0  # Sell signal at peaks with profit validation
                last_peak_type = 'up'
            elif is_trough and future_price >= current_price + dyn_mk:
                labels[i] = 0.0  # Buy signal at troughs with profit validation
                last_peak_type = 'down'
            else:
                # Mantener última tendencia pero sin validación específica
                if last_peak_type == 'up':
                    labels[i] = 2.0  # No confirmar sin profit
                elif last_peak_type == 'down':
                    labels[i] = 2.0  # No confirmar sin profit
                else:
                    labels[i] = 2.0
        elif direction == 0:  # solo buy
            if is_trough and future_price >= current_price + dyn_mk:
                labels[i] = 1.0  # Éxito direccional (señal de compra)
                last_peak_type = 'down'
            elif is_trough:
                labels[i] = 0.0  # Fracaso direccional (no profit)
                last_peak_type = 'down'
            else:
                labels[i] = 2.0  # Patrón no confiable
        elif direction == 1:  # solo sell
            if is_peak and future_price <= current_price - dyn_mk:
                labels[i] = 1.0  # Éxito direccional (señal de venta)
                last_peak_type = 'up'
            elif is_peak:
                labels[i] = 0.0  # Fracaso direccional (no profit)
                last_peak_type = 'up'
            else:
                labels[i] = 2.0  # Patrón no confiable
                
    return labels

def get_labels_zigzag(
    dataset, 
    label_peak_prominence=0.1, 
    label_markup=0.5,
    label_min_val=1,
    label_max_val=15,
    label_atr_period=14,
    direction=2
) -> pd.DataFrame:
    """
    Generates labels for a financial dataset based on zigzag peaks and troughs with profit target validation.
    This function identifies peaks and troughs using scipy.signal and validates signals with ATR * markup.
    """
    # Prepare data for label calculation
    high = dataset["high"].values
    low = dataset["low"].values
    close = dataset["close"].values
    # Find peaks and troughs
    peaks, _ = find_peaks(close, prominence=label_peak_prominence)
    troughs, _ = find_peaks(-close, prominence=label_peak_prominence)
    # Calculate ATR
    atr = calculate_atr_simple(high, low, close, period=label_atr_period)
    # Calculate labels
    labels = calculate_labels_zigzag(
        peaks, troughs, close, atr, label_markup, label_min_val, label_max_val, direction
    ) 
    # Align labels and fill with 2.0 using pandas to avoid manual padding
    labels = pd.Series(labels, index=dataset.index[:-label_max_val]).reindex_like(dataset).fillna(2.0)
    # Add labels to dataset
    dataset['labels_main'] = labels
    dataset['labels_main'] = dataset['labels_main'].fillna(2.0)
    return dataset

@njit(cache=True)
def calculate_labels_mean_reversion(close, atr, lvl, label_markup, label_min_val, label_max_val, q, direction=2):
    """
    direction: 0=buy only, 1=sell only, 2=both
    For buy: 1.0=éxito direccional, 0.0=fracaso, 2.0=no confiable
    For sell: 1.0=éxito direccional, 0.0=fracaso, 2.0=no confiable
    For both: 0.0=buy, 1.0=sell, 2.0=no confiable
    """
    labels = np.empty(len(close) - label_max_val, dtype=np.float64)
    for i in range(len(close) - label_max_val):
        dyn_mk = label_markup * atr[i]
        rand = np.random.randint(label_min_val, label_max_val + 1)
        curr_pr = close[i]
        curr_lvl = lvl[i]
        future_pr = close[i + rand]

        if direction == 0:  # buy only
            if curr_lvl < q[0] and (future_pr - dyn_mk) > curr_pr:
                labels[i] = 1.0  # éxito direccional (compra)
            elif curr_lvl < q[0]:
                labels[i] = 0.0  # fracaso direccional (compra fallida)
            else:
                labels[i] = 2.0  # patrón no confiable
        elif direction == 1:  # sell only
            if curr_lvl > q[1] and (future_pr + dyn_mk) < curr_pr:
                labels[i] = 1.0  # éxito direccional (venta)
            elif curr_lvl > q[1]:
                labels[i] = 0.0  # fracaso direccional (venta fallida)
            else:
                labels[i] = 2.0  # patrón no confiable
        else:  # both directions
            if curr_lvl > q[1] and (future_pr + dyn_mk) < curr_pr:
                labels[i] = 1.0  # sell
            elif curr_lvl < q[0] and (future_pr - dyn_mk) > curr_pr:
                labels[i] = 0.0  # buy
            else:
                labels[i] = 2.0  # no confiable
    return labels

def get_labels_mean_reversion(
    dataset,
    label_markup,
    label_min_val=1,
    label_max_val=15,
    label_rolling=0.5,
    label_quantiles=[.45, .55],
    label_filter='spline',
    label_decay_factor=0.95,
    label_shift=0,
    label_atr_period=14,
    direction=2
) -> pd.DataFrame:
    """
    Generates labels for a financial dataset based on mean reversion principles, with unidirectional support.

    Args:
        dataset (pd.DataFrame): DataFrame containing financial data with a 'close' column.
        label_markup (float): The percentage label_markup used to determine buy/sell signals.
        label_min_val (int, optional): Minimum number of consecutive days the label_markup must hold. Defaults to 1.
        label_max_val (int, optional): Maximum number of consecutive days the label_markup is considered. Defaults to 15.
        label_rolling (float, optional): Rolling window size for smoothing/averaging. 
        label_quantiles (list, optional): Quantiles to define the "reversion zone". Defaults to [.45, .55].
        label_filter (str, optional): Method for calculating the price deviation.
        label_decay_factor (float, optional): Exponential decay factor for weighting past data.
        label_shift (int, optional): Shift the smoothed price data forward/backward.
        label_atr_period (int, optional): ATR period.
        direction (int, optional): 0=buy only, 1=sell only, 2=both. Defaults to 2.

    Returns:
        pd.DataFrame: The original DataFrame with a new 'labels_main' column and filtered rows.
    """

    # Calculate the price deviation ('lvl') based on the chosen label_filter, applying decay_factor
    if label_filter == 'mean':
        # Usando bottleneck para acelerar el cálculo de media móvil
        rolling_mean = bn.move_mean(dataset['close'].values, window=label_rolling, min_count=1)
        diff = dataset['close'].values - rolling_mean
        weighted_diff = diff * np.exp(np.arange(len(diff)) * label_decay_factor / len(diff))
        dataset['lvl'] = weighted_diff
    elif label_filter == 'spline':
        x = np.array(range(dataset.shape[0]))
        y = dataset['close'].values
        spl = UnivariateSpline(x, y, k=3, s=label_rolling)
        yHat = spl(np.linspace(min(x), max(x), num=x.shape[0]))
        yHat_shifted = np.roll(yHat, shift=label_shift) # Apply the shift 
        diff = dataset['close'] - yHat_shifted
        weighted_diff = diff * np.exp(np.arange(len(diff)) * label_decay_factor / len(diff))
        dataset['lvl'] = weighted_diff
        dataset = dataset.dropna() 
    elif label_filter == 'savgol':
        smoothed_prices, filtering_successful = safe_savgol_filter(dataset['close'].values, label_rolling=label_rolling, label_polyorder=5)
        if not filtering_successful:
            return pd.DataFrame()
        diff = dataset['close'] - smoothed_prices
        weighted_diff = diff * np.exp(np.arange(len(diff)) * label_decay_factor / len(diff))
        dataset['lvl'] = weighted_diff
    # Ensure label_max_val does not exceed dataset length
    label_max_val = min(int(label_max_val), max(len(dataset) - 1, 1))
    if len(dataset) <= label_max_val:
        return pd.DataFrame()
    q = tuple(dataset['lvl'].quantile(label_quantiles).to_list())  # Calculate quantiles for the 'reversion zone'
    # Prepare data for label calculation
    lvl = dataset['lvl'].values
    # Calculate buy/sell labels with unidirectional support
    high = dataset['high'].values
    low = dataset['low'].values
    close = dataset['close'].values
    atr = calculate_atr_simple(high, low, close, period=label_atr_period)
    labels = calculate_labels_mean_reversion(close, atr, lvl, label_markup, label_min_val, label_max_val, q, direction=direction)
    # Align labels and fill with 2.0 using pandas to avoid padding manual
    labels = pd.Series(labels, index=dataset.index[:-label_max_val]).reindex_like(dataset).fillna(2.0)
    # Add labels to dataset
    dataset['labels_main'] = labels
    dataset['labels_main'] = dataset['labels_main'].fillna(2.0)
    # Remove temporary columns
    return dataset.drop(columns=['lvl'])

@njit(cache=True)
def calculate_labels_mean_reversion_multi(
    close_data, atr, lvl_data, q_list, label_markup, label_min_val, label_max_val, window_sizes, direction=2
):
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

        # direction: buy=0, sell=1, both=2
        if direction == 0:  # buy
            if buy_condition and (future_pr - dyn_mk) > curr_pr:
                labels.append(1.0)  # éxito direccional
            elif buy_condition:
                labels.append(0.0)  # fracaso direccional
            else:
                labels.append(2.0)  # patrón no confiable
        elif direction == 1:  # sell
            if sell_condition and (future_pr + dyn_mk) < curr_pr:
                labels.append(1.0)  # éxito direccional
            elif sell_condition:
                labels.append(0.0)  # fracaso direccional
            else:
                labels.append(2.0)  # patrón no confiable
        else:  # both
            if sell_condition and (future_pr + dyn_mk) < curr_pr:
                labels.append(1.0)
            elif buy_condition and (future_pr - dyn_mk) > curr_pr:
                labels.append(0.0)
            else:
                labels.append(2.0)
    return labels

def get_labels_mean_reversion_multi(
    dataset, 
    label_markup, 
    label_min_val=1, 
    label_max_val=15, 
    label_window_sizes_float=[0.2, 0.3, 0.5], 
    label_quantiles=[.45, .55], 
    label_atr_period=14, 
    direction=2
) -> pd.DataFrame:

    q = np.empty((len(label_window_sizes_float), 2))  # Initialize as 2D NumPy array
    lvl_data = np.empty((dataset.shape[0], len(label_window_sizes_float)))

    for i, label_rolling in enumerate(label_window_sizes_float):
        x = np.arange(dataset.shape[0])
        y = dataset['close'].values
        spl = UnivariateSpline(x, y, k=3, s=label_rolling)
        yHat = spl(np.linspace(x.min(), x.max(), x.shape[0]))
        lvl_data[:, i] = dataset['close'] - yHat
        # Store quantiles directly into the NumPy array
        quantile_values = np.quantile(lvl_data[:, i], label_quantiles)
        q[i, 0] = quantile_values[0]
        q[i, 1] = quantile_values[1]

    label_max_val = min(int(label_max_val), max(len(dataset) - 1, 1))
    if len(dataset) <= label_max_val:
        return pd.DataFrame()
    # Prepare data for label calculation
    close = dataset['close'].values
    high = dataset["high"].values
    low = dataset["low"].values
    atr = calculate_atr_simple(high, low, close, period=label_atr_period)
    # Prepare parameters for label calculation
    windows_t = List(label_window_sizes_float)
    q_t = List([(float(q[i,0]), float(q[i,1])) for i in range(len(label_window_sizes_float))])
    labels = calculate_labels_mean_reversion_multi(
        close, atr, lvl_data, q_t, label_markup, label_min_val, label_max_val, windows_t, direction
    )
    # Align labels and fill with 2.0 using pandas to avoid padding
    labels = pd.Series(labels, index=dataset.index[:-label_max_val]).reindex_like(dataset).fillna(2.0)
    dataset['labels_main'] = labels
    dataset['labels_main'] = dataset['labels_main'].fillna(2.0)
    return dataset

@njit(cache=True)
def calculate_labels_mean_reversion_vol(
    close_data, atr, lvl_data, volatility_group, quantile_groups_low, quantile_groups_high,
    label_markup, label_min_val, label_max_val, direction=2
):
    """
    direction: 2=both, 0=buy only, 1=sell only
    Etiquetado compatible con la metodología de direcciones únicas:
      - 2.0: patrón no confiable (correlación baja o sin direccionalidad)
      - 1.0: éxito direccional (buy/sell según dirección)
      - 0.0: fracaso direccional (buy/sell según dirección)
    """
    labels = []
    for i in range(len(close_data) - label_max_val):
        dyn_mk = label_markup * atr[i]
        rand = np.random.randint(label_min_val, label_max_val + 1)
        curr_pr = close_data[i]
        curr_lvl = lvl_data[i]
        curr_vol_group = volatility_group[i]
        future_pr = close_data[i + rand]

        low_q = quantile_groups_low[int(curr_vol_group)]
        high_q = quantile_groups_high[int(curr_vol_group)]

        if direction == 0:  # solo buy
            if curr_lvl < low_q and (future_pr - dyn_mk) > curr_pr:
                labels.append(1.0)  # éxito direccional (compra)
            elif curr_lvl < low_q:
                labels.append(0.0)  # fracaso direccional (compra fallida)
            else:
                labels.append(2.0)  # patrón no confiable
        elif direction == 1:  # solo sell
            if curr_lvl > high_q and (future_pr + dyn_mk) < curr_pr:
                labels.append(1.0)  # éxito direccional (venta)
            elif curr_lvl > high_q:
                labels.append(0.0)  # fracaso direccional (venta fallida)
            else:
                labels.append(2.0)  # patrón no confiable
        else:  # both directions
            if curr_lvl > high_q and (future_pr + dyn_mk) < curr_pr:
                labels.append(1.0)  # sell
            elif curr_lvl < low_q and (future_pr - dyn_mk) > curr_pr:
                labels.append(0.0)  # buy
            else:
                labels.append(2.0)  # no confiable
    return labels

def get_labels_mean_reversion_vol(
    dataset, label_markup, label_min_val=1, label_max_val=15, label_rolling=0.5,
    label_quantiles=[.45, .55], label_filter='spline', label_shift=1,
    label_vol_window=20, label_atr_period=14, direction=2
) -> pd.DataFrame:
    """
    Generates trading labels based on mean reversion principles, incorporating
    volatility-based adjustments to identify buy/sell opportunities.
    Now supports unidirectional trading (buy-only, sell-only) following the meta/main methodology.

    Args:
        dataset (pd.DataFrame): DataFrame containing financial data with a 'close' column.
        label_markup (float): The percentage label_markup used to determine buy/sell signals.
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
        label_vol_window (int, optional): Window size for calculating volatility. Defaults to 20.
        direction (int, optional): 2=both, 0=buy only, 1=sell only. Defaults to 2.

    Returns:
        pd.DataFrame: The original DataFrame with a new 'labels_main' column and filtered rows:
                       - 'labels_main' column: 
                            - 0: Buy (or éxito direccional en buy-only)
                            - 1: Sell (o éxito direccional en sell-only)
                            - 2: Patrón no confiable
                       - Rows with missing values (NaN) are removed.
                       - Temporary 'lvl', 'volatility', 'volatility_group' columns are removed.
    """

    # Calculate Volatility
    dataset['volatility'] = bn.move_std(dataset['close'].values, window=label_vol_window, min_count=1)
    if len(np.unique(dataset['volatility'])) < 2:
        # No se puede hacer qcut, todos los valores son iguales
        return pd.DataFrame()
    # Divide into 20 groups by volatility 
    dataset['volatility_group'] = pd.qcut(dataset['volatility'], q=20, labels=False, duplicates='drop')
    # Calculate price deviation ('lvl') based on the chosen label_filter
    if label_filter == 'mean':
        dataset['lvl'] = (dataset['close'] - dataset['close'].rolling(label_rolling).mean())
    elif label_filter == 'spline':
        x = np.array(range(dataset.shape[0]))
        y = dataset['close'].values
        spl = UnivariateSpline(x, y, k=3, s=label_rolling)
        yHat = spl(np.linspace(min(x), max(x), num=x.shape[0]))
        yHat_shifted = np.roll(yHat, shift=label_shift) # Apply the shift 
        dataset['lvl'] = dataset['close'] - yHat_shifted
    elif label_filter == 'savgol':
        smoothed_prices, filtering_successful = safe_savgol_filter(dataset['close'].values, label_rolling=label_rolling, label_polyorder=5)
        if not filtering_successful:
            print(f"🔍 DEBUG: Savitzky-Golay filtering failed in get_labels_mean_reversion_vol, returning empty dataset")
            return pd.DataFrame()
        dataset['lvl'] = dataset['close'] - smoothed_prices
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
        quantiles_values = group_data.quantile(label_quantiles).to_list()
        quantile_groups[group] = quantiles_values
        quantile_groups_low.append(quantiles_values[0])
        quantile_groups_high.append(quantiles_values[1])
    # Prepare data for label calculation
    lvl_data = dataset['lvl'].values
    volatility_group = dataset['volatility_group'].values
    # Convert quantile groups to numpy arrays
    quantile_groups_low = np.array(quantile_groups_low)
    high = dataset["high"].values
    low = dataset["low"].values
    close = dataset["close"].values
    atr = calculate_atr_simple(high, low, close, period=label_atr_period)
    quantile_groups_high = np.array(quantile_groups_high)
    # Calculate buy/sell labels with direction support
    labels = calculate_labels_mean_reversion_vol(
        close, atr, lvl_data, volatility_group, quantile_groups_low, quantile_groups_high,
        label_markup, label_min_val, label_max_val, direction
    )
    # Align labels and fill with 2.0 using pandas to avoid padding
    labels = pd.Series(labels, index=dataset.index[:-label_max_val]).reindex_like(dataset).fillna(2.0)
    # Add labels to dataset
    dataset['labels_main'] = labels
    dataset['labels_main'] = dataset['labels_main'].fillna(2.0)    
    # Remove temporary columns
    return dataset.drop(columns=['lvl', 'volatility', 'volatility_group'])

######### CLUSTERING BASED LABELING #########

def clustering_kmeans(
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

def clustering_hdbscan(
    dataset: pd.DataFrame,
    n_clusters: int = 5,
) -> pd.DataFrame:
    """
    Clustering simple usando HDBSCAN.
    Si el dataset está vacío o no hay meta features, asigna -1 a labels_meta.
    """
    try:
        import hdbscan
    except ImportError:
        print(f"🔍 DEBUG: HDBSCAN no está instalado, asignando -1 a labels_meta")
        dataset["labels_meta"] = -1
        return dataset

    if dataset.empty:
        dataset["labels_meta"] = -1
        return dataset
    meta_X = dataset.filter(regex="meta_feature")
    if meta_X.shape[1] == 0:
        dataset["labels_meta"] = -1
        return dataset
    meta_X_np = meta_X.to_numpy(np.float32)
    try:
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=n_clusters,
            core_dist_n_jobs=-1
        ).fit(meta_X_np)
        dataset["labels_meta"] = clusterer.labels_.astype(int)
    except Exception as e:
        print(f"🔍 DEBUG: Error en clustering_simple con HDBSCAN: {e}")
        dataset["labels_meta"] = -1
    return dataset

def clustering_markov(dataset, n_regimes: int, model_type="GMMHMM", n_iter = 10, n_mix = 3) -> pd.DataFrame:
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


def clustering_lgmm(dataset: pd.DataFrame, n_components: int,
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

@njit(cache=True)
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

@njit(cache=True, parallel=True)
def _euclidean_matrix_numba(window_sizes):
    n = window_sizes.shape[0]
    dist_mat = np.zeros((n, n), dtype=np.float64)
    for i in prange(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(window_sizes[i] - window_sizes[j])
            dist_mat[i, j] = dist_mat[j, i] = d
    return dist_mat

@njit(cache=True)
def _wasserstein1d_numba(x, y):
    x = np.sort(x)
    y = np.sort(y)
    return np.mean(np.abs(x - y))

@njit(cache=True, parallel=True)
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

@njit(cache=True, parallel=True)
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

@njit(cache=True)
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

@njit(cache=True, parallel=True)
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

# ✅ FUNCIÓN DE TESTING PARA VALIDACIÓN DE ALINEACIÓN
def test_labeling_alignment():
    """
    Función de testing para validar que todas las funciones de etiquetado
    mantengan correcta alineación entre el dataset de entrada y las etiquetas generadas.
    
    Esta función debe ejecutarse después de cualquier modificación a las funciones
    de etiquetado para asegurar integridad de datos.
    
    Returns:
        bool: True si todas las validaciones pasan, False en caso contrario
    """
    print("🧪 TESTING: Validando alineación de funciones de etiquetado...")
    
    # Crear dataset de prueba
    np.random.seed(42)  # Para reproducibilidad
    n_samples = 1000
    
    # Datos sintéticos que simulan precios financieros
    prices = np.cumsum(np.random.randn(n_samples) * 0.01) + 100
    volume = np.random.randint(1000, 10000, n_samples)
    
    test_dataset = pd.DataFrame({
        'time': pd.date_range('2020-01-01', periods=n_samples, freq='1H'),
        'open': prices + np.random.randn(n_samples) * 0.005,
        'high': prices + np.abs(np.random.randn(n_samples) * 0.01),
        'low': prices - np.abs(np.random.randn(n_samples) * 0.01),
        'close': prices,
        'volume': volume
    })
    test_dataset.set_index('time', inplace=True)
    
    # Lista de funciones a testear
    functions_to_test = [
        ('get_labels_random', lambda df: get_labels_random(df.copy(), direction=2)),
        ('get_labels_filter', lambda df: get_labels_filter(df.copy(), direction=2)),
        ('get_labels_clusters', lambda df: get_labels_clusters(df.copy(), label_markup=0.01, direction=2)),
        # Añadir más funciones según sea necesario
    ]
    
    all_tests_passed = True
    
    for func_name, func in functions_to_test:
        try:
            print(f"  • Testando {func_name}...")
            
            # Ejecutar función
            result_df = func(test_dataset)
            
            # Validaciones críticas
            validation_checks = [
                (len(result_df) == len(test_dataset), f"Longitud del dataset cambió: {len(result_df)} vs {len(test_dataset)}"),
                ('labels_main' in result_df.columns, "Columna 'labels_main' no encontrada"),
                (result_df.index.equals(test_dataset.index), "Índices no coinciden"),
                (not result_df['labels_main'].isna().all(), "Todas las etiquetas son NaN"),
                (result_df['close'].equals(test_dataset['close']), "Columna 'close' fue modificada incorrectamente")
            ]
            
            for check, error_msg in validation_checks:
                if not check:
                    print(f"    ❌ FALLO en {func_name}: {error_msg}")
                    all_tests_passed = False
                    break
            else:
                print(f"    ✅ {func_name} pasó todas las validaciones")
                
        except Exception as e:
            print(f"    ❌ ERROR en {func_name}: {str(e)}")
            all_tests_passed = False
    
    if all_tests_passed:
        print("🎉 TESTING COMPLETO: Todas las funciones mantienen correcta alineación")
    else:
        print("⚠️  TESTING FALLIDO: Se encontraron problemas de alineación")
    
    return all_tests_passed

# ✅ FUNCIÓN DE VALIDACIÓN DE INTEGRIDAD DE DATOS
def validate_dataset_integrity(dataset_original, dataset_labeled, function_name):
    """
    Valida que un dataset etiquetado mantenga la integridad de los datos originales.
    
    Args:
        dataset_original: Dataset antes del etiquetado
        dataset_labeled: Dataset después del etiquetado
        function_name: Nombre de la función de etiquetado para logging
        
    Returns:
        bool: True si la integridad se mantiene, False en caso contrario
    """
    checks = [
        (len(dataset_labeled) == len(dataset_original), 
         f"Longitud cambió: {len(dataset_labeled)} vs {len(dataset_original)}"),
        
        (dataset_labeled.index.equals(dataset_original.index), 
         "Índices no coinciden"),
        
        ('labels_main' in dataset_labeled.columns, 
         "Columna 'labels_main' no fue creada"),
        
        (np.allclose(dataset_labeled['close'].values, dataset_original['close'].values, rtol=1e-10), 
         "Columna 'close' fue modificada"),
        
        (not dataset_labeled['labels_main'].isna().all(), 
         "Todas las etiquetas son NaN"),
        
        (dataset_labeled['labels_main'].isin([0.0, 1.0, 2.0]).all(), 
         "Etiquetas contienen valores no válidos")
    ]
    
    for check, error_msg in checks:
        if not check:
            print(f"❌ VALIDACIÓN FALLIDA en {function_name}: {error_msg}")
            return False
    
    return True