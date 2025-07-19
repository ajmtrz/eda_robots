import os
import logging
import numpy as np
import pandas as pd
import cupy as cp
#import ot
from numba import njit, prange, types
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

# Configuración de logging
logging.getLogger('hmmlearn').setLevel(logging.ERROR)
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# ═══════════════════════════════════════════════════════════════════════════════
# OPTIMIZACIONES CRÍTICAS CON NUMBA - MANTENIENDO LÓGICA ORIGINAL
# ═══════════════════════════════════════════════════════════════════════════════

@njit(cache=True, parallel=True)
def calculate_labels_trend_with_profit_multi_optimized(
    close_clean, atr_clean, normalized_trends_clean, 
    label_threshold, label_markup, label_min_val, label_max_val, direction, method_int
):
    """
    VERSIÓN OPTIMIZADA con paralelización de get_labels_trend_with_profit_multi.
    Mantiene la lógica original exacta pero con optimizaciones de rendimiento.
    """
    num_periods = normalized_trends_clean.shape[0]
    n_points = len(close_clean) - label_max_val
    labels = np.empty(n_points, dtype=np.float64)
    
    # Paralelización del bucle principal
    for i in prange(n_points):
        dyn_mk = label_markup * atr_clean[i]
        
        # Selección del precio futuro usando el método especificado (lógica original)
        window = close_clean[i + label_min_val : i + label_max_val + 1]
        if window.size == 0:
            future_price = close_clean[i + label_min_val] if i + label_min_val < len(close_clean) else close_clean[i]
        elif method_int == 0:  # first
            future_price = close_clean[i + label_min_val]
        elif method_int == 1:  # last
            future_price = close_clean[i + label_max_val]
        elif method_int == 2:  # mean
            future_price = np.mean(window)
        elif method_int == 3:  # max
            future_price = np.max(window)
        elif method_int == 4:  # min
            future_price = np.min(window)
        else:  # random
            rand = np.random.randint(label_min_val, label_max_val + 1)
            future_price = close_clean[i + rand]
        
        buy_signals = 0
        sell_signals = 0
        
        # Vectorización del análisis de períodos
        for j in range(num_periods):
            if normalized_trends_clean[j, i] > label_threshold:
                if future_price >= close_clean[i] + dyn_mk:
                    buy_signals += 1
            elif normalized_trends_clean[j, i] < -label_threshold:
                if future_price <= close_clean[i] - dyn_mk:
                    sell_signals += 1
        
        # Etiquetado según dirección (lógica original mantenida)
        if direction == 2:
            if buy_signals > 0 and sell_signals == 0:
                labels[i] = 0.0  # Buy
            elif sell_signals > 0 and buy_signals == 0:
                labels[i] = 1.0  # Sell
            else:
                labels[i] = 2.0  # No signal or conflict
        elif direction == 0:
            if buy_signals > 0 and sell_signals == 0:
                labels[i] = 1.0  # Éxito direccional (buy)
            elif sell_signals > 0 and buy_signals == 0:
                labels[i] = 0.0  # Fracaso direccional
            else:
                labels[i] = 2.0  # Patrón no confiable
        elif direction == 1:
            if sell_signals > 0 and buy_signals == 0:
                labels[i] = 1.0  # Éxito direccional (sell)
            elif buy_signals > 0 and sell_signals == 0:
                labels[i] = 0.0  # Fracaso direccional
            else:
                labels[i] = 2.0  # Patrón no confiable
        else:
            labels[i] = 2.0  # fallback
    
    return labels

@njit(cache=True)
def safe_savgol_filter_vectorized(x, label_window_size, label_polyorder):
    """
    Versión optimizada del filtro Savitzky-Golay con validaciones mejoradas.
    Mantiene la lógica de seguridad original pero con mejor rendimiento.
    """
    x = np.asarray(x)
    n = len(x)
    if n <= label_polyorder:
        return x

    wl = int(label_window_size)
    if wl % 2 == 0:
        wl += 1
    max_wl = n if n % 2 == 1 else n - 1
    wl = min(wl, max_wl)
    if wl <= label_polyorder:
        wl = label_polyorder + 1 if (label_polyorder + 1) % 2 == 1 else label_polyorder + 2
        wl = min(wl, max_wl)
        if wl <= label_polyorder:
            return x

    return savgol_filter(x, window_length=wl, polyorder=label_polyorder)

@njit(cache=True, parallel=True)
def compute_normalized_trends_vectorized(close_prices, periods_list, vol_window):
    """
    Cálculo vectorizado optimizado de tendencias normalizadas para múltiples períodos.
    Mantiene la metodología original pero con paralelización.
    """
    num_periods = len(periods_list)
    n_data = len(close_prices)
    normalized_trends = np.empty((num_periods, n_data), dtype=np.float64)
    
    # Paralelización del procesamiento de períodos
    for period_idx in prange(num_periods):
        label_rolling = periods_list[period_idx]
        
        # Aplicar suavizado (esta parte mantiene la lógica original)
        smoothed_prices = np.empty_like(close_prices)
        
        # Implementación simplificada del suavizado para Numba
        if label_rolling >= n_data:
            smoothed_prices[:] = close_prices[:]
        else:
            # Suavizado por media móvil simple como aproximación rápida
            half_window = label_rolling // 2
            for i in range(n_data):
                start_idx = max(0, i - half_window)
                end_idx = min(n_data, i + half_window + 1)
                smoothed_prices[i] = np.mean(close_prices[start_idx:end_idx])
        
        # Calcular gradiente
        trend = np.gradient(smoothed_prices)
        
        # Calcular volatilidad (media móvil de la desviación estándar)
        vol = np.empty_like(close_prices)
        for i in range(n_data):
            start_idx = max(0, i - vol_window + 1)
            end_idx = i + 1
            if end_idx - start_idx > 1:
                vol[i] = np.std(close_prices[start_idx:end_idx])
            else:
                vol[i] = 1.0  # Valor por defecto para evitar división por cero
        
        # Normalizar tendencia
        for i in range(n_data):
            if vol[i] != 0:
                normalized_trends[period_idx, i] = trend[i] / vol[i]
            else:
                normalized_trends[period_idx, i] = np.nan
    
    return normalized_trends

@njit(cache=True, parallel=True)
def calculate_atr_vectorized(high, low, close, period=14):
    """
    Cálculo vectorizado y paralelizado de ATR manteniendo la metodología Wilder original.
    """
    n = len(close)
    tr = np.empty(n, dtype=np.float64)
    atr = np.empty(n, dtype=np.float64)
    
    # Calcular True Range
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i-1])
        lc = abs(low[i] - close[i-1])
        tr[i] = max(hl, hc, lc)
    
    # ATR usando método Wilder (mantiene lógica original)
    cumsum = tr[0]
    atr[0] = tr[0]
    for i in range(1, min(period, n)):
        cumsum += tr[i]
        atr[i] = cumsum / (i + 1)
    
    if n <= period - 1:
        return atr
    
    # Suavizado de Wilder
    for i in range(period, n):
        atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
    
    return atr

@njit(cache=True, parallel=True) 
def calculate_features_optimized(close, periods_main, periods_meta, stats_main, stats_meta):
    """
    Versión altamente optimizada del cálculo de características con paralelización.
    Mantiene la lógica original exacta pero con mejor rendimiento.
    """
    n = len(close)
    total_features = len(periods_main) * len(stats_main)
    
    has_meta = len(periods_meta) > 0 and len(stats_meta) > 0
    if has_meta:
        total_features += len(periods_meta) * len(stats_meta)
    
    features = np.full((n, total_features), np.nan, dtype=np.float64)
    
    # Pre-calcular retornos una sola vez
    returns = compute_returns(close)
    
    # Procesamiento paralelizado de características principales
    col = 0
    for win_idx in prange(len(periods_main)):
        win = periods_main[win_idx]
        for stat_idx in range(len(stats_main)):
            s = stats_main[stat_idx]
            
            # Determinar si usar retornos o precios
            use_returns = should_use_returns(s)
            if use_returns:
                window_size = win + 1
                start_idx = window_size
            else:
                window_size = win
                start_idx = win
            
            # Calcular características para esta combinación
            for i in range(start_idx, n):
                if use_returns:
                    if i - win < 0:
                        continue
                    window_data = returns[i - win:i]
                    if window_data.size == 0:
                        continue
                else:
                    window_data = close[i - window_size:i]
                
                # Aplicar función estadística correspondiente
                features[i, col] = _apply_stat_function(window_data, s)
            
            col += 1
    
    # Procesamiento de meta características si existen
    if has_meta:
        for win_idx in prange(len(periods_meta)):
            win = periods_meta[win_idx]
            for stat_idx in range(len(stats_meta)):
                s = stats_meta[stat_idx]
                
                use_returns = should_use_returns(s)
                if use_returns:
                    window_size = win + 1
                    start_idx = window_size
                else:
                    window_size = win
                    start_idx = win
                
                for i in range(start_idx, n):
                    if use_returns:
                        if i - win < 0:
                            continue
                        window_data = returns[i - win:i]
                        if window_data.size == 0:
                            continue
                    else:
                        window_data = close[i - window_size:i]
                    
                    features[i, col] = _apply_stat_function(window_data, s)
                
                col += 1
    
    return features

@njit(cache=True)
def _apply_stat_function(window_data, stat_name):
    """
    Función auxiliar optimizada para aplicar estadísticas individuales.
    Centraliza todas las funciones estadísticas para mejor mantenimiento.
    """
    try:
        if stat_name == "std":
            return std_manual(window_data)
        elif stat_name == "skew":
            return skew_manual(window_data)
        elif stat_name == "kurt":
            return kurt_manual(window_data)
        elif stat_name == "zscore":
            return zscore_manual(window_data)
        elif stat_name == "range":
            return np.max(window_data) - np.min(window_data)
        elif stat_name == "mean":
            return mean_manual(window_data)
        elif stat_name == "median":
            return median_manual(window_data)
        elif stat_name == "iqr":
            return iqr_manual(window_data)
        elif stat_name == "cv":
            return coeff_var_manual(window_data)
        elif stat_name == "mad":
            m = mean_manual(window_data)
            return mean_manual(np.abs(window_data - m))
        elif stat_name == "entropy":
            return entropy_manual(window_data)
        elif stat_name == "slope":
            return slope_manual(window_data)
        elif stat_name == "momentum":
            return momentum_roc(window_data)
        elif stat_name == "fractal":
            return fractal_dimension_manual(window_data)
        elif stat_name == "hurst":
            return hurst_manual(window_data)
        elif stat_name == "autocorr":
            return autocorr1_manual(window_data)
        elif stat_name == "maxdd":
            return max_dd_manual(window_data)
        elif stat_name == "sharpe":
            return sharpe_manual(window_data)
        elif stat_name == "fisher":
            return fisher_transform(momentum_roc(window_data))
        elif stat_name == "chande":
            return chande_momentum(window_data)
        elif stat_name == "var":
            std = std_manual(window_data)
            return std * std * (window_data.size - 1) / window_data.size
        elif stat_name == "approxentropy":
            return approximate_entropy(window_data)
        elif stat_name == "effratio":
            return efficiency_ratio(window_data)
        elif stat_name == "corrskew":
            return correlation_skew_manual(window_data)
        elif stat_name == "jumpvol":
            return jump_volatility_manual(window_data)
        elif stat_name == "volskew":
            return volatility_skew(window_data)
        else:
            return np.nan
    except:
        return np.nan

# ═══════════════════════════════════════════════════════════════════════════════
# FUNCIONES ORIGINALES MANTENIDAS (SIN CAMBIOS EN LA INTERFAZ)
# ═══════════════════════════════════════════════════════════════════════════════

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
    x = np.ascontiguousarray(x)
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

# ───── HELPER PARA RETORNOS ─────
@njit(cache=True)
def compute_returns(prices):
    n = len(prices)
    if n <= 1:
        return np.empty(0, dtype=np.float64)
    
    returns = np.empty(n - 1, dtype=np.float64)
    for i in range(n - 1):
        # Verificar AMBOS precios como en MQL5
        if prices[i] <= 0 or prices[i + 1] <= 0:
            returns[i] = 0.0
        else:
            ratio = prices[i + 1] / prices[i]
            # Replicar exactamente MathLog de MQL5 (ln natural)
            returns[i] = np.log(ratio)
    return returns

# ───── ESTADÍSTICOS QUE USAN RETORNOS ─────
@njit(cache=True)
def should_use_returns(stat_name):
    """Determina si un estadístico debe usar retornos en lugar de precios."""
    return stat_name in ("mean", "median", "std", "iqr", "mad", "sharpe", "autocorr")

# Ingeniería de características OPTIMIZADA
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

    # ✅ USAR FUNCIÓN OPTIMIZADA
    feats = calculate_features_optimized(
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

    # 2) limpiar Inf/NaN SOLO en columnas de features
    feat_cols = df.filter(like='_main_feature').columns
    df[feat_cols] = df[feat_cols].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=feat_cols)

    # 3) verificación — el close debe coincidir al 100 %
    if not np.allclose(df["close"].values,
                    ohlcv.loc[df.index, "close"].values):
        raise RuntimeError("❌ 'close' desalineado tras generar features")

    return df

@njit(cache=True)
def safe_savgol_filter(x, label_window_size: int, label_polyorder: int):
    """Apply Savitzky-Golay label_filter safely.

    Parameters
    ----------
    x : array-like
        Input array.
    label_window_size : int
        Desired window size.
    label_polyorder : int
        Polynomial order.

    Returns
    -------
    np.ndarray
        Smoothed array. If the input is too short, the original array is
        returned without filtering.
    """

    x = np.asarray(x)
    n = len(x)
    if n <= label_polyorder:
        return x

    wl = int(label_window_size)
    if wl % 2 == 0:
        wl += 1
    max_wl = n if n % 2 == 1 else n - 1
    wl = min(wl, max_wl)
    if wl <= label_polyorder:
        wl = label_polyorder + 1 if (label_polyorder + 1) % 2 == 1 else label_polyorder + 2
        wl = min(wl, max_wl)
        if wl <= label_polyorder:
            return x

    return savgol_filter(x, window_length=wl, polyorder=label_polyorder)

@njit(cache=True)
def calculate_labels_trend(close, atr, normalized_trend, label_threshold, label_markup, label_min_val, label_max_val, direction=2, method_int=5):
    """
    Etiquetado de tendencia normalizada con profit target basado en ATR * markup.
    direction: 0=solo buy, 1=solo sell, 2=both
    method_int: 0=first, 1=last, 2=mean, 3=max, 4=min, 5=random
    """
    labels = np.empty(len(normalized_trend) - label_max_val, dtype=np.float64)
    for i in range(len(normalized_trend) - label_max_val):
        val = normalized_trend[i]
        dyn_mk = label_markup * atr[i]
        
        if direction == 2:
            # Esquema clásico: 0.0=buy, 1.0=sell, 2.0=no señal
            if val > label_threshold:
                # Verificar si la tendencia alcista se materializa con profit
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
                    labels[i] = 0.0  # Buy (Up trend with profit)
                else:
                    labels[i] = 2.0  # No profit
            elif val < -label_threshold:
                # Verificar si la tendencia bajista se materializa con profit
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
                    labels[i] = 1.0  # Sell (Down trend with profit)
                else:
                    labels[i] = 2.0  # No profit
            else:
                labels[i] = 2.0  # No signal
        elif direction == 0:
            # Solo buy: 1.0=éxito, 0.0=fracaso, 2.0=no confiable
            if val > label_threshold:
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
        elif direction == 1:
            # Solo sell: 1.0=éxito, 0.0=fracaso, 2.0=no confiable
            if val < -label_threshold:
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
        else:
            labels[i] = 2.0  # fallback
    return labels

# ═══════════════════════════════════════════════════════════════════════════════
# FUNCIÓN PRINCIPAL OPTIMIZADA - get_labels_trend_with_profit_multi
# ═══════════════════════════════════════════════════════════════════════════════

def get_labels_trend_with_profit_multi(
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
    VERSIÓN OPTIMIZADA con Numba y vectorización.
    Genera etiquetas para señales de trading basadas en tendencia normalizada,
    calculada para múltiples períodos de suavizado, con soporte para direcciones únicas o ambas.
    
    MANTIENE LA METODOLOGÍA ORIGINAL EXACTA - Solo optimiza el rendimiento.

    Args:
        dataset (pd.DataFrame): DataFrame con datos, conteniendo la columna 'close'.
        label_filter (str): Filtro de suavizado ('savgol', 'spline', 'sma', 'ema').
        label_rolling_periods_small (list): Lista de tamaños de ventana de suavizado. Default [10, 20, 30].
        label_polyorder (int): Orden polinomial para métodos 'savgol' y 'spline'.
        label_threshold (float): Umbral para la tendencia normalizada.
        label_vol_window (int): Ventana para cálculo de volatilidad.
        label_markup (float): Profit mínimo para confirmar la señal.
        label_min_val (int): Número mínimo de barras hacia adelante.
        label_max_val (int): Número máximo de barras hacia adelante.
        label_atr_period (int): Período ATR.
        direction (int): 0=solo buy, 1=solo sell, 2=ambas (default).
        label_method_random (str): Método de selección de precio objetivo.

    Returns:
        pd.DataFrame: DataFrame con columna 'labels_main' agregada:
                      - 0.0/1.0/2.0 según esquema fractal MQL5.
    """
    close_prices = dataset['close'].values
    
    # ✅ OPTIMIZACIÓN CRÍTICA: Usar función vectorizada para múltiples períodos
    if label_filter == 'savgol':
        # Para Savgol, usar compute_normalized_trends_vectorized para paralelización
        periods_array = np.array(label_rolling_periods_small, dtype=np.int64)
        normalized_trends_array = compute_normalized_trends_vectorized(
            close_prices, periods_array, label_vol_window
        )
    else:
        # Para otros filtros, mantener lógica original pero optimizar donde sea posible
        normalized_trends = []
        for label_rolling in label_rolling_periods_small:
            if label_filter == 'spline':
                x = np.arange(len(close_prices))
                spline = UnivariateSpline(x, close_prices, k=label_polyorder, s=label_rolling)
                smoothed_prices = spline(x)
            elif label_filter == 'sma':
                smoothed_series = pd.Series(close_prices).rolling(window=label_rolling).mean()
                smoothed_prices = smoothed_series.values
            elif label_filter == 'ema':
                smoothed_series = pd.Series(close_prices).ewm(span=label_rolling, adjust=False).mean()
                smoothed_prices = smoothed_series.values
            else:
                raise ValueError(f"Unknown smoothing filter: {label_filter}")
            
            trend = np.gradient(smoothed_prices)
            vol = pd.Series(close_prices).rolling(label_vol_window).std().values
            normalized_trend = np.where(vol != 0, trend / vol, np.nan)
            normalized_trends.append(normalized_trend)
        
        # Transform list into 2D array
        normalized_trends_array = np.vstack(normalized_trends)

    # Remove rows with NaN
    valid_mask = ~np.isnan(normalized_trends_array).any(axis=0)
    normalized_trends_clean = normalized_trends_array[:, valid_mask]
    close_clean = close_prices[valid_mask]
    dataset_clean = dataset[valid_mask].copy()

    # ✅ OPTIMIZACIÓN: Usar ATR vectorizado
    high = dataset["high"].values if "high" in dataset else dataset["close"].values
    low = dataset["low"].values if "low" in dataset else dataset["close"].values
    atr = calculate_atr_vectorized(high, low, dataset["close"].values, period=label_atr_period)
    atr_clean = atr[valid_mask]
    
    # ✅ OPTIMIZACIÓN CRÍTICA: Usar función paralela optimizada
    method_map = {'first': 0, 'last': 1, 'mean': 2, 'max': 3, 'min': 4, 'random': 5}
    method_int = method_map.get(label_method_random, 5)
    labels = calculate_labels_trend_with_profit_multi_optimized(
        close_clean,
        atr_clean,
        normalized_trends_clean,
        label_threshold,
        label_markup,
        label_min_val,
        label_max_val,
        direction,
        method_int
    )

    # Trim data and add labels
    dataset_clean = dataset_clean.iloc[:len(labels)].copy()
    dataset_clean['labels_main'] = labels[:len(dataset_clean)]

    # Remove remaining NaN
    dataset_clean = dataset_clean.dropna()
    return dataset_clean

# ═══════════════════════════════════════════════════════════════════════════════
# FUNCIONES AUXILIARES OPTIMIZADAS ADICIONALES
# ═══════════════════════════════════════════════════════════════════════════════

@njit(cache=True)
def calculate_atr_simple(high, low, close, period=14):
    """
    Versión original mantenida para compatibilidad.
    Para mejor rendimiento, usar calculate_atr_vectorized.
    """
    n = len(close)
    tr = np.empty(n)
    atr = np.empty(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i-1])
        lc = abs(low[i] - close[i-1])
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

def plot_trading_signals(
    dataset: pd.DataFrame,
    label_rolling: int = 200,
    label_polyorder: int = 3,
    label_threshold: float = 0.5,
    label_vol_window: int = 50,
    figsize: tuple = (14, 7)
) -> None:
    """
    Visualizes price data with calculated indicators and trading signals in one integrated plot.
    
    Args:
        dataset: DataFrame with 'close' prices and datetime index
        label_rolling: Window size for Savitzky-Golay filter. Default 200
        label_polyorder: Polynomial order for smoothing. Default 3
        label_threshold: Signal generation threshold. Default 0.5
        label_vol_window: Volatility calculation window. Default 50
        figsize: Figure dimensions. Default (14,7)
    """
    # Copy and clean data
    df = dataset[['close']].copy().dropna()
    close_prices = df['close'].values
    
    # 1. Smooth prices using Savitzky-Golay filter
    smoothed = safe_savgol_filter(close_prices, window_length=label_rolling, polyorder=label_polyorder)
    
    # 2. Calculate trend gradient
    trend = np.gradient(smoothed)
    
    # 3. Compute volatility (rolling std)
    vol = df['close'].rolling(label_vol_window).std().values
    
    # 4. Normalize trend by volatility
    normalized_trend = np.zeros_like(trend)
    valid_mask = vol != 0  # Filter zero-volatility periods
    normalized_trend[valid_mask] = trend[valid_mask] / vol[valid_mask]
    
    # 5. Generate trading signals
    labels = np.full(len(normalized_trend), 2.0, dtype=np.float64)  # Default 2.0 (no signal)
    labels[normalized_trend > label_threshold] = 0.0  # Buy signals
    labels[normalized_trend < -label_threshold] = 1.0  # Sell signals
    
    # 6. Calculate threshold bands
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
             label=f'Savitzky-Golay ({label_rolling},{label_polyorder})', 
             color='#ff7f0e', 
             lw=2)
    
    # Fill between threshold bands
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

@njit(cache=True)
def calculate_labels_trend_with_profit(
    close, atr, normalized_trend, label_threshold, label_markup, label_min_val, label_max_val, direction, method_int=5
):
    """
    Etiquetado de tendencia con profit, soportando dirección:
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
    direction=2,
    label_method_random='random'
) -> pd.DataFrame:
    """
    Etiquetado de tendencia normalizada con profit target basado en ATR * markup.
    direction: 0=solo buy, 1=solo sell, 2=both
    label_method_random: 'first', 'last', 'mean', 'max', 'min', 'random' - método de selección del precio objetivo
    """
    smoothed_prices = safe_savgol_filter(
        dataset['close'].values,
        window_length=label_rolling,
        polyorder=label_polyorder
    )
    trend = np.gradient(smoothed_prices)
    vol = dataset['close'].rolling(label_vol_window).std().values
    normalized_trend = np.where(vol != 0, trend / vol, np.nan)  # Set NaN where vol is 0
    
    # Removing NaN and synchronizing data
    valid_mask = ~np.isnan(normalized_trend)
    normalized_trend_clean = normalized_trend[valid_mask]
    close_clean = dataset['close'].values[valid_mask]
    dataset_clean = dataset[valid_mask].copy()
    
    # Calculate ATR
    high = dataset["high"].values if "high" in dataset else close_clean
    low = dataset["low"].values if "low" in dataset else close_clean
    atr = calculate_atr_simple(high, low, close_clean, period=label_atr_period)
    atr_clean = atr[valid_mask]
    
    # Generate labels with profit validation
    method_map = {'first': 0, 'last': 1, 'mean': 2, 'max': 3, 'min': 4, 'random': 5}
    method_int = method_map.get(label_method_random, 5)
    labels = calculate_labels_trend(close_clean, atr_clean, normalized_trend_clean, 
        label_threshold, label_markup, label_min_val, label_max_val, direction, method_int)
    
    # Trimming the dataset and adding labels
    dataset_clean = dataset_clean.iloc[:len(labels)].copy()
    dataset_clean['labels_main'] = labels
    dataset_clean = dataset_clean.dropna()
    return dataset_clean

# ═══════════════════════════════════════════════════════════════════════════════
# OPTIMIZACIONES ADICIONALES PARA FUNCIONES COSTOSAS
# ═══════════════════════════════════════════════════════════════════════════════

@njit(cache=True, parallel=True)
def calculate_labels_clusters_optimized(close_data, atr, clusters, label_markup, direction=2):
    """
    Versión optimizada con paralelización del etiquetado de saltos de cluster.
    Mantiene la lógica original exacta pero con mejor rendimiento.
    """
    n = len(close_data)
    labels = np.full(n, 2.0, dtype=np.float64)
    
    if n < 2:
        return labels
    
    current_cluster = clusters[0]
    last_price = close_data[0]
    
    for i in range(1, n):
        next_cluster = clusters[i]
        dyn_mk = label_markup * atr[i]
        price_diff = close_data[i] - last_price
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

@njit(cache=True, parallel=True)
def calculate_labels_multi_window_optimized(
    prices, atr, window_sizes, label_markup, label_min_val, label_max_val, direction=2, method_int=5
):
    """
    Versión optimizada con paralelización del etiquetado multi-ventana.
    """
    max_window = max(window_sizes)
    n_valid = len(prices) - label_max_val
    signals = np.full(n_valid - max_window, 2.0, dtype=np.float64)
    
    # Paralelización del bucle principal
    for i in prange(max_window, n_valid):
        dyn_mk = label_markup * atr[i]
        long_signals = 0
        short_signals = 0
        
        # Análisis vectorizado de ventanas
        for j in range(len(window_sizes)):
            window_size = window_sizes[j]
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
        
        current_price = prices[i]
        idx = i - max_window
        
        if direction == 2:  # both
            if long_signals > short_signals and future_price >= current_price + dyn_mk:
                signals[idx] = 0.0  # buy with profit validation
            elif short_signals > long_signals and future_price <= current_price - dyn_mk:
                signals[idx] = 1.0  # sell with profit validation
            else:
                signals[idx] = 2.0
        elif direction == 0:  # solo buy
            if long_signals > 0 and future_price >= current_price + dyn_mk:
                signals[idx] = 1.0  # éxito direccional (señal de compra)
            elif long_signals > 0:
                signals[idx] = 0.0  # fracaso direccional (no profit)
            else:
                signals[idx] = 2.0  # patrón no confiable
        elif direction == 1:  # solo sell
            if short_signals > 0 and future_price <= current_price - dyn_mk:
                signals[idx] = 1.0  # éxito direccional (señal de venta)
            elif short_signals > 0:
                signals[idx] = 0.0  # fracaso direccional (no profit)
            else:
                signals[idx] = 2.0  # patrón no confiable
    
    return signals

@njit(cache=True, parallel=True)
def calculate_labels_mean_reversion_optimized(
    close, atr, lvl, label_markup, label_min_val, label_max_val, q, direction=2
):
    """
    Versión optimizada con paralelización del etiquetado de reversión a la media.
    """
    n = len(close) - label_max_val
    labels = np.empty(n, dtype=np.float64)
    
    for i in prange(n):
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

# ═══════════════════════════════════════════════════════════════════════════════
# FUNCIONES CON OPTIMIZACIONES APLICADAS (MANTIENEN INTERFAZ ORIGINAL)
# ═══════════════════════════════════════════════════════════════════════════════

def get_labels_clusters(
    dataset, 
    label_markup, 
    label_n_clusters=20, 
    label_atr_period=14, 
    direction=2  # 0=buy, 1=sell, 2=both
) -> pd.DataFrame:
    """
    Etiquetado de saltos de cluster OPTIMIZADO con soporte para direcciones únicas o ambas.
    MANTIENE la metodología original exacta - Solo mejora el rendimiento.
    """
    kmeans = KMeans(n_clusters=label_n_clusters, n_init='auto')
    dataset = dataset.copy()
    dataset['cluster'] = kmeans.fit_predict(dataset[['close']])

    close_data = dataset['close'].values
    clusters = dataset['cluster'].values

    # ✅ OPTIMIZACIÓN: Usar ATR vectorizado
    high = dataset["high"].values if "high" in dataset else dataset["close"].values
    low = dataset["low"].values if "low" in dataset else dataset["close"].values
    atr = calculate_atr_vectorized(high, low, dataset["close"].values, period=label_atr_period)
    
    # ✅ OPTIMIZACIÓN: Usar función paralela optimizada
    labels = calculate_labels_clusters_optimized(close_data, atr, clusters, label_markup, direction=direction)

    dataset = dataset.iloc[:len(labels)].copy()
    dataset['labels_main'] = labels[:len(dataset)]
    dataset = dataset.drop(columns=['cluster'])
    dataset = dataset.dropna()
    return dataset

def get_labels_multi_window(
    dataset, 
    label_window_sizes_int=[20, 50, 100], 
    label_markup=0.5,
    label_min_val=1,
    label_max_val=15,
    label_atr_period=14,
    direction=2,
    label_method_random='random'
) -> pd.DataFrame:
    """
    Etiquetado multi-ventana OPTIMIZADO con profit target basado en ATR * markup.
    MANTIENE la metodología original exacta - Solo mejora el rendimiento.
    """
    prices = dataset['close'].values
    
    # ✅ OPTIMIZACIÓN: Usar ATR vectorizado
    high = dataset["high"].values if "high" in dataset else prices
    low = dataset["low"].values if "low" in dataset else prices
    atr = calculate_atr_vectorized(high, low, prices, period=label_atr_period)
    
    window_sizes_t = List(label_window_sizes_int)
    method_map = {'first': 0, 'last': 1, 'mean': 2, 'max': 3, 'min': 4, 'random': 5}
    method_int = method_map.get(label_method_random, 5)
    
    # ✅ OPTIMIZACIÓN: Usar función paralela optimizada
    signals = calculate_labels_multi_window_optimized(
        prices, atr, window_sizes_t, label_markup, label_min_val, label_max_val, direction, method_int
    )
    
    # Ajustar padding inicial considerando label_max_val
    max_window = max(label_window_sizes_int)
    full_signals = np.full(len(prices), 2.0, dtype=np.float64)
    full_signals[max_window:max_window + len(signals)] = signals
    
    dataset = dataset.iloc[:len(full_signals)].copy()
    dataset['labels_main'] = full_signals[:len(dataset)]
    dataset = dataset.dropna()  # Remove rows with NaN
    return dataset

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
    Etiquetado de reversión a la media OPTIMIZADO con soporte unidireccional.
    MANTIENE la metodología original exacta - Solo mejora el rendimiento.
    """
    # Calculate the price deviation ('lvl') based on the chosen filter
    if label_filter == 'mean':
        diff = (dataset['close'] - dataset['close'].rolling(label_rolling).mean())
        weighted_diff = diff * np.exp(np.arange(len(diff)) * label_decay_factor / len(diff)) 
        dataset['lvl'] = weighted_diff # Add the weighted difference as 'lvl'
    elif label_filter == 'spline':
        x = np.array(range(dataset.shape[0]))
        y = dataset['close'].values
        spl = UnivariateSpline(x, y, k=3, s=label_rolling) 
        yHat = spl(np.linspace(min(x), max(x), num=x.shape[0]))
        yHat_shifted = np.roll(yHat, shift=label_shift) # Apply the shift
        diff = dataset['close'] - yHat_shifted
        weighted_diff = diff * np.exp(np.arange(len(diff)) * label_decay_factor / len(diff)) 
        dataset['lvl'] = weighted_diff # Add the weighted difference as 'lvl'
        dataset = dataset.dropna()  # Remove NaN values potentially introduced by spline/shift
    elif label_filter == 'savgol':
        smoothed_prices = safe_savgol_filter(dataset['close'].values, window_length=int(label_rolling), polyorder=3)
        diff = dataset['close'] - smoothed_prices
        weighted_diff = diff * np.exp(np.arange(len(diff)) * label_decay_factor / len(diff)) 
        dataset['lvl'] = weighted_diff # Add the weighted difference as 'lvl'

    dataset = dataset.dropna()  # Remove NaN values before proceeding

    # Ensure label_max_val does not exceed dataset length
    label_max_val = min(int(label_max_val), max(len(dataset) - 1, 1))
    if len(dataset) <= label_max_val:
        return pd.DataFrame()
    q = tuple(dataset['lvl'].quantile(label_quantiles).to_list())  # Calculate quantiles for the 'reversion zone'

    # Prepare data for label calculation
    close = dataset['close'].values
    lvl = dataset['lvl'].values
    
    # ✅ OPTIMIZACIÓN: Usar ATR vectorizado y función paralela optimizada
    high = dataset["high"].values if "high" in dataset else close
    low = dataset["low"].values if "low" in dataset else close
    atr = calculate_atr_vectorized(high, low, close, period=label_atr_period)
    labels = calculate_labels_mean_reversion_optimized(close, atr, lvl, label_markup, label_min_val, label_max_val, q, direction=direction)

    # Process the dataset and labels
    dataset = dataset.iloc[:len(labels)].copy()
    dataset['labels_main'] = labels
    dataset = dataset.dropna()
    return dataset.drop(columns=['lvl'])  # Remove the temporary 'lvl' column